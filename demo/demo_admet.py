import os, sys
sys.path.append("/home/dataset-local/tyl/projects_dir/Molcular/PASIF-release")
import argparse
import shutil
import subprocess
from copy import deepcopy
from torchvision.transforms import Compose
from torch_geometric.loader import DataLoader
from repo.models import get_model
from repo.models.classifier.classifier import PropPredictor
from repo.utils.misc import *
from repo.utils.molecule.constants import *
from repo.utils.data import recursive_to
from repo.tools.rdkit_utils import *
from repo.datasets.parsers import torchify_dict
from repo.datasets.parsers.protein_parser import PDBProteinFA
from repo.datasets.parsers.molecule_parser import parse_sdf_file
from repo.datasets.transforms.merge import MergeKeys
from repo.datasets.transforms.molecule_featurizer import SetFragGen, RemoveLigand
from repo.datasets.transforms.protein_featurizer import FeaturizeProteinFullAtom, FeaturizeOfftargetFullAtom
from repo.datasets.transforms.init_lig import AssignGenSize, AssignGenType, AssignGenPos, AssignMolSize, AssignMolType, AssignMolPos
from repo.datasets.transforms.translation import CenterPos
from tqdm import tqdm

def split_batch_into_samples(batch, mode='add_aromatic'):
    batch_idx = batch[-1]
    if batch_idx.numel() == 0:
        return []
    B = batch_idx.max() + 1
    batch_split = []
    for i in range(B):
        idx = (batch_idx == i)
        sample = {}
        sample['pos'] = batch[0].cpu()[idx].tolist()
        sample['type'] = batch[1].cpu()[idx].numpy()
        if len(sample['type'].shape) == 2:
            sample['type'] = sample['type'].argmax(axis=-1)
        sample['atom'] = get_atomic_number_from_index(sample['type'], mode)
        sample['aromatic'] = is_aromatic_from_index(sample['type'], mode)
        batch_split.append(sample)
    return batch_split

def translate(result, translation):
    result_pos = result[0].cpu()
    result_pos += translation.cpu()
    return [result_pos] + [result[k+1] for k in range(len(result) - 1)]

parma_dict = {'ppb':     {'model_type': 'reg', 'ckp': '5000.pt', 'opt': 0.8},
              'oatp1b3': {'model_type': 'cls', 'ckp': '150.pt',  'opt': 0.2},
              'oatp1b1': {'model_type': 'cls', 'ckp': '400.pt',  'opt': 0.2},
              'cyp2c19_inh': {'model_type': 'cls', 'ckp': '5000.pt',  'opt': 0.2},}

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--prop', type=str, default='oatp1b3')
    parser.add_argument('--frag', type=str, default='./case/admet/4xli_B_rec_4xli_1n1_lig_tt_min_0.sdf')
    parser.add_argument('--target', type=str, default='./case/admet/4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10.pdb')
    parser.add_argument('--checkpoint', type=str, default='./logs/denovo/diffbp/pretrain/checkpoints/pretrained.pt')
    parser.add_argument('--model_name', type=str, default='diffbp')
    parser.add_argument('--sample_num', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--out_root', type=str, default='./case/admet/output')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--threshold', type=int, default=-1)
    parser.add_argument('--threshold_ratio', type=int, default=0.6)
    args = parser.parse_args()

    seed_all(args.seed)
    model_type = parma_dict[args.prop]['model_type']
    pt_name = parma_dict[args.prop]['ckp']
    cls_ckpt_path = f'./logs/admet/{args.prop}/add_aromatic/self-train/checkpoints/{pt_name}'

    mode = 'add_aromatic'
    distribution = 'absorbing'
    mol_pos_dist = 'gaussian'

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    print(lsd)
    cls_ckpt = torch.load(cls_ckpt_path, map_location='cpu')
    classifier = PropPredictor(cls_ckpt['config']['model'], model_type=model_type).to(args.device)
    classifier.load_state_dict(cls_ckpt['model'])

    if os.path.exists(args.out_root) is False:
        os.makedirs(args.out_root)
    save_dir = os.path.join(args.out_root, args.model_name)
    save_dir = os.path.join(save_dir, args.prop)
    os.makedirs(save_dir, exist_ok=True)
    
    target_dict = PDBProteinFA(args.target).to_dict_atom()
    ligand_dict = parse_sdf_file(args.frag)
    data = EasyDict(
        {'protein': torchify_dict(target_dict),
         'ligand':torchify_dict(ligand_dict),
        }
    )

    transfom_list = [FeaturizeProteinFullAtom(), 
                     RemoveLigand(),
                     CenterPos(center_flag='protein'),
                     AssignMolSize(),
                     AssignMolType(mode=mode, distribution=distribution),
                     AssignMolPos(distribution='gaussian'),
                     MergeKeys(keys=['protein', 'ligand',],
                               excluded_subkeys=['gen_bond_index', 'gen_bond_type', 'bond_index', 
                                                 'bond_type', 'ctx_bond_index', 'ctx_bond_type', 
                                                 'gen_index', 'ctx_index', 'cross_bond_index', 'cross_bond_type'])
                     ]
    transfom_func = Compose(transfom_list)
    data_list_repeat = [transfom_func(deepcopy(data)) for _ in range(args.sample_num)]
    print(f"sample atom num: {[data['ligand_pos'].shape[0] for data in data_list_repeat]}")
    loader = DataLoader(PygDatasetFromList(data_list_repeat), 
                        batch_size=args.batch_size, 
                        shuffle=False,
                        follow_batch = ['protein_element', 'ligand_element']
                        )
    diff_T = cfg_ckpt.model.generator.num_diffusion_timesteps
    count = 0
    for batch in tqdm(loader, desc='specify', dynamic_ncols=True):

        try:
            batch = batch.to(args.device)
        except:
            batch = recursive_to(batch, args.device)

        ts = list(reversed(range(0, diff_T, 1)))
        traj_batch, pred_prop = model.dr_guid(batch, ts=ts, classifier=classifier, just_lig=True, 
                                                    opt_value=parma_dict[args.prop]['opt'])

        result_batch = translate(traj_batch[0], batch.protein_translation[:1])
        result_split = split_batch_into_samples(result_batch, mode=mode)

        basic_mode = True if mode=='basic' else False
        for result in result_split:
            try:
                try:
                    mol = reconstruct_mol(result['pos'], 
                                            result['atom'], 
                                            result['aromatic'], 
                                            basic_mode=basic_mode)
                except:
                    mol = obabel_recover_bond(result['pos'], 
                                                result['atom'])
                    
                mol, success = evaluate_validity(mol, args.threshold, args.threshold_ratio)
                if success:
                    if count >= args.sample_num:
                        break
                    count += 1
                    data = {'pos': np.array(result['pos']),
                            'atom': np.array(result['atom']),
                            'entry': 'specify'}
                    torch.save(data, os.path.join(save_dir, 'sample_%04d.pt' % count))
                    save_mol(mol, os.path.join(save_dir, 'sample_%04d.sdf' % count))
            except:
                continue
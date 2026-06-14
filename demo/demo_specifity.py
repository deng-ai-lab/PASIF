import os, sys
sys.path.append("/home/dataset-local/tyl/projects_dir/Molcular/PASIF-release")
import argparse
import shutil
import subprocess
from copy import deepcopy
from torchvision.transforms import Compose
from torch_geometric.loader import DataLoader
from repo.models import get_model
from repo.models.classifier.classifier import PropPredictor, AffinityPredictor
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
from repo.datasets.transforms.translation import CenterPosSpecify, CenterPosWholeSpecify

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--frag', type=str, default='./case/specificity/docked/docked_9bhk.sdf')
    parser.add_argument('--target', type=str, default='./case/specificity/docked/docked_9bhk_pocket10.pdb')
    parser.add_argument('--off_target', type=str, default='./case/specificity/docked/docked_2buj_pocket10.pdb')
    parser.add_argument('--checkpoint', type=str, default='./logs/denovo/diffbp/pretrain/checkpoints/pretrained.pt')
    parser.add_argument('--classifier', type=str, default='./logs/affinity/add_aromatic/self-train/checkpoints/180000.pt')
    # parser.add_argument('--classifier', type=str, default='./logs/affinity/basic/self-train/checkpoints/98000.pt')
    parser.add_argument('--model_name', type=str, default='diffbp')
    parser.add_argument('--sample_num', type=int, default=50)
    parser.add_argument('--out_root', type=str, default='./case/specificity/output')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:3')
    parser.add_argument('--threshold', type=int, default=-1)
    parser.add_argument('--threshold_ratio', type=int, default=0.6)
    args = parser.parse_args()

    seed_all(args.seed)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    print(lsd)
    cls_ckpt = torch.load(args.classifier, map_location='cpu')
    classifier = PropPredictor(cls_ckpt['config']['model']).to(args.device)
    classifier.load_state_dict(cls_ckpt['model'])

    if os.path.exists(args.out_root) is False:
        os.makedirs(args.out_root)
    
    if args.model_name[:8] == 'diffsbdd':
        mode = 'basic'
        distribution = 'zeros'
        mol_pos_dist = 'zero_mean_gaussian'
    elif args.model_name[:10] == 'targetdiff':
        mode = 'add_aromatic'
        distribution = 'uniform'
        mol_pos_dist = 'gaussian'
    elif args.model_name[:6] == 'diffbp':
        mode = 'add_aromatic'
        distribution = 'absorbing'
        mol_pos_dist = 'gaussian'
    
    target_dict = PDBProteinFA(args.target).to_dict_atom()
    offtarget_dict = PDBProteinFA(args.off_target).to_dict_atom()
    ligand_dict = parse_sdf_file(args.frag)
    data = EasyDict(
        {'protein': torchify_dict(target_dict),
         'offtarget': torchify_dict(offtarget_dict),
         'ligand':torchify_dict(ligand_dict),
        }
    )

    if args.model_name[:8] == 'diffsbdd':
        cen_func = CenterPosWholeSpecify()
    else:
        cen_func = CenterPosSpecify(center_flag='protein')
    transfom_list = [FeaturizeProteinFullAtom(), 
                     FeaturizeOfftargetFullAtom(),
                     RemoveLigand(),
                     AssignMolSize(distribution='prior_distcond'), 
                     AssignMolType(mode=mode, distribution=distribution),
                     cen_func,
                     AssignMolPos(distribution=mol_pos_dist),
                     MergeKeys(keys=['protein', 'ligand', 'offtarget'],
                               excluded_subkeys=['gen_bond_index', 'gen_bond_type', 'bond_index', 
                                                 'bond_type', 'ctx_bond_index', 'ctx_bond_type', 
                                                 'gen_index', 'ctx_index', 'cross_bond_index', 'cross_bond_type'])
                     ]
    transfom_func = Compose(transfom_list)
    data_list_repeat = [transfom_func(deepcopy(data)) for _ in range(args.sample_num)]
    print(f"sample atom num: {[data['ligand_pos'].shape[0] for data in data_list_repeat]}")
    batch_size = 8
    loader = DataLoader(PygDatasetFromList(data_list_repeat), 
                        batch_size=batch_size, 
                        shuffle=False,
                        follow_batch = ['protein_element', 'ligand_element', 'offtarget_element']
                        )
    diff_T = cfg_ckpt.model.generator.num_diffusion_timesteps
    count = 0
    for batch in tqdm(loader, desc='specify', dynamic_ncols=True):

        try:
            batch = batch.to(args.device)
        except:
            batch = recursive_to(batch, args.device)

        ts = list(reversed(range(0, diff_T, 1)))
        save_params = {}
        if len(args.model_name.split('-')) > 1:
            if args.model_name.split('-')[1] == 'comp':
                traj_batch = model.sample(batch)
            else:
                traj_batch = model.specify_guid(batch, ts=ts, classifier=classifier,)
        else:
            traj_batch = model.specify_guid(batch, ts=ts, classifier=classifier,)

        save_dir = os.path.join(args.out_root, args.model_name)
        os.makedirs(save_dir, exist_ok=True)

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
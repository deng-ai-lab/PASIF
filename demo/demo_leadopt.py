import os, sys
sys.path.append("/home/dataset-local/tyl/projects_dir/Molcular/PASIF-release")
import argparse
import copy
import json
import subprocess
from copy import deepcopy
from tqdm.auto import tqdm
from torchvision.transforms import Compose
from torch_geometric.loader import DataLoader
import torch
from torch_scatter import scatter_mean
from repo.datasets.pl import get_pl_dataset
from repo.models import get_model
from repo.utils.misc import *
from repo.utils.molecule.constants import *
from repo.tools.rdkit_utils import reconstruct_mol, evaluate_validity, save_mol, atom_from_fg, obabel_recover_bond
from repo.utils.data import recursive_to
from repo.models.classifier.classifier import PropPredictor
from repo.modules.e3nn.gvptransformer import GVPTransformer
from repo.models.diffusion.sampler import Sampler
from repo.datasets.parsers import torchify_dict
from repo.datasets.parsers.protein_parser import PDBProteinFA
from repo.datasets.parsers.molecule_parser import parse_sdf_file
from repo.datasets.transforms.protein_featurizer import FeaturizeProteinFullAtom
from repo.datasets.transforms.init_lig import AssignMolSizeAround, AssignLeadOptSize, AssignGenSize, AssignGenType, AssignGenPos
from repo.datasets.transforms.translation import CenterPos
from repo.datasets.transforms.merge import MergeKeys


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

def split_batch_into_samples_fg(batch, mode=None):
    batch_idx = batch[-1]
    B = batch_idx.max() + 1
    batch_split = []
    for i in range(B):
        idx = (batch_idx == i)
        sample = {}
        sample['pos_center'] = batch[0].cpu()[idx].tolist()
        sample['fg_type'] = batch[1].cpu()[idx].numpy()
        if len(sample['fg_type'].shape) == 2:
            sample['fg_type'] = sample['fg_type'].argmax(axis=-1)
        sample['orientation'] = batch[2].cpu()[idx].numpy()
        batch_split.append(sample)
    return batch_split

def translate(result, translation):
    result_pos = result[0].cpu()
    result_pos += translation.cpu()
    return [result_pos] + [result[k+1] for k in range(len(result) - 1)]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frag', type=str, default="./case/leadopt/frag.sdf")
    parser.add_argument('--target', type=str, default="./case/leadopt/pocket.pdb")
    parser.add_argument('--checkpoint', type=str, default='./logs/denovo/diffbp/pretrain/checkpoints/pretrained.pt')
    parser.add_argument('--model_name', type=str, default='diffbp')
    parser.add_argument('--sample_num', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--out_root', type=str, default='./case/leadopt/output')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--threshold', type=int, default=-1)
    parser.add_argument('--threshold_ratio', type=int, default=0.6)
    args = parser.parse_args()
    
    seed_all(args.seed)

    save_dir = args.out_root + f'/{args.model_name}'
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    if args.model_name[:8] == 'diffsbdd':
        mode = 'basic'
        distribution = 'gaussian'
    elif args.model_name[:10] == 'targetdiff':
        mode = 'add_aromatic'
        distribution = 'uniform'
    elif args.model_name[:6] == 'diffbp':
        mode = 'add_aromatic'
        distribution = 'absorbing'

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    model = get_model(cfg_ckpt.model).to(args.device)
    lsd = model.load_state_dict(ckpt['model'])
    print(lsd)

    target_dict = PDBProteinFA(args.target).to_dict_atom()
    ligand_dict = parse_sdf_file(args.frag)
    data = EasyDict(
        {'protein': torchify_dict(target_dict),
         'ligand':torchify_dict(ligand_dict),
        }
    )

    transfom_list = [
                     FeaturizeProteinFullAtom(), 
                     AssignLeadOptSize(mode),
                     AssignGenType(mode=mode, distribution=distribution),
                     CenterPos(center_flag='ligand', mask_flag='ctx_flag'),
                     AssignGenPos(distribution='gaussian'),
                     MergeKeys(keys=['protein', 'ligand'],
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
    for batch in tqdm(loader, desc='lead opt', dynamic_ncols=True):

        try:
            batch = batch.to(args.device)
        except:
            batch = recursive_to(batch, args.device)

        ts = list(reversed(range(0, diff_T*1, 1)))
        traj_batch = model.dr_slover(batch, ts=ts)

        if traj_batch is None:
            continue
    
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
                    count += 1
                    data = {'pos': np.array(result['pos']),
                            'atom': np.array(result['atom']),
                            'entry': 'specify'}
                    torch.save(data, os.path.join(save_dir, 'sample_%04d.pt' % count))
                    save_mol(mol, os.path.join(save_dir, 'sample_%04d.sdf' % count))
            except:
                continue


if __name__ == '__main__':
    main()

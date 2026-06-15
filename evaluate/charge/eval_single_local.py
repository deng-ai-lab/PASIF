import os, sys
sys.path.append("/home/dataset-local/tyl/projects_dir/Molcular/PASIF-release")
import numpy as np
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from pyscf import lib

import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_root', type=str, 
                        default="./case/charge/output_local/diffbp")
    parser.add_argument('--ed_path', type=str, 
                        default="./case/charge/dft.npy")
    parser.add_argument('--mask_path', type=str, 
                        default="./case/charge/mask.npy")
    args = parser.parse_args()
    
    eval_root = args.eval_root
    data = np.load(args.ed_path)
    mask = np.load(args.mask_path)

    sdf_list = [f for f in os.listdir(eval_root) 
                if f.endswith('.sdf') and os.path.isfile(os.path.join(eval_root, f))]
    grid_corrds = data[:, :3]
    rho_gt = data[:, -1]
    grid_corrds = grid_corrds[mask]
    rho_gt = rho_gt[mask]
    rho_save = np.zeros_like(rho_gt)

    num = 1
    result_dict = {'file': [], 'mol_num': [], 'q': []}
    for sdf_file in tqdm(sdf_list):

        try:
            mol = Chem.SDMolSupplier(os.path.join(eval_root, sdf_file))[0]
            mol = Chem.AddHs(mol, addCoords=True)
            atom_coords = mol.GetConformer().GetPositions()
            atom_types = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        except:
            print(f'load error occur in {sdf_file}')
            continue
        dist_mat = np.linalg.norm((grid_corrds[:, None, :] - atom_coords[None, :, :]), axis=-1)
        min_mask = dist_mat.min(0)<3.0
        if min_mask.sum().item() < 1.0:
            q = np.array([0.])
        else:
            min_idx = dist_mat.argmin(0)
            rho_select = rho_gt[min_idx]
            q = (rho_select[min_mask] * atom_types[min_mask]).sum() / atom_types[min_mask].sum()

        num += 1

        result_dict['file'].append(sdf_file)
        result_dict['mol_num'].append(Chem.RemoveAllHs(mol).GetNumAtoms())
        result_dict['q'].append(q.item())

    data_df = pd.DataFrame(result_dict)
    data_df.to_csv(os.path.join(eval_root, 'q_local.csv'), index=False)
    print(data_df.loc[:, 'q'].mean())

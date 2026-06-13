import os, sys
sys.path.append("/home/dataset-local/tyl/projects_dir/Molcular/CBGBench-master")
import numpy as np
from rdkit import Chem
import pandas as pd
from tqdm import tqdm
from pyscf import lib
from experiment.density.utils import dft_from_mol, eval_rho

import argparse


if __name__ == '__main__':

    # dataset-local/tyl/projects_dir/Molcular/ED2Mol-main/results/denovo/CHIB1_ASPFM_39_433_0
    # "./results/charge/diffbp-comp/CHIB1_ASPFM_39_433_0"
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_root', type=str, 
                        default="./results/charge/targetdiff/ABL2_HUMAN_274_551_0/4xli_B_rec_4xli_1n1_lig_tt_min_0_pocket10")
    parser.add_argument('--ed_path', type=str, 
                        default="/home/dataset-local/tyl/projects_dir/Molcular/ED2Mol-main/results/ligED/ABL2_HUMAN_274_551_0/ligED.npy")
    args = parser.parse_args()

    eval_num = 5

    eval_root = args.eval_root
    save_path = os.path.join(eval_root, 'density.npy')
    data = np.load(args.ed_path)
    if os.path.exists(save_path):
        density = np.load(save_path, allow_pickle=True).item()
        keys = density.keys()
    else:
        density = {}
        keys = []
    sdf_list = [f for f in os.listdir(eval_root) 
                if f.endswith('.sdf') and os.path.isfile(os.path.join(eval_root, f)) and 'sample' in f]
    grid_corrds = data[:, :3]
    rho_gt = data[:, -1]
    rho_save = np.zeros_like(rho_gt)

    num = 1
    # result_dict = {'file': [], 'mol_num': [], 'q': [], 'nma': [], 'cover': []}
    result_dict = {'file': [], 'mol_num': [], 'q': []}
    for sdf_file in tqdm(sdf_list):
        if num > eval_num:
            print('Eval Enough Molecule!')
            break

        try:
            mol = Chem.SDMolSupplier(os.path.join(eval_root, sdf_file))[0]
            mol = Chem.AddHs(mol, addCoords=True)
            atom_coords = mol.GetConformer().GetPositions()
            atom_types = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
        except:
            print(f'load error occur in {sdf_file}')
            continue

        # if sdf_file in density.keys():
        #     rho = density[sdf_file][:, -1]
        # else:
        #     try:
        #         energy, dm, mol_pyscf = dft_from_mol(mol)
        #         rho = eval_rho(grid_corrds/lib.param.BOHR, mol_pyscf, dm)
        #     except:
        #         print(f'DFT compute error occur in {sdf_file}')
        #         continue
        #     density[sdf_file] = np.concatenate((grid_corrds, rho[:, None]), axis=-1)

        # # NMA
        # scale = rho_gt.sum() / rho.sum()
        # nma = np.abs((scale*rho - rho_gt)).sum() / rho_gt.sum()
        # Q
        dist_mat = np.linalg.norm((grid_corrds[:, None, :] - atom_coords[None, :, :]), axis=-1)
        min_idx = dist_mat.argmin(0)
        rho_select = rho_gt[min_idx]
        q = (rho_select * atom_types).sum() / atom_types.sum()
        # # Cover
        # rho_scale = scale * rho
        # high_scale = rho_scale > np.percentile(rho_scale, 99)
        # high_gt = rho_gt > np.percentile(rho_gt, 99)
        # cover = np.logical_and(high_scale, high_gt).sum() / high_gt.sum()

        # rho_save += rho.reshape(-1)
        num += 1

        result_dict['file'].append(sdf_file)
        result_dict['mol_num'].append(Chem.RemoveAllHs(mol).GetNumAtoms())
        result_dict['q'].append(q.item())
        # result_dict['nma'].append(nma.item())
        # result_dict['cover'].append(cover.item())
    # rho_mean = rho_save / num
    # scale = rho_gt.sum() / rho_mean.sum()
    # nma = np.abs((scale*rho_mean - rho_gt)).sum() / rho_gt.sum()
    # print(nma)
    data_df = pd.DataFrame(result_dict)
    data_df.to_csv(os.path.join(eval_root, 'q_value.csv'), index=False)
    # np.save(save_path, density)
    # print(data_df.loc[:, 'q'].mean(), data_df.loc[:, 'nma'].mean(), data_df.loc[:, 'cover'].mean())
    print(data_df.loc[:, 'q'].mean())


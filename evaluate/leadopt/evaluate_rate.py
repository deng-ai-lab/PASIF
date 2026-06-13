import os, sys
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem, rdMolAlign
import numpy as np
import pandas as pd
from tqdm import tqdm


def get_all_sencond_subfolders(base_path, max_depth=2):
    root_path = base_path.rstrip(os.sep)
    num_sep = root_path.count(os.sep)

    deepest_subfolders = []
    for root, dirs, files in os.walk(base_path):
        current_sep = root.count(os.sep)
        if current_sep - num_sep >= max_depth:
            dirs.clear()
        if not dirs: 
            deepest_subfolders.append(root)
    return deepest_subfolders

def align_molecules_o3a(ref_path, probe_path):
    # 读取分子并加氢（O3A 建议加氢以获得更好的力场描述）
    ref_mol = Chem.SDMolSupplier(ref_path)[0]
    ref_mol = Chem.AddHs(ref_mol, addCoords=True)
    
    probe_mol = Chem.SDMolSupplier(probe_path)[0]
    probe_mol = Chem.AddHs(probe_mol, addCoords=True)

    # 为分子分配 MMFF 属性
    ref_params = AllChem.MMFFGetMoleculeProperties(ref_mol)
    probe_params = AllChem.MMFFGetMoleculeProperties(probe_mol)

    # 获取 O3A 引擎
    
    pyO3A = rdMolAlign.GetO3A(probe_mol, ref_mol, probe_params, ref_params)
    
    # 执行对齐并返回得分/RMSD
    rmsd = pyO3A.Align()
    
    # print(f"O3A 对齐 RMSD: {rmsd:.4f}")
    
    return ref_mol, probe_mol, rmsd

if __name__ == '__main__':

    # tasks = ['linker', 'frag', 'scaffold', 'sidechain']
    test_data_root = './data/crossdocked_test'
    tasks = ['linker', 'frag', 'sidechain', 'scaffold']  # 'linker', 'frag', 'sidechain', 
    rate_dict = {}
    for task in tasks:
        task_rate = 0.
        num = 0
        # eval_root = f'./results/{task}/diffgui/frag_diff-no_bond'
        eval_root = f'./results/{task}/targetdiff/pretrain-inpaint'
        deepest_subfolders = get_all_sencond_subfolders(eval_root)

        div_list = {'file': [], 'div': []}
        for result_path in tqdm(deepest_subfolders, desc='processing'):
            ref_file_name = '/'.join(result_path.split('/')[-2:])[:-8] + task + '.sdf'
            ref_file_path = os.path.join(test_data_root, ref_file_name)
            sdf_files = [f for f in os.listdir(result_path) if f.endswith('.sdf')]
            eval_csv = os.path.join(result_path, 'molecule_properties.csv')

            if len(sdf_files) == 0 or os.path.exists(eval_csv) ==False:
                task_rate += 0
                num += 1
                continue
            eval_df = pd.read_csv(eval_csv)
            eval_df.set_index('file_names', inplace=True)
            if len(eval_df) == 0:
                task_rate += 0
                num += 1
                continue
            ref_dock = eval_df.loc['reference']['vina_dock_result']
            ref_lbe = eval_df.loc['reference']['lbe_result']
            imp_bool = eval_df.iloc[:-1, 2].values < ref_dock
            
            rmsd_result = []
            idx = 0
            for file_name in sdf_files:
                docked_file = 'docked_' + file_name
                docked_file = os.path.join('docking_results', docked_file)
                if docked_file not in eval_df.index:
                    continue
                vina_dock = eval_df.loc[docked_file]['vina_dock_result']

                sdf_path = os.path.join(result_path, file_name)
                if 'rmsd' in eval_df.columns:
                    if np.isnan(eval_df.loc[docked_file, 'rmsd']).item() is False:
                        rmsd = eval_df.loc[docked_file, 'rmsd']
                    else:
                        try:
                            ref_mol, probe_mol, rmsd = align_molecules_o3a(ref_file_path, sdf_path)
                        except:
                            print(f'align error occur in {sdf_path}')
                            rmsd = np.nan
                else:
                    try:
                        ref_mol, probe_mol, rmsd = align_molecules_o3a(ref_file_path, sdf_path)
                    except:
                        print(f'align error occur in {sdf_path}')
                        rmsd = np.nan
                eval_df.loc[docked_file, 'rmsd'] = rmsd
                idx += 1
                if idx == 100:
                    break
            eval_df.to_csv(eval_csv)
            rmsd_values = eval_df.loc[:, 'rmsd'].values[:-1]
            vina_dock = eval_df.loc[:, 'vina_dock_result'].values[:-1]
            lbe = eval_df.loc[:, 'lbe_result'].values[:-1]
            rmsd_mask = np.logical_and(rmsd_values<0.1, ~np.isnan(rmsd_values))
            vina_mask = lbe > 0.35

            now_rate = np.logical_and(rmsd_mask, vina_mask).sum() / 100
            task_rate += now_rate
            num += 1
            
        
        task_rate = task_rate / num
        rate_dict[task] = task_rate
    print(f'{rate_dict}')
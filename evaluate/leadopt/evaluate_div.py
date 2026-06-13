import os, sys
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
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

def cal_div(smiles_list):

    mols = [Chem.MolFromSmiles(s) for s in smiles_list if Chem.MolFromSmiles(s)]
    # 2. 生成 Morgan 指纹 (类似于 ECFP4)
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]

    # 3. 计算所有分子对之间的 Tanimoto 距离 (1 - 相似度)
    n_mols = len(fps)
    dissimilarity_matrix = []

    for i in range(n_mols):
        for j in range(i + 1, n_mols):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            dissimilarity_matrix.append(1 - sim)

    # 4. 输出多样性评分
    avg_diversity = np.mean(dissimilarity_matrix)

    return avg_diversity

def run_one_data(mol_paths):

    smiles_list = []
    for mol_path in mol_paths:
        try:
            mol = Chem.SDMolSupplier(mol_path, sanitize=False)[0]
            smiles = Chem.MolToSmiles(mol)
        except:
            print(f'read error occur in {mol_path}')
        smiles_list.append(smiles)
    
    avg_diversity = cal_div(smiles_list)

    return avg_diversity

if __name__ == '__main__':

    # tasks = ['linker', 'frag', 'scaffold', 'sidechain']
    tasks = ['linker', 'frag', 'scaffold', 'sidechain']
    for task in tasks:
        # eval_root = f'./results/{task}/targetdiff/pretrain-inpaint'
        eval_root = f'./results/{task}/diffgui/frag_diff-no_bond'
        deepest_subfolders = get_all_sencond_subfolders(eval_root)

        # data_df = pd.read_csv(os.path.join(eval_root, 'div.csv'))
        # print(f'{task}-{data_df.iloc[-1, 1]}')

        div_list = {'file': [], 'div': []}
        for result_path in tqdm(deepest_subfolders, desc='processing'):

            sdf_files = [os.path.join(result_path, f) for f in os.listdir(result_path) if f.endswith('.sdf')]
            if len(sdf_files) == 0:
                continue
            avg_div = run_one_data(sdf_files)

            div_list['file'].append('/'.join(result_path.split('/')[-2:]))
            div_list['div'].append(avg_div)
        
        div_array = np.array(div_list['div'])
        div_array = div_array[~np.isnan(div_array)]
        div_list['file'].append('final')
        div_list['div'].append(div_array.mean().item())
        data_df = pd.DataFrame(div_list)
        data_df = data_df.round({'div': 2})
        data_df.to_csv(os.path.join(eval_root, 'div.csv'), index=False)
        print(1)
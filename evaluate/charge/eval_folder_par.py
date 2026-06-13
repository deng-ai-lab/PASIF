import os, sys
sys.path.append("/home/dataset-local/tyl/projects_dir/Molcular/CBGBench-master")
from pathlib import Path
import pandas as pd
import numpy as np
from evaluate.charge.eval_single_par import calculate_overlap

if __name__ == '__main__':

    ref_folder = Path('./data/charge_test')
    mol_folder = Path('./results/par/diffsbdd/denovo')
    out_path = mol_folder / 'par.csv'

    root = Path(mol_folder)
    
    data_dict = {'name': [], 'rate': [], 'ref_num': [], 'gen_num': []}
    for file_path in root.glob('*.phore'):
        
        ref_path = ref_folder / file_path.name[:-6] 
        ref_path = ref_path / file_path.name
        mol_path = mol_folder / f'{file_path.name}'

        rate, ref_num, gen_num = calculate_overlap(ref_file=ref_path, mol_file=mol_path)

        if rate is None:
            continue

        data_dict['name'].append(file_path.name[:-6])
        data_dict['rate'].append(rate)
        data_dict['ref_num'].append(ref_num)
        data_dict['gen_num'].append(gen_num)

    data_df = pd.DataFrame(data_dict)
    data_df.to_csv(out_path, index=False)

    res = np.zeros([12])
    samplenum = np.zeros([12])
    gennum = np.zeros([12])
    for i in range(1, 12):
        if i <= 2:
            mask = data_df['ref_num'] <= i
        elif i >= 11:
            mask = data_df['ref_num'] >= i
        else:
            mask = data_df['ref_num'] == i
        now_rate = data_df['rate'][mask].mean()
        res[i-1] = now_rate
        samplenum[i-1] = mask.sum()
        gennum[i-1] = data_df['gen_num'][mask].mean()
    
    print(res)
    print(samplenum)
    print(gennum)
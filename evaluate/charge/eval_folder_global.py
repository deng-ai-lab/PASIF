import os
import argparse
import subprocess
import joblib
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count
from pathlib import Path

def get_second_level_dirs(root_path):
    # 将传入的字符串路径转换为 Path 对象
    root = Path(root_path)
    
    # 确保根路径存在且是一个目录
    if not root.is_dir():
        return []
    
    # root.glob('*/*') 会匹配：root下的任意一层/再下一层
    # p.is_dir() 确保匹配到的路径是一个文件夹，而不是文件
    second_level_dirs = [str(p) for p in root.glob('*/*') if p.is_dir()]
    
    return second_level_dirs

def get_all_deepest_subfolders(base_path):
    deepest_subfolders = []
    for root, dirs, files in os.walk(base_path):
        if not dirs:  
            deepest_subfolders.append(root)
    return deepest_subfolders

def run_evaluation(args):
    result_path, base_result_path, base_pdb_path, ed_path = args
    try:
        relative_path = os.path.relpath(result_path, base_result_path)
        if relative_path[-3:] == 'pdb':
            pdb_sub_path = os.path.join(base_pdb_path, relative_path)
        else:
            pdb_sub_path = os.path.join(base_pdb_path, relative_path + ".pdb")
        # if os.path.exists(os.path.join(result_path, 'molecule_properties.csv')):
        #     print(f"Skipping {result_path}, already evaluated.")
        #     return
        if os.path.exists(os.path.join(result_path, 'sample_0001.sdf')) is False:
            print(f"No design sample {result_path}")
            return 
        if os.path.exists(pdb_sub_path) and '/'.join(result_path.split('/')[-2:]) == relative_path:
            print(f"Processing {result_path} with PDB {pdb_sub_path}")

            cmd = [
                "python", "./evaluate/charge/eval_single_global.py",
                "--eval_root", result_path,
                "--ed_path", ed_path,
            ]
            subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error processing {result_path}: {e}")

def main(base_result_path, base_pdb_path, base_ed_path):
    # deepest_subfolders = get_all_deepest_subfolders(base_result_path)
    deepest_subfolders = get_second_level_dirs(base_result_path)

    nthreads = multiprocessing.cpu_count()
    nthreads = nthreads // 4  # Limit to 1/4 of available CPU cores
    print("Number of CPU cores:", nthreads)

    args_list = []
    for result_path in deepest_subfolders:            
        if 'docking_results' in result_path:
            result_path = os.path.dirname(result_path)
        if '.ipynb_checkpoints' in result_path:
            continue
            #result_path = os.path.dirname(result_path)
        dir_name = result_path.split('/')[-2]
        ed_path = os.path.join(base_ed_path, dir_name)
        ed_path = os.path.join(ed_path, 'ligED.npy')
        args_list.append((result_path, base_result_path, base_pdb_path, ed_path,))

    res_list = joblib.Parallel(
            n_jobs=1,
        )(
            joblib.delayed(run_evaluation)(args_idx)
            for args_idx in tqdm(args_list, dynamic_ncols=True, desc='Preprocessing...')
        )


    # with Pool(processes=nthreads) as pool:
    #     for _ in tqdm(pool.imap(run_evaluation, args_list), total=len(args_list)):
    #         pass

    print('evaluation done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_result_path', type=str, default='./results/denovo/diffsbdd/pretrain-sample', help="Base result path to traverse")
    parser.add_argument('--base_pdb_path', type=str, default='./data/crossdocked_test/', help="Base PDB path for constructing pdb_path")
    parser.add_argument('--base_ed_path', type=str, default='/home/dataset-local/tyl/projects_dir/Molcular/ED2Mol-main/results/ligED', help="Base PDB path for constructing pdb_path")
    # parser.add_argument('--exhaustiveness', type=int, default=16, help="Exhaustiveness parameter for Vina docking")
    # parser.add_argument('--eval_ref', type=bool, default=True, help="Whether to evaluate the reference ligand")
    # parser.add_argument('--verbose', type=eval, default=False, help="Verbose output")
    args = parser.parse_args()

    main(args.base_result_path, args.base_pdb_path, args.base_ed_path)
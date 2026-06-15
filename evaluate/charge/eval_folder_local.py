import os
import argparse
import subprocess
import joblib
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count

def get_all_deepest_subfolders(base_path):
    deepest_subfolders = []
    for root, dirs, files in os.walk(base_path):
        if not dirs:  
            deepest_subfolders.append(root)
    return deepest_subfolders

def run_evaluation(args):
    result_path, base_result_path, base_pdb_path, ed_path, mask_path = args
    try:
        relative_path = os.path.relpath(result_path, base_result_path)
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
                "python", "./evaluate/charge/eval_single_local.py",
                "--eval_root", result_path,
                "--ed_path", ed_path,
                "--mask_path", mask_path
            ]
            subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error processing {result_path}: {e}")

def main(base_result_path, base_pdb_path, base_ed_path):
    deepest_subfolders = get_all_deepest_subfolders(base_result_path)

    nthreads = multiprocessing.cpu_count()
    nthreads = nthreads // 4  # Limit to 1/4 of available CPU cores
    print("Number of CPU cores:", nthreads)

    args_list = []
    for result_path in deepest_subfolders:
        if 'docking_results' in result_path:
            result_path = os.path.dirname(result_path)
        if '.ipynb_checkpoints' in result_path:
            result_path = os.path.dirname(result_path)
        if os.path.exists(os.path.join(result_path, 'sample_0001.sdf')) is False:
            print(f"No design sample {result_path}")
            continue 
        dir_name = result_path.split('/')[-2]
        ed_path = os.path.join(base_ed_path, dir_name)
        ed_path = os.path.join(ed_path, 'dft.npy')
        mask_path = os.path.join(base_ed_path, dir_name)
        mask_path = os.path.join(mask_path, 'mask.npy')
        if os.path.exists(ed_path) == False or os.path.exists(mask_path) == False:
            continue
        args_list.append((result_path, base_result_path, base_pdb_path, ed_path, mask_path))

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
    parser.add_argument('--base_ed_path', type=str, default='./data/charge_test', help="Base PDB path for constructing pdb_path")
    # parser.add_argument('--exhaustiveness', type=int, default=16, help="Exhaustiveness parameter for Vina docking")
    # parser.add_argument('--eval_ref', type=bool, default=True, help="Whether to evaluate the reference ligand")
    # parser.add_argument('--verbose', type=eval, default=False, help="Verbose output")
    args = parser.parse_args()

    main(args.base_result_path, args.base_pdb_path, args.base_ed_path)
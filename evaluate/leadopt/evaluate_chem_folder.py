import os, sys
sys.path.append("/home/dataset-local/tyl/projects_dir/Molcular/PASIF-release")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
    result_path, base_result_path, base_pdb_path, exhaustiveness, eval_ref, verbose = args
    try:
        relative_path = os.path.relpath(result_path, base_result_path)
        pdb_sub_path = os.path.join(base_pdb_path, relative_path + ".pdb")
        if os.path.exists(os.path.join(result_path, 'sample_0001.sdf')) is False:
            print(f"No design sample {result_path}")
            return 
        if os.path.exists(pdb_sub_path) and '/'.join(result_path.split('/')[-2:]) == relative_path:
            print(f"Processing {result_path} with PDB {pdb_sub_path}")

            cmd = [
                "python", "./evaluate/leadopt/evaluate_chem_single.py",
                "--result_path", result_path,
                "--pdb_path", pdb_sub_path,
                "--exhaustiveness", str(exhaustiveness),
                "--eval_ref", str(eval_ref),
                "--verbose", str(verbose)
            ]
            subprocess.run(cmd, check=True)
    except Exception as e:
        print(f"Error processing {result_path}: {e}")

def main(base_result_path, base_pdb_path, exhaustiveness, eval_ref, verbose):
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
        args_list.append((result_path, base_result_path, base_pdb_path, exhaustiveness, eval_ref, verbose))
    
    res_list = joblib.Parallel(
            n_jobs=1,
        )(
            joblib.delayed(run_evaluation)(args_idx)
            for args_idx in tqdm(args_list, dynamic_ncols=True, desc='Preprocessing...')
        )

    print('evaluation done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_result_path', type=str, default='./results/scaffold/targetdiff/pretrain-dr_slover', help="Base result path to traverse")
    parser.add_argument('--base_pdb_path', type=str, default='./data/crossdocked_test', help="Base PDB path for constructing pdb_path")
    parser.add_argument('--exhaustiveness', type=int, default=16, help="Exhaustiveness parameter for Vina docking")
    parser.add_argument('--eval_ref', type=bool, default=True, help="Whether to evaluate the reference ligand")
    parser.add_argument('--verbose', type=eval, default=False, help="Verbose output")
    args = parser.parse_args()

    main(args.base_result_path, args.base_pdb_path, args.exhaustiveness, args.eval_ref, args.verbose)
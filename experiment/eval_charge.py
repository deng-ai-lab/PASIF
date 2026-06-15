import os, sys
sys.path.append("/home/dataset-local/tyl/projects_dir/Molcular/PASIF-release")
import torch
import subprocess
import configparser

import joblib
from tqdm import tqdm

def GetConfigs(config_path):
    configs = configparser.ConfigParser()
    configs.read(config_path)
    return configs

def process_one(test_turple):

    pocket = test_turple[0]
    file_name = pocket.split('/')[0]
    sdf_file = pocket.split('/')[1]
    ligand = test_turple[1]

    data_root_local = os.path.join(data_root, 'crossdocked_test')
    pocket_path = os.path.join(data_root_local, pocket)
    mol_path = os.path.join(data_root_local, ligand)

    ligED_path = os.path.join(data_root, 'electron')
    ligED_path = os.path.join(ligED_path, f'{file_name}/ligED-{sdf_file[:-4]}.npy')
    if os.path.exists(ligED_path) is False:
        print(f'{file_name} should be processed!')
        return 1

    out_path = os.path.join(f'./results/charge/{model}', pocket[:-4])
    if os.path.exists(os.path.join(out_path, 'sample_0001.sdf')):
        print(f'{pocket[:-4]} already be processed!')
        return 1

    if model == 'diffgui':
        py_file = "./experiment/charge_global_diffgui.py"
    else:
        py_file = "./experiment/charge_global.py"

    # if model == 'diffgui':
    #     cmd = [
    #     "python", py_file,
    #     "--density_path", ligED_path,
    #     "--frag", mol_path,
    #     "--target", pocket_path,
    #     "--checkpoint", f'./logs/denovo/{model}/pretrain/checkpoints/pretrained.pt',
    #     '--model_name', model,
    #     '--device', device,
    #     '--out_root', './results/charge'
    # ]
    # else:

    cmd = [
        "python", py_file,
        "--density_path", ligED_path,
        "--frag", mol_path,
        "--target", pocket_path,
        "--checkpoint", f'./logs/denovo/{model}/pretrain/checkpoints/pretrained.pt',
        '--model_name', model,
        '--device', device,
        '--out_root', './results/charge'
    ]

    subprocess.run(cmd, check=True)


if __name__ == '__main__':

    # assign mol size !!!!!
    model = 'diffbp'
    device = 'cuda:0'
    split_path = torch.load(f"./data/split_by_name_10m.pt")

    data_root = "./data/"

    test_list = split_path['test']

    res_list = joblib.Parallel(
            n_jobs=1,
        )(
            joblib.delayed(process_one)(test_turple)
            for test_turple in tqdm(test_list, dynamic_ncols=True, desc='Preprocessing...')
        )







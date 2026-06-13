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

    if use_protein :
        data_root_local = os.path.join(data_root, 'crossdocked_v1.1_rmsd1.0')
        pocket_path = os.path.join(data_root_local, file_name)
        pocket_path = os.path.join(pocket_path, sdf_file[:10]+'.pdb')
        mol_path = os.path.join(data_root_local, ligand)
        output_dir = f'/home/dataset-local/tyl/projects_dir/Molcular/ED2Mol-main/results/ligED/protein/{file_name}'
    else:
        data_root_local = os.path.join(data_root, 'crossdocked_test')
        pocket_path = os.path.join(data_root_local, pocket)
        mol_path = os.path.join(data_root_local, ligand)
        output_dir = f'/home/dataset-local/tyl/projects_dir/Molcular/ED2Mol-main/results/ligED/{file_name}'
    
    ligED_path = os.path.join(output_dir, 'ligED.npy')
    if os.path.exists(ligED_path) is False:
        print(f'{file_name} should be processed!')
        return 1

    out_path = os.path.join(f'./results/charge/{model}', pocket[:-4])
    if os.path.exists(os.path.join(out_path, 'sample_0001.sdf')):
        print(f'{pocket[:-4]} already be processed!')
        return 1

    if model == 'diffgui':
        cmd = [
        "python", "./experiment/charge_opt_vae_diffgui.py",
        "--density_path", ligED_path,
        "--frag", mol_path,
        "--target", pocket_path,
        "--checkpoint", f'./logs/denovo/{model}/pretrain/checkpoints/pretrained.pt',
        '--model_name', model,
        '--device', device,
        '--out_root', './results/charge'
    ]
    else:

        cmd = [
            "python", "./experiment/charge_opt_vae.py",
            "--density_path", ligED_path,
            "--frag", mol_path,
            "--target", pocket_path,
            "--checkpoint", f'./logs/denovo/{model}/pretrain/checkpoints/pretrained.pt',
            '--model_name', model,
            '--device', device
        ]

    subprocess.run(cmd, check=True)


if __name__ == '__main__':

    # assign mol size !!!!!
    model = 'diffsbdd'
    device = 'cuda:1'
    split_path = torch.load(f"./data/split_by_name_10m.pt")

    use_protein = False
    if use_protein:
        data_root = "./raw_data/"
    else:
        data_root = "./data/"

    test_list = split_path['test']

    res_list = joblib.Parallel(
            n_jobs=1,
        )(
            joblib.delayed(process_one)(test_turple)
            for test_turple in tqdm(test_list, dynamic_ncols=True, desc='Preprocessing...')
        )







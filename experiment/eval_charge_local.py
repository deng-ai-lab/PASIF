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
    
    ligED_path = os.path.join('./data/charge_test/', file_name)
    ligED_path = os.path.join(ligED_path, 'dft.npy')
    mask_path = os.path.join('./data/charge_test/', file_name)
    mask_path = os.path.join(mask_path, 'mask.npy')
    if os.path.exists(ligED_path) is False:
        print(f'{file_name} should be processed!')
        return 1

    if os.path.exists(f'./results/charge_local/{model}/{pocket[:-4]}/sample_0001.sdf'):
        print(f'{file_name} already processed!')
        return 1
    
    # configs.set('sample', 'receptor', pocket_path)
    # configs.set('sample', 'output_dir', output_dir)
    # tmp_path = f'./tmp/{file_name}.yml'

    # with open(tmp_path, 'w') as f:
    #     configs.write(f)
    if model == 'diffgui':
        cmd = [
        "python", "./experiment/charge_opt_local_diffgui.py",
        "--density_path", ligED_path,
        "--target", pocket_path,
        "--frag", mol_path,
        "--mask", mask_path,
        "--checkpoint", f'./logs/denovo/{model}/pretrain/checkpoints/pretrained.pt',
        '--model_name', model,
        '--device', device
    ]
    else:
        cmd = [
            "python", "./experiment/charge_opt_local.py",
            "--density_path", ligED_path,
            "--target", pocket_path,
            "--frag", mol_path,
            "--mask", mask_path,
            "--checkpoint", f'./logs/denovo/{model}/pretrain/checkpoints/pretrained.pt',
            '--model_name', model,
            '--device', device
        ]

    subprocess.run(cmd, check=True)


if __name__ == '__main__':

    # assign mol size !!!!!
    model = 'diffsbdd'
    device = 'cuda:2'
    # config_path = './configs/ligED.yml'
    split_path = torch.load(f"./data/split_by_name_10m.pt")

    use_protein = False
    if use_protein:
        data_root = "./raw_data/"
    else:
        data_root = "./data/"

    # configs = GetConfigs(config_path)

    test_list = split_path['test']

    res_list = joblib.Parallel(
            n_jobs=1,
        )(
            joblib.delayed(process_one)(test_turple)
            for test_turple in tqdm(test_list, dynamic_ncols=True, desc='Preprocessing...')
        )

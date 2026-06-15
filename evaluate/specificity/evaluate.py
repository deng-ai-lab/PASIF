import os, sys
sys.path.append("/home/dataset-local/tyl/projects_dir/Molcular/PASIF-release")
import subprocess
import shutil
import AutoDockTools
import numpy as np
import pandas as pd
from vina import Vina
from openbabel import pybel
from repo.tools import scoring
from rdkit import Chem
from rdkit.Chem import Crippen, QED
from rdkit.Chem import rdMolDescriptors
import uuid

def parse_coords(pdbqt_file):
        """
        从PDBQT文件中提取ATOM和HETATM的坐标。
        """
        coords = []
        with open(pdbqt_file, 'r') as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    # PDBQT格式固定：30-38是X, 38-46是Y, 46-54是Z
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                    except ValueError:
                        continue
        return np.array(coords)

def shift_ligand_to_protein(ligand_file, protein_file, output_ligand_file):

        lig_coords = parse_coords(ligand_file)
        prot_coords = parse_coords(protein_file)
        
        if len(lig_coords) == 0 or len(prot_coords) == 0:
            raise ValueError("无法从输入文件中提取原子坐标，请检查PDBQT格式。")

        lig_center = lig_coords.mean(0)
        prot_center = prot_coords.mean(0)
        
        translation_vector = prot_center - lig_center
        
        with open(ligand_file, 'r') as f_in, open(output_ligand_file, 'w') as f_out:
            for line in f_in:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    orig_x = float(line[30:38])
                    orig_y = float(line[38:46])
                    orig_z = float(line[46:54])
                    
                    new_x = orig_x + translation_vector[0]
                    new_y = orig_y + translation_vector[1]
                    new_z = orig_z + translation_vector[2]
                    
                    new_line = (
                        line[:30] + 
                        f"{new_x:8.3f}{new_y:8.3f}{new_z:8.3f}" + 
                        line[54:]
                    )
                    f_out.write(new_line)
                else:
                    f_out.write(line)
        
        return prot_center

def get_box(pdb_path):
    with open(pdb_path, 'r') as f: 
        lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
        xs = [float(l[31:39]) for l in lines]
        ys = [float(l[39:47]) for l in lines]
        zs = [float(l[47:55]) for l in lines]
        pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
        box_size = [(max(xs) - min(xs)), (max(ys) - min(ys)), (max(zs) - min(zs))]
        return pocket_center, box_size

def prepare_lig(sdf_path, output_path):
    pdb_path = output_path[:-5] + '.pdb'
    mol = next(pybel.readfile("sdf", sdf_path))
    mol.addh()
    mol.write("pdb", pdb_path, overwrite=True)

    prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_ligand4.py')
    subprocess.Popen(['python3', prepare_receptor, '-l', pdb_path, '-o', output_path],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

def prepare_rec(pdb_path, output_path):
    prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
    subprocess.Popen(['python3', prepare_receptor, '-r', pdb_path, '-o', output_path],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

def trans_pdbqt2pdb(pdb_path, pdbqt_path):
    mol = next(pybel.readfile("pdbqt", pdbqt_path))
    mol.write("pdb", pdb_path, overwrite=True)

def trans_pdbqt2sdf(pdbqt_path, sdf_path):
    subprocess.run(["obabel", "-ipdbqt", pdbqt_path, "-osdf", "-O", sdf_path])

def get_value(df, row, col):
    try:
        return df.loc[row][col]
    except KeyError:
        return float('nan')

if __name__ == '__main__':

    name = 'MerTK'
    target = '9bhk'    
    off_target = '2buj'
    model = 'diffbp'
    protein_path = f"./data/specificity/{name}/{target}_protein.pdb"
    protein_pdbqt = protein_path[:-4] + '.pdbqt'
    protein_path_off = f"./data/specificity/{name}/{off_target}_protein.pdb"
    protein_pdbqt_off = protein_path_off[:-4] + '.pdbqt'
    docked_root = f"./results/specificity/{name}-{target}-{off_target}/{model}/docked_result"
    docked_root_off = f"./results/specificity/{name}-{target}-{off_target}/{model}/off_docked_result"

    result_path = f'./results/specificity/{name}-{target}-{off_target}/{model}'
    sdf_files = [f
                 for f in sorted(os.listdir(result_path)) if f.endswith(".sdf")]
    
    shift = False
    
    temp_path = os.path.join(result_path, 'temp')
    os.makedirs(temp_path, exist_ok=True)
    os.makedirs(docked_root, exist_ok=True)
    os.makedirs(docked_root_off, exist_ok=True)
    
    if os.path.exists(protein_pdbqt) is False:
        prepare_rec(protein_path, protein_pdbqt)
    if os.path.exists(protein_pdbqt_off) is False:
        prepare_rec(protein_path_off, protein_pdbqt_off)
    
    result_dict = {'file': [],
                   'vina score': [], 'vina optimize': [], 'vina dock': [],
                   'off vina score': [], 'off vina optimize': [], 'off vina dock': [], 
                   'qed': [], 'sa': [], 'logp': [], 'lipinski': [], 'specify': [], 
                   }
    csv_path = os.path.join(result_path, 'molecule_properties_protein.csv')
    property_csv_flag = os.path.exists(csv_path)
    if property_csv_flag:
        property_df = pd.read_csv(csv_path)
        property_df.set_index('file', inplace=True)
        property_df[[k for k in ['vina score', 'vina optimize', 'vina dock', 
                                 'off vina score', 'off vina optimize', 'off vina dock', 
                                 'qed', 'sa', 'logp', 'lipinski', 'specify']]] = \
        property_df[[k for k in ['vina score', 'vina optimize', 'vina dock', 
                                 'off vina score', 'off vina optimize', 'off vina dock', 
                                 'qed', 'sa', 'logp', 'lipinski', 'specify']]].apply(pd.to_numeric)
    for f in sdf_files:
        ligand_path = os.path.join(result_path, f)
        flag = False
        if property_csv_flag:
            if f[:-4] in property_df.index:
                result_dict['file'].append(f[:-4])
                result_dict['qed'].append(property_df.loc[f[:-4]]['qed'])
                result_dict['logp'].append(property_df.loc[f[:-4]]['logp'])
                result_dict['sa'].append(property_df.loc[f[:-4]]['sa'])
                result_dict['lipinski'].append(property_df.loc[f[:-4]]['lipinski'])

                result_dict['vina score'].append(property_df.loc[f[:-4]]['vina score'])
                result_dict['vina optimize'].append(property_df.loc[f[:-4]]['vina optimize'])
                result_dict['vina dock'].append(property_df.loc[f[:-4]]['vina dock'])
                result_dict['off vina score'].append(property_df.loc[f[:-4]].get('off vina score', float('nan')))
                result_dict['off vina optimize'].append(property_df.loc[f[:-4]].get('off vina optimize', float('nan')))
                result_dict['off vina dock'].append(property_df.loc[f[:-4]].get('off vina dock', float('nan')))
                result_dict['specify'].append(property_df.loc[f[:-4]].get('specify', float('nan')))
                flag = True
        if flag is False:
            try:
                mol = Chem.SDMolSupplier(ligand_path)[0]
                chem_results = scoring.get_chem(mol)

                tmp_file = uuid.uuid4().hex
                ligand_pdbqt = os.path.join(temp_path, f"{tmp_file}.pdbqt")
                shift_lig_pdbqt = ligand_pdbqt[:-6] + '-shift.pdbqt'
                docked_pdbqt = os.path.join(temp_path, f"docked_{tmp_file}.pdbqt")
                docked_path = os.path.join(docked_root, 'docked_'+f)
                if os.path.exists(ligand_pdbqt) is False:
                    prepare_lig(ligand_path, ligand_pdbqt)
                if shift:
                    shift_ligand_to_protein(ligand_pdbqt, protein_pdbqt, shift_lig_pdbqt)
                else:
                    shift_lig_pdbqt = ligand_pdbqt
                v = Vina(sf_name='vina', verbosity=True)
                v.set_receptor(protein_pdbqt)
                v.set_ligand_from_file(shift_lig_pdbqt)
                center, box_size = get_box(protein_path)
                v.compute_vina_maps(center=center, box_size=box_size)
                score_result = v.score()
                optimize_result = v.optimize()
                v.dock(exhaustiveness=16, n_poses=20)
                dock_result = v.score()
                v.write_poses(docked_pdbqt, n_poses=1, overwrite=True)

                trans_pdbqt2sdf(docked_pdbqt, docked_path)
                
                if protein_path_off is not None:
                    docked_pdbqt = os.path.join(temp_path, f"off_docked_{tmp_file}.pdbqt")
                    docked_path = os.path.join(docked_root, 'off_docked_'+f)

                    if shift:
                        shift_ligand_to_protein(ligand_pdbqt, protein_pdbqt_off, shift_lig_pdbqt)
                    else:
                        shift_lig_pdbqt = ligand_pdbqt

                    v = Vina(sf_name='vina', verbosity=True)
                    v.set_receptor(protein_pdbqt_off)
                    v.set_ligand_from_file(shift_lig_pdbqt)
                    center, box_size = get_box(protein_path_off)
                    v.compute_vina_maps(center=center, box_size=box_size)
                    try:
                        off_score_result = v.score()
                        off_optimize_result = v.optimize()
                    except:
                        off_score_result = np.array([0])
                        off_optimize_result = np.array([0])
                    v.dock(exhaustiveness=16, n_poses=20)
                    off_dock_result = v.score()
                    v.write_poses(docked_pdbqt, n_poses=1, overwrite=True)
                    trans_pdbqt2sdf(docked_pdbqt, docked_path)

                    off_score = off_score_result[0].item()
                    off_opt_score = off_optimize_result[0].item()
                    off_dock_score = off_dock_result[0].item()
                    specify = dock_result[0].item()/off_dock_result[0].item()
                    
                else:
                    off_score = float('nan')
                    off_opt_score = float('nan')
                    off_dock_score = float('nan')
                    specify = float('nan')
            except:
                print(f'Error occur when processing {f}')
                continue
            result_dict['file'].append(f[:-4])
            result_dict['qed'].append(chem_results['qed'])
            result_dict['logp'].append(chem_results['logp'])
            result_dict['sa'].append(chem_results['sa'])
            result_dict['lipinski'].append(chem_results['lipinski'])

            result_dict['vina score'].append(score_result[0].item())
            result_dict['vina optimize'].append(optimize_result[0].item())
            result_dict['vina dock'].append(dock_result[0].item())

            result_dict['off vina score'].append(off_score)
            result_dict['off vina optimize'].append(off_opt_score)
            result_dict['off vina dock'].append(off_dock_score)
            result_dict['specify'].append(specify)
    
    df = pd.DataFrame(result_dict)
    df.to_csv(csv_path, index=False)
    shutil.rmtree(temp_path)    
    print(1)

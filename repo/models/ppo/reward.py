import os, sys
sys.path.append("/home/lfj/projects_dir/tyl/Molcular/CBGBench-master")
import torch
import subprocess
import shutil
import AutoDockTools
import numpy as np
import pandas as pd
from vina import Vina
from openbabel import pybel
from repo.tools import scoring
from rdkit import Chem
import uuid
from repo.utils.molecule.constants import *
from repo.tools.rdkit_utils import reconstruct_mol, evaluate_validity, save_mol, atom_from_fg, obabel_recover_bond


def get_box(pdb_path):
    with open(pdb_path, 'r') as f: 
        lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
        xs = [float(l[31:39]) for l in lines]
        ys = [float(l[39:47]) for l in lines]
        zs = [float(l[47:55]) for l in lines]
        # print(max(xs), min(xs))
        # print(max(ys), min(ys))
        # print(max(zs), min(zs))
        pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
        box_size = [(max(xs) - min(xs)), (max(ys) - min(ys)), (max(zs) - min(zs))]
        return pocket_center, box_size

def prepare_lig(sdf_path, output_path):
    pdb_path = output_path[:-5] + 'pdb'
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


class Reward(object):
    def __init__(self, tmp_root, mode, threshold_ratio=0.8, data_roots=['./raw_data/crossdocked_v1.1_rmsd1.0', './raw_data/crossdocked_v1.1_rmsd1.0_pocket10']):
        
        self.tmp_root = tmp_root
        self.mode = mode
        self.threshold_ratio = threshold_ratio
        self.data_roots = data_roots

    def translate(self, result, translation):
        result_pos = result[0].cpu()
        result_pos += translation.cpu()
        return [result_pos] + [result[k+1] for k in range(len(result) - 1)]

    def split_batch_into_samples(self, batch):
        batch_idx = batch[-1]
        if batch_idx.numel() == 0:
            return []
        B = batch_idx.max() + 1
        batch_split = []
        for i in range(B):
            idx = (batch_idx == i)
            sample = {}
            sample['pos'] = batch[0].cpu()[idx].tolist()
            sample['type'] = batch[1].cpu()[idx].numpy()
            if len(sample['type'].shape) == 2:
                sample['type'] = sample['type'].argmax(axis=-1)
            sample['atom'] = get_atomic_number_from_index(sample['type'], self.mode)
            sample['aromatic'] = is_aromatic_from_index(sample['type'], self.mode)
            sample['file'] = f"{uuid.uuid4().hex}.sdf"
            batch_split.append(sample)
        return batch_split

    def save_mol(self, result_split):
        save_result = []
        basic_mode = True if self.mode=='basic' else False
        for i, result in enumerate(result_split):
            save_path = os.path.join(self.tmp_root, result['file'])
            try:
                try:
                    mol = reconstruct_mol(result['pos'], 
                                          result['atom'], 
                                          result['aromatic'], 
                                          basic_mode=basic_mode)
                except:
                    mol = obabel_recover_bond(result['pos'], 
                                              result['atom'])
                    
                mol, success = evaluate_validity(mol, -1, self.threshold_ratio)
                if success:
                    save_mol(mol, save_path)
                    success_flag = True
                else:
                    success_flag = False
            except:
                success_flag = False
            save_result.append((save_path, success_flag))
        return save_result
    
    def preprocess(self, traj_batch):
        batch_split = self.split_batch_into_samples(traj_batch)
        save_result = self.save_mol(batch_split)
        return save_result
    
    def vina_eval(self, save_result, protein_list, ligand_list):
        """

        return: [(ref_score, score)]
        """

        ref_score_list = []
        score_list = []
        success_list = []
        for i in range(len(save_result)):
            success_flag = save_result[i][1]
            if success_flag:
                
                pocket_files = protein_list[i].split('/')
                protein_name = pocket_files[1].split('_')[:3]
                protein_name = '_'.join(protein_name)
                pocket_path = os.path.join(self.data_roots[1], protein_list[i])
                all_protein_path = os.path.join(self.data_roots[0], pocket_files[0]+'/'+protein_name+'.pdb')   # in data_root
                if os.path.exists(pocket_path):
                    protein_path = pocket_path
                else:
                    protein_path = all_protein_path
                protein_pdbqt = protein_path[:-4] + '.pdbqt'  # in data_root
                if os.path.exists(protein_pdbqt) is False:
                    prepare_rec(protein_path, protein_pdbqt)
                ref_path = os.path.join(self.data_roots[0], ligand_list[i])
                ref_pdbqt = ref_path[:-4] + '.pdbqt'  # in data_root
                if os.path.exists(ref_pdbqt) is False:
                    prepare_lig(ref_path, ref_pdbqt)
                
                ligand_path = save_result[i][0]  # in tmp_root
                ligand_pdbqt = ligand_path[:-4] + '.pdbqt'

                prepare_lig(ligand_path, ligand_pdbqt)
                try:
                    center, box_size = get_box(protein_path)

                    ref_v = Vina(sf_name='vina', verbosity=False)
                    ref_v.set_receptor(protein_pdbqt)
                    ref_v.set_ligand_from_file(ref_pdbqt)
                    ref_v.compute_vina_maps(center=center, box_size=box_size)
                    ref_score = ref_v.score()[0]
                
                    v = Vina(sf_name='vina', verbosity=False)
                    v.set_receptor(protein_pdbqt)
                    v.set_ligand_from_file(ligand_pdbqt)
                    v.compute_vina_maps(center=center, box_size=box_size)
                    # score = v.score()[0]
                    # optimize_result = v.optimize()
                    v.dock(exhaustiveness=16, n_poses=20)
                    score = v.score()[0]
                    ref_score_list.append(ref_score)
                    score_list.append(score)
                    success_list.append(success_flag)
                except:
                    ref_score_list.append(0)
                    score_list.append(0)
                    success_list.append(False)
            else:
                ref_score_list.append(0)
                score_list.append(0)
                success_list.append(success_flag)
        
        return torch.tensor(ref_score_list), torch.tensor(score_list), torch.tensor(success_list)

    def compute_reward(self, ref_score, score, success_flag):
        
        impg_threshold = 0.05
        r_bar = [1., 0, -1., -2.]

        impg = (ref_score - score) / (abs(score) + 1.e-5)
        
        good_r = (impg > impg_threshold).float()
        mid_r = (impg.abs() <= impg_threshold).float()
        bad_r = (impg < -impg_threshold).float()
        r_success = good_r * r_bar[0] + mid_r * r_bar[1] + bad_r * r_bar[2]
        r_success = impg

        r_out = success_flag.float() * r_success + (1. - success_flag.float()) * r_bar[3]
        
        return r_out
    
    def reward(self, traj_batch, batch):
        
        os.makedirs(self.tmp_root, exist_ok=True)
        result_batch = self.translate(traj_batch[0], batch.protein_translation[:1])
        save_result = self.preprocess(result_batch)
        ref_score, score, success_flag = \
            self.vina_eval(save_result, batch['protein_entry'], batch['ligand_entry'])
        r_out = self.compute_reward(ref_score, score, success_flag)
        if os.path.exists(self.tmp_root):
            shutil.rmtree(self.tmp_root)
        return r_out    


class ConstantReward(object):
    def __init__(self, tmp_root, mode, data_paths, threshold_ratio=0.8):
        
        self.tmp_root = tmp_root
        self.mode = mode
        self.threshold_ratio = threshold_ratio
        self.pocket_path = data_paths[0]
        self.ligand_path = data_paths[1]

    def translate(self, result, translation):
        result_pos = result[0].cpu()
        result_pos += translation.cpu()
        return [result_pos] + [result[k+1] for k in range(len(result) - 1)]

    def split_batch_into_samples(self, batch):
        batch_idx = batch[-1]
        if batch_idx.numel() == 0:
            return []
        B = batch_idx.max() + 1
        batch_split = []
        for i in range(B):
            idx = (batch_idx == i)
            sample = {}
            sample['pos'] = batch[0].cpu()[idx].tolist()
            sample['type'] = batch[1].cpu()[idx].numpy()
            if len(sample['type'].shape) == 2:
                sample['type'] = sample['type'].argmax(axis=-1)
            sample['atom'] = get_atomic_number_from_index(sample['type'], self.mode)
            sample['aromatic'] = is_aromatic_from_index(sample['type'], self.mode)
            sample['file'] = f"{uuid.uuid4().hex}.sdf"
            batch_split.append(sample)
        return batch_split

    def save_mol(self, result_split):
        save_result = []
        basic_mode = True if self.mode=='basic' else False
        for i, result in enumerate(result_split):
            save_path = os.path.join(self.tmp_root, result['file'])
            try:
                try:
                    mol = reconstruct_mol(result['pos'], 
                                          result['atom'], 
                                          result['aromatic'], 
                                          basic_mode=basic_mode)
                except:
                    mol = obabel_recover_bond(result['pos'], 
                                              result['atom'])
                    
                mol, success = evaluate_validity(mol, -1, self.threshold_ratio)
                if success:
                    save_mol(mol, save_path)
                    success_flag = True
                else:
                    success_flag = False
            except:
                success_flag = False
            save_result.append((save_path, success_flag))
        return save_result
    
    def preprocess(self, traj_batch):
        batch_split = self.split_batch_into_samples(traj_batch)
        save_result = self.save_mol(batch_split)
        return save_result
    
    def vina_eval(self, save_result):
        """

        return: [(ref_score, score)]
        """

        ref_score_list = []
        score_list = []
        success_list = []

        protein_path = self.pocket_path
        protein_pdbqt = protein_path[:-4] + '.pdbqt'  # in data_root
        if os.path.exists(protein_pdbqt) is False:
            prepare_rec(protein_path, protein_pdbqt)
        ref_path = self.ligand_path
        ref_pdbqt = ref_path[:-4] + '.pdbqt'  # in data_root
        if os.path.exists(ref_pdbqt) is False:
            prepare_lig(ref_path, ref_pdbqt)
        
        center, box_size = get_box(protein_path)
        ref_v = Vina(sf_name='vina', verbosity=False)
        ref_v.set_receptor(protein_pdbqt)
        ref_v.set_ligand_from_file(ref_pdbqt)
        ref_v.compute_vina_maps(center=center, box_size=box_size)
        ref_v.dock(exhaustiveness=16, n_poses=20)
        ref_score = ref_v.score()[0]

        for i in range(len(save_result)):
            success_flag = save_result[i][1]
            if success_flag:
                
                ligand_path = save_result[i][0]  # in tmp_root
                ligand_pdbqt = ligand_path[:-4] + '.pdbqt'

                try:
                    prepare_lig(ligand_path, ligand_pdbqt)

                    v = Vina(sf_name='vina', verbosity=False)
                    v.set_receptor(protein_pdbqt)
                    v.set_ligand_from_file(ligand_pdbqt)
                    v.compute_vina_maps(center=center, box_size=box_size)
                    # score = v.score()[0]
                    v.dock(exhaustiveness=16, n_poses=20)
                    score = v.score()[0]
                    ref_score_list.append(ref_score)
                    score_list.append(score)
                    success_list.append(success_flag)
                except:
                    ref_score_list.append(0.)
                    score_list.append(0.)
                    success_list.append(False)
            else:
                ref_score_list.append(0.)
                score_list.append(0.)
                success_list.append(success_flag)
        
        return torch.tensor(ref_score_list), torch.tensor(score_list), torch.tensor(success_list)

    def compute_reward(self, ref_score, score, success_flag):
        
        impg_threshold = 0.05

        impg = (ref_score - score) / (abs(score) + 1.e-5)

        impg_thresholds = [0.05, 0.10, 0.15, 0.20]
        thresholds_num = len(impg_thresholds)
        r_success = torch.zeros_like(ref_score)
        for i in range(thresholds_num):
            r_success[impg>impg_thresholds[i]] = i+1
            r_success[impg<-impg_thresholds[i]] = -(i+1)
        
        # good_r = (impg > impg_threshold).float()
        # mid_r = (impg.abs() <= impg_threshold).float()
        # bad_r = (impg < -impg_threshold).float()
        # r_success = good_r * r_bar[0] + mid_r * r_bar[1] + bad_r * r_bar[2]
        # r_success = impg

        r_out = success_flag.float() * r_success + (1. - success_flag.float()) * (- thresholds_num - 1)
        
        return r_out, impg[success_flag].mean()
    
    def reward(self, traj_batch, batch):
        
        os.makedirs(self.tmp_root, exist_ok=True)
        result_batch = self.translate(traj_batch[0], batch.protein_translation[:1])
        save_result = self.preprocess(result_batch)
        ref_score, score, success_flag = \
            self.vina_eval(save_result)
        r_out, impg_success = self.compute_reward(ref_score, score, success_flag)
        success_rate = success_flag.sum() / success_flag.shape[0]
        if os.path.exists(self.tmp_root):
            shutil.rmtree(self.tmp_root)
        return r_out, score[success_flag.bool()].mean(), impg_success, success_rate
    
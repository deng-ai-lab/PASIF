import torch
import torch.nn.functional as F
import numpy as np

from ._base import get_index
from rdkit import Chem
from rdkit.Chem.QED import qed
from torch_geometric.data import Data
# from repo.datasets.classifier_data import ProteinLigandData
from repo.utils.protein.constants import *
from repo.utils.molecule.constants import aromatic_feat_map_idx
from repo.utils.mol_prop.sascorer import compute_sa_score
from repo.utils.mol_prop.scoring_func import obey_lipinski


class FeaturizeProteinFullAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor(atomic_numbers)  # H, C, N, O, S, Se
        self.max_num_aa = len(aa_name_number)

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data):
        data_prot = {}
        element = (data['protein_element'].view(-1, 1) == self.atomic_numbers.view(1, -1)).float()
        amino_acid = data['protein_atom_to_aa_type']
        is_backbone = data['protein_is_backbone'].view(-1, 1).long()
        x = torch.cat([element, is_backbone], dim=-1)

        data_prot['atom_feature'] = x
        data_prot['aa_type'] = amino_acid
        data_prot['pos'] = data['protein_pos']
        data_prot['element'] = data['protein_element']
        data_prot['lig_flag'] = torch.zeros_like(data['protein_element'], dtype=torch.bool)
        data_prot['atom_type'] = torch.tensor([get_index(e, h, a, 'basic') for e, h, a in zip(data['protein_element'], 
                                                                                 torch.zeros_like(data['protein_element']), 
                                                                                 torch.zeros_like(data['protein_element']))])
        data_prot['alpha_carbon_indicator'] = torch.tensor([True if name =="CA" else False for name in data['protein_molecule_name']])

        # data.protein = data_prot   
        data['protein'] = data_prot

        return data

class FeaturizeProteinResidue(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor(atomic_numbers)  # H, C, N, O, S, Se
        self.max_num_aa = 21

    @property
    def feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data):
        data_prot = {}
        element = (data['protein_element'].view(-1, 1) == self.atomic_numbers.view(1, -1)).float()
        amino_acid = F.one_hot(data['protein_atom_to_aa_type'], num_classes=self.max_num_aa)
        is_backbone = data['protein_is_backbone'].view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)

        data_prot['atom_feature'] = x
        data_prot['aa_type'] = amino_acid
        data_prot['pos'] = data['protein_pos']
        data_prot['element'] = data['protein_element']
        data_prot['lig_flag'] = torch.zeros_like(data['protein_element'], dtype=torch.bool)
        data_prot['atom_type'] = torch.tensor([get_index(e, h, a, 'basic') for e, h, a in zip(data['protein_element'], 
                                                                                 torch.zeros_like(data['protein_element']), 
                                                                                 torch.zeros_like(data['protein_element']))])
        data_prot['alpha_carbon_indicator'] = torch.tensor([True if name =="CA" else False for name in data['protein_molecule_name']])

        # data.protein = data_prot   
        data['protein'] = data_prot

        return data

class FeaturizeLigandFullAtom(object):

    def __init__(self, mode='add_aromatic'):
        super().__init__()
        self.mode = mode
        
    def __call__(self, data):
        data_lig = {}
        element_list = data['ligand_element']
        hybridization_list = data['ligand_hybridization']
        aromatic_list = [v[aromatic_feat_map_idx] for v in data['ligand_atom_feature']]
        # add_aro: class_num=14 / no_aro: class_num=10
        x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        data_lig['atom_type'] = x
        data_lig['lig_flag'] = torch.ones_like(x, dtype=torch.bool)
        data_lig['pos'] = data['ligand_pos']
        data_lig['element'] = data['ligand_element']
        
        # if hasattr(data.ligand, 'gen_flag'):
        if 'gen_flag' in data.keys():
            data_lig['gen_flag'] = data['gen_flag']
        else:
            data_lig['gen_flag'] = torch.ones_like(x, dtype=torch.bool)
        # if hasattr(data.ligand, 'ctx_flag'):
        if 'ctx_flag' in data.keys():
            data_lig['ctx_flag'] = data['ctx_flag']
        else:
            data_lig['ctx_flag'] = torch.zeros_like(x, dtype=torch.bool)

        data['ligand'] = data_lig
        return data
    
class CenterPos(object):
    def __init__(self):
        pass 

    def __call__(self, data):
        protein_pos = data['protein']['pos']
        ligand_pos = data['ligand']['pos']
        
        data_center = (ligand_pos.sum(0) + protein_pos.sum(0)) / (len(ligand_pos) + len(protein_pos))
        data_center = data_center.unsqueeze(0)
        
        data['protein']['pos'] = data['protein']['pos'] - data_center
        data['protein']['translation'] = data_center.expand(data['protein']['pos'].size(0), -1)
        
        data['ligand']['pos'] = data['ligand']['pos'] - data_center
        data['ligand']['translation'] = data_center.expand(data['ligand']['pos'].size(0), -1)
        # if hasattr(data, 'ligand') and hasattr(data.ligand, 'pos'):
        #     data['ligand'].pos = data['ligand'].pos - data_center
        #     data['ligand'].translation = data_center.expand(data['ligand'].pos.size(0), -1)

        return data

class MergeKeys(object):

    def __init__(self, keys, to_graph=True, excluded_subkeys=[]):
        super().__init__()
        self.keys = keys
        self.to_graph = to_graph
        self.excluded_keys = excluded_subkeys
        
    def __call__(self, data):
        data_merge = {}
        for key in self.keys:
            graph = data[key]
            for k, v in graph.items():
                if k not in self.excluded_keys:
                    data_merge[key + '_' + k] = v
        if self.to_graph:
            return Data(**data_merge)
        else:
            return data_merge

class NormalizeVina(object):
    def __init__(self, mode='pl'):
        super().__init__()
        self.mode = mode
        if 'pl' in mode:
            self.max_v = 0
            self.min_v = -16
        elif mode == 'pdbbind':
            self.max_v = 16
            self.min_v = 0
        else:
            raise ValueError
    
    def _trans(self, vina_score):
        if 'pl' in self.mode:
            return (self.max_v - np.clip(vina_score, self.min_v, self.max_v)) / (self.max_v - self.min_v)
        elif self.mode == 'pdbbind':
            return np.clip(vina_score, self.min_v, self.max_v) / (self.max_v - self.min_v)
        else:
            raise ValueError

    def __call__(self,  data):
        data['ligand']['affinity'] = self._trans(data['affinity'])
        return data
    
class AddMolProp(object):

    def __init__(self):
        super().__init__()
        self.max_sa = 1.0
        self.min_sa = 0.17
        self.max_qed = 0.95
        self.min_qed = 0.01

    def __call__(self, data):
        smi = data['ligand_smiles']
        mol = Chem.MolFromSmiles(smi)
        data['ligand']['qed'] = qed(mol)
        data['ligand']['qed_norm'] = (qed(mol) - self.min_qed) / (self.max_qed - self.min_qed)
        # data['ligand']['sa'] = compute_sa_score(mol)
        # data['ligand']['sa_norm'] = (compute_sa_score(mol) - self.min_sa) / (self.max_sa - self.min_sa)
        data['ligand']['lipinski'] = obey_lipinski(mol)
        data['ligand']['lipinski_norm'] = data['ligand']['lipinski'] / 5
        return data
    
class NormalizeLabel(object):
    def __init__(self, prop='caco2'):
        super().__init__()
        self.mode = prop
        if prop == 'ames':
            self.max_v = 1
            self.min_v = 0
        elif prop == 'caco2':
            self.max_v = -3
            self.min_v = -8
        elif prop == 'herg':
            self.max_v = 1
            self.min_v = 0
        elif prop == 'oatp1b3':
            self.max_v = 1
            self.min_v = 0
        elif prop == 'oatp1b1':
            self.max_v = 1
            self.min_v = 0
        elif prop == 'ppb':
            self.max_v = 100
            self.min_v = 0
        elif prop == 'cyp2c19_inh':
            self.max_v = 1
            self.min_v = 0
        elif prop == 'pgp_sub':
            self.max_v = 1
            self.min_v = 0
        else:
            raise ValueError
    
    def _trans(self, vina_score):
        return (self.max_v - np.clip(vina_score, self.min_v, self.max_v)) / (self.max_v - self.min_v)

    def __call__(self,  data):
        data['ligand']['label'] = self._trans(data['label'])
        return data
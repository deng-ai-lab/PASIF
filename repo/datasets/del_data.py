import os, sys
import numpy as np
import pickle
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
sys.path.append('/home/dataset-local/tyl/projects_dir/Molcular/Delete-main/')
from torch_geometric.data import Data


def torchify_dict(data):
    output = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            output[k] = torch.from_numpy(v)
        else:
            output[k] = v
    return output

class ProteinLigandData(Data):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_protein_ligand_dicts(protein_dict=None, ligand_dict=None, **kwargs):
        instance = ProteinLigandData(**kwargs)

        if protein_dict is not None:
            for key, item in protein_dict.items():
                instance['protein_' + key] = item

        if ligand_dict is not None:
            for key, item in ligand_dict.items():
                instance['ligand_' + key] = item

        instance['ligand_nbh_list'] = {i.item():[j.item() for k, j in enumerate(instance.ligand_bond_index[1]) 
                                                 if instance.ligand_bond_index[0, k].item() == i] 
                                                 for i in instance.ligand_bond_index[0]}
        return instance

    def __inc__(self, key, value, *args, **kwargs):
        if key == 'ligand_bond_index':
            return self['ligand_element'].size(0)
        elif key == 'ligand_context_bond_index':
            return self['ligand_context_element'].size(0)

        elif key == 'mask_ctx_edge_index_0':
            return self['ligand_masked_element'].size(0)
        elif key == 'mask_ctx_edge_index_1':
            return self['ligand_context_element'].size(0)
        elif key == 'mask_compose_edge_index_0':
            return self['ligand_masked_element'].size(0)
        elif key == 'mask_compose_edge_index_1':
            return self['compose_pos'].size(0)

        elif key == 'compose_knn_edge_index':  # edges for message passing of encoder 
            return self['compose_pos'].size(0)

        elif key == 'real_ctx_edge_index_0':
            return self['pos_real'].size(0)
        elif key == 'real_ctx_edge_index_1':
            return self['ligand_context_element'].size(0)
        elif key == 'real_compose_edge_index_0':  # edges for edge type prediction
            return self['pos_real'].size(0)
        elif key == 'real_compose_edge_index_1':
            return self['compose_pos'].size(0)

        elif key == 'real_compose_knn_edge_index_0':  # edges for message passing of  field
            return self['pos_real'].size(0)
        elif key == 'fake_compose_knn_edge_index_0':
            return self['pos_fake'].size(0)
        elif (key == 'real_compose_knn_edge_index_1') or (key == 'fake_compose_knn_edge_index_1'):
            return self['compose_pos'].size(0)

        elif (key == 'idx_protein_in_compose') or (key == 'idx_ligand_ctx_in_compose'):
            return self['compose_pos'].size(0)
            
        elif key == 'index_real_cps_edge_for_atten':
            return self['real_compose_edge_index_0'].size(0)
        elif key == 'tri_edge_index':
            return self['compose_pos'].size(0)

        elif key == 'idx_generated_in_ligand_masked':
            return self['ligand_masked_element'].size(0)
        elif key == 'idx_focal_in_compose':
            return self['compose_pos'].size(0)
        elif key == 'idx_protein_all_mask':
            return self['compose_pos'].size(0)
        else:
            return super().__inc__(key, value)
        

class SurfLigandPairDataset(Dataset):

    def __init__(self, raw_path, cfg, transform=None):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.index_path = os.path.join(self.raw_path, 'index.pkl')
        task = cfg.get('version', 'linker')
        tmp = os.path.basename(self.raw_path) + f'_mol_{task}.lmdb'
        self.processed_path = os.path.join(os.path.dirname(self.raw_path), tmp)
        tmp = os.path.basename(self.raw_path) + f'_molname2id_{task}.pt'
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path), tmp)
        self.procesed_dir = './data/pl_del/'

        if not os.path.exists(self.procesed_dir):
            os.makedirs(self.procesed_dir)
        
        # version_plus = cfg.get('version', 'linker')
        # self.raw_path_plus = cfg.raw_path.rstrip('/')
        # self.index_path_plus = os.path.join(self.raw_path_plus, 'index.pkl')
        # self.procesed_dir_plus = cfg.get('processed_dir', './data/pl_dcomp/')

        # self.processed_path_plus = os.path.join(self.procesed_dir_plus,
        #                                    os.path.basename(self.raw_path) + f'_processed_{version_plus}.lmdb')
        # self.name2id_path_plus = (os.path.join(self.procesed_dir_plus, 'crossdocked_name2id_{}.pt'.format(version_plus)) 
        #                 if 'crossdocked' in self.raw_path 
        #                 else os.path.join(self.procesed_dir, self.raw_path.split('/')[-1] + '_name2id_{}.pt'.format(version_plus)))

        self.transform = transform
        self.db = None
        self.keys = None
        if not os.path.exists(self.name2id_path):
            self._precompute_name2id()
        self.name2id = torch.load(self.name2id_path)

    def get_pickle_data(self, idx, processed_path):
        if self.db is None:
            self._connect_db(processed_path)
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        return data
    
    def get_raw(self, idx):
        return self.get_pickle_data(idx, self.processed_path)
        
    def _connect_db(self, processed_path):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _precompute_name2id(self):
        name2id = {}
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            name = (data.protein_filename, data.ligand_filename)
            name2id[name] = i
        torch.save(name2id, self.name2id_path)

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
    
    def __len__(self):
        if self.db is None:
            self._connect_db(self.processed_path)
        return len(self.keys)

    def __getitem__(self, idx):
        # data_plus = self.get_pickle_data(idx, 
        #                                  self.processed_path_plus)
        data = self.get_pickle_data(idx, 
                                    self.processed_path)
        data.id = idx
        # data.masked_idx = data_plus.ligand.gen_index
        # data.context_idx = data_plus.ligand.ctx_index
        
        assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        return data
import os, sys
sys.path.append("/home/dataset-local/tyl/projects_dir/Molcular/PASIF-release")
import argparse
import re
import shutil
import subprocess
from copy import deepcopy
from scipy import spatial
from torchvision.transforms import Compose
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from repo.models import get_model
from repo.models.diffgui.model import DiffGui
from repo.models.classifier.classifier import PropPredictor, AffinityPredictor
from repo.utils.misc import *
from repo.utils.molecule.constants import *
from repo.utils.data import recursive_to
from repo.utils.diffgui.atom_num_config import CONFIG
from repo.tools.rdkit_utils import *
from repo.datasets.diffgui_data import pdb_to_pocket, pdb_to_pocket_specify
from repo.datasets.parsers import torchify_dict
from repo.datasets.parsers.protein_parser import PDBProteinFA
from repo.datasets.parsers.molecule_parser import parse_sdf_file
from repo.datasets.transforms.merge import MergeKeys
from repo.datasets.transforms.init_lig import sample_atom_num
from repo.datasets.transforms.diffgui_transform import FeatureComplex, FeatureComplexSpecify,  make_data_placeholder
from repo.utils.diffgui.reconstruct import reconstruct_from_generated_with_edges

from tqdm import tqdm

def seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge):
    outputs_pred = outputs['pred']
    outputs_traj = outputs['traj']

    new_outputs = []
    for i_mol in range(n_graphs):
        ind_node = (batch_node == i_mol)
        ind_halfedge = (batch_halfedge == i_mol)
        assert ind_node.sum() * (ind_node.sum()-1) == ind_halfedge.sum() * 2
        new_pred_this = [outputs_pred[0][ind_node],  # node type
                         outputs_pred[1][ind_node],  # node pos
                         outputs_pred[2][ind_halfedge]]  # halfedge type
                        
        new_traj_this = [outputs_traj[0][:, ind_node],  # node type. The first dim is time
                         outputs_traj[1][:, ind_node],  # node pos
                         outputs_traj[2][:, ind_halfedge]]  # halfedge type
        
        halfedge_index_this = halfedge_index[:, ind_halfedge]
        assert ind_node.nonzero()[0].min() == halfedge_index_this.min()
        halfedge_index_this = halfedge_index_this - ind_node.nonzero()[0].min()

        new_outputs.append({
            'pred': new_pred_this,
            'traj': new_traj_this,
            'halfedge_index': halfedge_index_this,
        })
    return new_outputs

def get_space_size(pos):
        aa_dist = torch.pdist(pos)
        aa_dist = torch.sort(aa_dist, descending=True)[0]
        return torch.median(aa_dist[:10])

def get_pocket_size(pocket_pos):
    aa_dist = spatial.distance.pdist(pocket_pos, metric="euclidean")
    aa_dist_sort = np.sort(aa_dist)[::-1]
    return np.median(aa_dist_sort[:10])

def get_bin_idx(pocket_size):
    bounds = CONFIG["bounds"]
    for i in range(len(bounds)):
        if bounds[i] > pocket_size:
            return i
    return len(bounds)

def sample_atom_num(pocket_size):
    bin_idx = get_bin_idx(pocket_size)
    num_atom_list, prob_list = CONFIG["bins"][bin_idx]
    atom_num = np.random.choice(num_atom_list, p=prob_list)
    return atom_num

def translate(result, translation):
    result_pos = result[0].cpu()
    result_pos += translation.cpu()
    return [result_pos] + [result[k+1] for k in range(len(result) - 1)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--frag', type=str, default='./data/specificity/MerTK/docked/docked_9bhk.sdf')
    parser.add_argument('--target', type=str, default='./data/specificity/MerTK/docked/docked_9bhk_pocket10.pdb')
    parser.add_argument('--off_target', type=str, default='./data/specificity/MerTK/docked/docked_2buj_pocket10.pdb')
    parser.add_argument('--checkpoint', type=str, default='./logs/denovo/diffgui/pretrain/checkpoints/pretrained.pt')
    parser.add_argument('--classifier', type=str, default='./logs/affinity/diffgui/self-train/checkpoints/182000.pt')
    parser.add_argument('--model_name', type=str, default='diffgui')
    parser.add_argument('--tag', type=str, default='no_bond')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sample_num', type=int, default=5)
    parser.add_argument('--out_root', type=str, default='./results/specificity/MerTK-9bhk-2buj')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--threshold', type=int, default=-1)
    parser.add_argument('--threshold_ratio', type=int, default=0.6)
    args = parser.parse_args()

    seed_all(args.seed)

    ckpt = torch.load(args.checkpoint, map_location='cpu')
    cfg_ckpt = ckpt['config']
    cfg_ckpt.train_mode = args.tag
    cls_ckpt = torch.load(args.classifier, map_location='cpu')
    classifier = PropPredictor(cls_ckpt['config']['model'], protein_dim=28).to(args.device)
    classifier.load_state_dict(cls_ckpt['model'])

    if os.path.exists(args.out_root) is False:
        os.makedirs(args.out_root)
    
    if args.model_name[:8] == 'diffsbdd':
        mode = 'basic'
        distribution = 'zeros'
        mol_pos_dist = 'zero_mean_gaussian'
    elif args.model_name[:10] == 'targetdiff':
        mode = 'add_aromatic'
        distribution = 'uniform'
        mol_pos_dist = 'gaussian'
    elif args.model_name[:6] == 'diffbp':
        mode = 'add_aromatic'
        distribution = 'absorbing'
        mol_pos_dist = 'gaussian'
    elif args.model_name[:7] == 'diffgui':
        mode = 'diffgui'
        distribution = 'absorbing'
        mol_pos_dist = 'gaussian'
    
    data = pdb_to_pocket_specify(pocket_pdb_path=args.target, 
                                   off_target_pdb_path=args.off_target,
                                   ligand_sdf_path = 'None'
                                 )
    featurizer = FeatureComplexSpecify(cfg_ckpt.data.transform.ligand_atom_mode, 
                                       sample=True)
    transfom_func = Compose([featurizer])
    model = DiffGui(
                    config=cfg_ckpt.model,
                    protein_node_types=featurizer.protein_feat_dim,
                    ligand_node_types=featurizer.atom_feat_dim,
                    num_edge_types=featurizer.bond_feat_dim,
                ).to(args.device)
    model.load_state_dict(ckpt['model'])

    logp = torch.tensor([float(2.00)], device=args.device).unsqueeze(-1)
    tpsa = torch.tensor([float(100)], device=args.device).unsqueeze(-1)
    sa = torch.tensor([float(1.00)], device=args.device).unsqueeze(-1)
    qed = torch.tensor([float(0.80)], device=args.device).unsqueeze(-1)
    aff = torch.tensor([float(12.00)], device=args.device).unsqueeze(-1)
    batch_lab = torch.cat((logp, tpsa, sa, qed, aff), dim=1)
    batch_lab = torch.tensor([list(batch_lab[0]) for _ in range(args.batch_size)]).to(args.device)


    data_list_repeat = [transfom_func(deepcopy(data)) for _ in range(args.sample_num)]
    loader = DataLoader(PygDatasetFromList(data_list_repeat), 
                        batch_size=args.batch_size, 
                        shuffle=False,
                        follow_batch = ['protein_element', 'ligand_element', 'off_element']
                        )
    diff_T = 1000
    sample_idx = 0
    for batch in tqdm(loader, desc='specify', dynamic_ncols=True):

        try:
            batch = batch.to(args.device)
        except:
            batch = recursive_to(batch, args.device)
        
        n_graphs = batch.protein_element_batch.max() + 1
        pocket_size = get_pocket_size(batch.protein_pos.detach().cpu().numpy())
        ligand_num_atoms = [sample_atom_num(pocket_size).astype(int) for _ in range(n_graphs)]
        print(f'Sample num atoms: {ligand_num_atoms}')
        batch_holder = make_data_placeholder(n_nodes_list=ligand_num_atoms, device=args.device)
        batch_node, halfedge_index, batch_halfedge = batch_holder['batch_node'], batch_holder['halfedge_index'], batch_holder['batch_halfedge']
        
        ts = list(reversed(range(0, diff_T, 1)))
        save_params = {}
        if len(args.model_name.split('-')) > 1:
            if args.model_name.split('-')[1] == 'comp':
                outputs = model.sample(
                        n_graphs=n_graphs,
                        protein_node=batch.protein_atom_feat.float(), 
                        protein_pos=batch.protein_pos, 
                        protein_batch=batch.protein_element_batch,
                        ligand_batch=batch_node,
                        halfedge_index=halfedge_index,
                        halfedge_batch=batch_halfedge,
                        batch_lab=batch_lab,
                        gui_strength=3.0,
                    )
            else:
                outputs = model.specify_guid(
                        n_graphs=n_graphs,
                        protein_node=batch.protein_atom_feat.float(), 
                        protein_pos=batch.protein_pos, 
                        protein_batch=batch.protein_element_batch,
                        off_target_node=batch.off_atom_feat.float(), 
                        off_target_pos=batch.off_pos, 
                        off_target_batch=batch.off_element_batch,
                        ligand_batch=batch_node,
                        halfedge_index=halfedge_index,
                        halfedge_batch=batch_halfedge,
                        batch_lab=batch_lab,
                        gui_strength=3.0,
                        classifier = classifier
                    )
        else:
            outputs = model.specify_guid(
                        n_graphs=n_graphs,
                        protein_node=batch.protein_atom_feat.float(), 
                        protein_pos=batch.protein_pos, 
                        protein_batch=batch.protein_element_batch,
                        off_target_node=batch.off_atom_feat.float(), 
                        off_target_pos=batch.off_pos, 
                        off_target_batch=batch.off_element_batch,
                        ligand_batch=batch_node,
                        halfedge_index=halfedge_index,
                        halfedge_batch=batch_halfedge,
                        batch_lab=batch_lab,
                        gui_strength=3.0,
                        classifier = classifier
                    )

        outputs = {key:[v.cpu().numpy() for v in value] for key, value in outputs.items()}
        
        # decode outputs to molecules
        batch_node, halfedge_index, batch_halfedge = batch_node.cpu().numpy(), halfedge_index.cpu().numpy(), batch_halfedge.cpu().numpy()
        try:
            output_list = seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
        except Exception as e:
            print(f'Separate results error: {e}')
            continue
        gen_list = []
        add_edge = 'openbabel'
        mol_list = []
        for i_mol, output_mol in enumerate(output_list):
            mol_info = featurizer.decode_output(
                pred_node=output_mol['pred'][0],
                pred_pos=output_mol['pred'][1],
                pred_halfedge=output_mol['pred'][2],
                halfedge_index=output_mol['halfedge_index'],
            )  
            if add_edge == 'openbabel':
                del mol_info['bond_index']
                del mol_info['bond_type']
                del mol_info['bond_prob']
            try:
                rdmol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
            except:
                rdmol = obabel_recover_bond(mol_info['atom_pos'].tolist(), 
                                          mol_info['element'].tolist())
            
            rdmol, success = evaluate_validity(rdmol, threshold_ratio=0.6)
            mol_info['rdmol'] = rdmol
            smiles = Chem.MolToSmiles(rdmol)
            mol_info['smiles'] = smiles
            contain_B = re.search(r'B(?![rR]\b)', smiles)
            if '.' in smiles:
                print('Incomplete molecule: %s' % smiles)
            elif contain_B:
                print('Element Boron in molecule: %s' % smiles)
            else:   # Pass checks!
                mol_list.append(mol_info)

        sorted_mol_list = mol_list
        sdf_dir = os.path.join(args.out_root, args.model_name)
        os.makedirs(sdf_dir, exist_ok=True)
        for i, data_finished in enumerate(sorted_mol_list):
            rdmol = data_finished['rdmol']
            try:
                Chem.MolToMolFile(rdmol, os.path.join(sdf_dir, 'sample_%04d.sdf' % (sample_idx+1)))
                sample_idx += 1
            except:
                continue
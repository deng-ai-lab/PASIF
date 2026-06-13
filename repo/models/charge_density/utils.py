import ase
from rdkit import Chem
from repo.tools.rdkit_utils import *
from repo.utils.molecule.constants import *
from repo.utils.diffgui.reconstruct import reconstruct_from_generated_with_edges
from repo.models.charge_density.data import load_data, load_data_given_gridpos, load_data_given_gridpos_v2, CollateFuncAtoms, insert_grid_pos

def split_batch_into_samples(batch, mode='add_aromatic'):
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
            top_1 = sample['type'].argmax(axis=-1)
            # if mode != 'basic':
            partitioned = np.argpartition(sample['type'], -2, axis=-1)
            top_2 = partitioned.take(-2, axis=-1)
            h_mask = top_1 == 0
            top_1[h_mask] = top_2[h_mask]
            sample['type'] = top_1
        sample['atom'] = get_atomic_number_from_index(sample['type'], mode)
        sample['aromatic'] = is_aromatic_from_index(sample['type'], mode)
        batch_split.append(sample)
    return batch_split

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

def prepare_input(batch, device, params):
    mode = params['mode']
    batch_split = split_batch_into_samples(batch, mode)
    basic_mode = True if mode=='basic' else False
    density_dicts = []
    for result in batch_split:
        try:
            mol = reconstruct_mol(result['pos'], 
                                  result['atom'], 
                                  result['aromatic'], 
                                  basic_mode=basic_mode)
        except:
            mol = obabel_recover_bond(result['pos'], 
                                      result['atom'])
        ori_num = len(result['pos'])
        density_dict = load_data(mol, ori_num, cell_param=params['cell'])
        prop_pos = density_dict['grid_position'].get_all_grid()
        density_dict = insert_grid_pos(density_dict=density_dict, probe_pos=prop_pos, 
                                       atoms=density_dict['atoms'], cutoff=params['cutoff'])
        density_dicts.append(density_dict)
    collate_fn = CollateFuncAtoms(
        cutoff=params['cutoff'],
        pin_memory=device.type == "cuda",
        set_pbc_to=params['set_pbc'],
    )
    graph_dict = collate_fn(density_dicts)
    device_batch = {
        k: v.to(device=device, non_blocking=True) for k, v in graph_dict.items()
    }
    return device_batch

def prepare_input_diffgui(outputs, 
                          batch_node, halfedge_index, batch_halfedge, device, params):
    n_graphs = batch_node.max() + 1
    try:
        output_list = seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
    except Exception as e:
        print(f'Separate results error: {e}')
    add_edge = 'openbabel'
    density_dicts = []
    for i_mol, output_mol in enumerate(output_list):
        mol_info = params['featurizer'].decode_output(
            pred_node=output_mol['pred'][0],
            pred_pos=output_mol['pred'][1],
            pred_halfedge=output_mol['pred'][2],
            halfedge_index=output_mol['halfedge_index'],
        )  # note: traj is not used
        if add_edge == 'openbabel':
            del mol_info['bond_index']
            del mol_info['bond_type']
            del mol_info['bond_prob']
        try:
            mol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
        except:
            mol = obabel_recover_bond(mol_info['atom_pos'].tolist(), 
                                        mol_info['element'].tolist())
        ori_num = len(output_mol['pred'][1])
        density_dict = load_data(mol, ori_num, cell_param=params['cell'])
        prop_pos = density_dict['grid_position'].get_all_grid()
        density_dict = insert_grid_pos(density_dict=density_dict, probe_pos=prop_pos, 
                                       atoms=density_dict['atoms'], cutoff=params['cutoff'])
        density_dicts.append(density_dict)
    collate_fn = CollateFuncAtoms(
        cutoff=params['cutoff'],
        pin_memory=device.type == "cuda",
        set_pbc_to=params['set_pbc'],
    )
    graph_dict = collate_fn(density_dicts)
    device_batch = {
        k: v.to(device=device, non_blocking=True) for k, v in graph_dict.items()
    }
    return device_batch

def prepare_input_given_grid_diffgui(outputs, batch_node, halfedge_index, 
                                     batch_halfedge, device, params, label, offset, local=False):
    n_graphs = batch_node.max() + 1
    grid_pos = params['grid_pos']
    try:
        output_list = seperate_outputs(outputs, n_graphs, batch_node, halfedge_index, batch_halfedge)
    except Exception as e:
        print(f'Separate results error: {e}')
    add_edge = 'openbabel'
    density_dicts = []
    for i_mol, output_mol in enumerate(output_list):
        mol_info = params['featurizer'].decode_output(
            pred_node=output_mol['pred'][0],
            pred_pos=output_mol['pred'][1],
            pred_halfedge=output_mol['pred'][2],
            halfedge_index=output_mol['halfedge_index'],
            ignore_H=True
        )  # note: traj is not used
        if add_edge == 'openbabel':
            del mol_info['bond_index']
            del mol_info['bond_type']
            del mol_info['bond_prob']
        try:
            mol = reconstruct_from_generated_with_edges(mol_info, add_edge=add_edge)
        except:
            mol = obabel_recover_bond(mol_info['atom_pos'].tolist(), 
                                        mol_info['element'].tolist())
        ori_num = len(output_mol['pred'][1])
        density_dict = load_data_given_gridpos(mol, ori_num, grid_pos=grid_pos, addH=local)
        # density_dict = load_data_given_gridpos_v2(mol_info['atom_pos'], mol_info['element'], 
        #                                           ori_num, grid_pos=grid_pos-offset[i_mol])
        if local:
            prop_pos = density_dict['grid_position']
            mask = np.ones_like(prop_pos[..., -1], dtype=bool)
        else:
            prop_pos, mask = filter_grid_pos(density_dict['grid_position'], density_dict['atoms'], label[i_mol])
        spans = prop_pos.max(0) - prop_pos.min(0)
        density_dict['origin'] = prop_pos.min(0)
        density_dict['atoms'].set_cell([spans[0].item(), spans[1].item(), spans[2].item(), 
                                        90, 90, 90])
        density_dict = insert_grid_pos(density_dict=density_dict, probe_pos=prop_pos, 
                                       atoms=density_dict['atoms'], cutoff=params['cutoff'])
        density_dict['filter_mask'] = mask
        density_dict['label'] = label[i_mol][mask]
        density_dicts.append(density_dict)
    collate_fn = CollateFuncAtoms(
        cutoff=params['cutoff'],
        pin_memory=device.type == "cuda",
        set_pbc_to=params['set_pbc'],
    )
    graph_dict = collate_fn(density_dicts)
    device_batch = {
        k: v.to(device=device, non_blocking=True) for k, v in graph_dict.items()
    }
    return device_batch


def prepare_input_given_grid(batch, device, params, label, local=False):
    mode = params['mode']
    grid_pos = params['grid_pos']
    batch_split = split_batch_into_samples(batch, mode)
    basic_mode = True if mode=='basic' else False
    density_dicts = []
    i = 0
    for result in batch_split:
        try:
            # mol = obabel_recover_bond(result['pos'], 
            #                           result['atom'])
            mol = reconstruct_mol(result['pos'], 
                                  result['atom'], 
                                  result['aromatic'], 
                                  basic_mode=basic_mode)
        except:
            mol = obabel_recover_bond(result['pos'], 
                                      result['atom'])
        ori_num = len(result['pos'])
        # if mode == 'basic':
        #     density_dict = load_data_given_gridpos(mol, ori_num, grid_pos=params['grid_pos'], addH=False)
        # else:
        density_dict = load_data_given_gridpos(mol, ori_num, grid_pos=params['grid_pos'], addH=local)
        if local:
            prop_pos = density_dict['grid_position']
            mask = np.ones_like(prop_pos[..., -1], dtype=bool)
        else:
            prop_pos, mask = filter_grid_pos(density_dict['grid_position'], density_dict['atoms'], label[i])
        spans = prop_pos.max(0) - prop_pos.min(0)
        density_dict['atoms'].set_cell([spans[0].item(), spans[1].item(), spans[2].item(), 
                                        90, 90, 90])
        density_dict = insert_grid_pos(density_dict=density_dict, probe_pos=prop_pos, 
                                       atoms=density_dict['atoms'], cutoff=params['cutoff'])
        density_dict['filter_mask'] = mask
        density_dict['label'] = label[i][mask]
        density_dicts.append(density_dict)
        i += 1
    collate_fn = CollateFuncAtoms(
        cutoff=params['cutoff'],
        pin_memory=device.type == "cuda",
        set_pbc_to=params['set_pbc'],
    )
    graph_dict = collate_fn(density_dicts)
    device_batch = {
        k: v.to(device=device, non_blocking=True) for k, v in graph_dict.items()
    }
    return device_batch

def filter_grid_pos(grid_pos, atoms, label):

    atom_pos = atoms.get_positions()
    dist_mat = np.linalg.norm((atom_pos[None, :, :] - grid_pos[:, None, :]), axis=-1)
    min_dist = dist_mat.min(-1)
    mask_dist = (min_dist <3.)   # 3
    mask_label = label > 0.1
    mask = np.logical_or(mask_dist, mask_label)

    return grid_pos[mask], mask
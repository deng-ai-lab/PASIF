import ase
import math
import asap3
import torch
import numpy as np
from rdkit import Chem
from pyscf import lib

from repo.models.charge_density.layer import pad_and_stack

from typing import List

def _cell_heights(cell_object):
    volume = cell_object.volume
    crossproducts = np.cross(cell_object[[1, 2, 0]], cell_object[[2, 0, 1]])
    crosslengths = np.sqrt(np.sum(np.square(crossproducts), axis=1))
    heights = volume / crosslengths
    return heights

def ceil_float(x, step_size):
    # Round up to nearest step_size and subtract a small epsilon
    x = math.ceil(x/step_size) * step_size
    eps = 2*np.finfo(float).eps * x
    return x - eps

class LazyMeshGrid():
    def __init__(self, cell, grid_step, origin=None, adjust_grid_step=False):
        self.cell = cell
        if adjust_grid_step:
            n_steps = np.round(self.cell.lengths()/grid_step)
            self.scaled_grid_vectors = [np.arange(n)/n for n in n_steps]
            self.adjusted_grid_step = self.cell.lengths()/n_steps
        else:
            self.scaled_grid_vectors = [np.arange(0, l, grid_step)/l for l in self.cell.lengths()]
        self.shape = np.array([len(g) for g in self.scaled_grid_vectors] + [3])
        if origin is None:
            self.origin = np.zeros(3)
        else:
            self.origin = origin

        self.origin = np.expand_dims(self.origin, 0)

    def __getitem__(self, indices):
        indices = np.array(indices)
        indices_shape = indices.shape
        if not (len(indices_shape) == 2 and indices_shape[0] == 3):
            raise NotImplementedError("Indexing must be a 3xN array-like object")
        gridA = self.scaled_grid_vectors[0][indices[0]]
        gridB = self.scaled_grid_vectors[1][indices[1]]
        gridC = self.scaled_grid_vectors[2][indices[2]]

        grid_pos = np.stack([gridA, gridB, gridC], 1)
        grid_pos = np.dot(grid_pos, self.cell)
        grid_pos += self.origin

        return grid_pos
    
    def get_all_grid(self):
        frac_coords = lib.cartesian_prod([self.scaled_grid_vectors[0], 
                                          self.scaled_grid_vectors[1], 
                                          self.scaled_grid_vectors[2]])
        all_grid_pos = np.dot(frac_coords, self.cell)
        all_grid_pos += self.origin
        return all_grid_pos


class AseNeigborListWrapper:
    """
    Wrapper around ASE neighborlist to have the same interface as asap3 neighborlist

    """

    def __init__(self, cutoff, atoms):
        self.neighborlist = ase.neighborlist.NewPrimitiveNeighborList(
            cutoff, skin=0.0, self_interaction=False, bothways=True
        )
        self.neighborlist.build(
            atoms.get_pbc(), atoms.get_cell(), atoms.get_positions()
        )
        self.cutoff = cutoff
        self.atoms_positions = atoms.get_positions()
        self.atoms_cell = atoms.get_cell()

    def get_neighbors(self, i, cutoff):
        assert (
            cutoff == self.cutoff
        ), "Cutoff must be the same as used to initialise the neighborlist"

        indices, offsets = self.neighborlist.get_neighbors(i)

        rel_positions = (
            self.atoms_positions[indices]
            + offsets @ self.atoms_cell
            - self.atoms_positions[i][None]
        )

        dist2 = np.sum(np.square(rel_positions), axis=1)

        return indices, rel_positions, dist2

class CollateFuncAtoms:
    def __init__(self, cutoff, pin_memory=True, set_pbc_to=None):
        self.cutoff = cutoff
        self.pin_memory = pin_memory
        self.set_pbc = set_pbc_to

    def __call__(self, input_dicts: List):
        graphs = []
        for i in input_dicts:
            if self.set_pbc is not None:
                atoms = i["atoms"].copy()
                atoms.set_pbc(self.set_pbc)
            else:
                atoms = i["atoms"]

            graph_dict = atoms_to_graph_dict(
                atoms,
                self.cutoff,
            )

            graph_dict["translate"] = torch.tensor(i["translate"], dtype=torch.float32)
            graph_dict["heavy_mask"] = torch.tensor(i["heavy_mask"], dtype=torch.bool)
            graph_dict["probe_edges"] = torch.tensor(i["probe_edges"], dtype=torch.long)
            graph_dict["probe_edges_displacement"] = torch.tensor(i["probe_edges_displacement"])
            graph_dict["probe_xyz"] = torch.tensor(i["probe_xyz"], dtype=torch.float32)
            graph_dict["num_probe_edges"] = torch.tensor(i["num_probe_edges"], dtype=torch.long)
            graph_dict["num_probes"] = torch.tensor(i["num_probes"], dtype=torch.long)
            
            if 'label' in i.keys():
                graph_dict["label"] = torch.tensor(i["label"], dtype=torch.float32)
            graphs.append(graph_dict)

        return collate_list_of_dicts(graphs, pin_memory=self.pin_memory)


def load_molecule(atoms, grid_step, vacuum, cell_param=None):
    coords_old = atoms.get_positions()
    atoms = atoms.copy()
    atoms.center(vacuum=vacuum) # This will create a cell around the atoms
    coords_new = atoms.get_positions()

    # Readjust cell lengths to be a multiple of grid_step
    if cell_param is None:
        a, b, c, ang_bc, ang_ac, ang_ab = atoms.get_cell_lengths_and_angles()
        a, b, c = ceil_float(a, grid_step), ceil_float(b, grid_step), ceil_float(c, grid_step)
        atoms.set_cell([a, b, c, ang_bc, ang_ac, ang_ab])
    else:
        atoms.set_cell(cell_param)

    origin = np.zeros(3)
    translate = (coords_old.mean(0) - coords_new.mean(0))

    grid_pos = LazyMeshGrid(atoms.get_cell(), grid_step)

    return atoms, grid_pos, origin, translate

def load_data(mol, ori_num, cell_param=None):
    mol_addH = Chem.AddHs(mol)
    symbols = [atom.GetSymbol() for atom in mol_addH.GetAtoms()]
    conf = mol_addH.GetConformer()
    coords = conf.GetPositions()

    atoms = ase.Atoms(symbols, positions=[(coords[i][0], coords[i][1], coords[i][2]) 
                                        for i in range(coords.shape[0])])
    atoms, grid_pos, origin, translate = load_molecule(atoms, grid_step=0.5, vacuum=1.0, cell_param=cell_param)

    # ori_num = mol.GetNumAtoms()
    addH_num = mol_addH.GetNumAtoms()
    heavy_mask = np.zeros((addH_num), dtype=np.bool_)
    heavy_mask[:ori_num] = True

    metadata = {"filename": ""}
    res = {
        "atoms": atoms,
        "origin": origin,
        "grid_position": grid_pos,
        "metadat": metadata,
        "translate": translate,
        "heavy_mask": heavy_mask
    }

    return res

def load_data_given_gridpos(mol, ori_num, grid_pos, addH=False):
    if addH:
        mol_addH = Chem.AddHs(mol, addCoords=True)
    else:
        mol_addH = mol
    symbols = [atom.GetSymbol() for atom in mol_addH.GetAtoms()]
    conf = mol_addH.GetConformer()
    coords = conf.GetPositions()

    offset = coords.mean(0)
    coords = coords - offset
    grid_pos = grid_pos - offset
    origin = grid_pos.min(0)
    atoms = ase.Atoms(symbols, positions=[(coords[i][0], coords[i][1], coords[i][2]) 
                                        for i in range(coords.shape[0])])
    addH_num = mol_addH.GetNumAtoms()
    heavy_mask = np.zeros((addH_num), dtype=np.bool_)
    heavy_mask[:ori_num] = True

    metadata = {"filename": ""}
    res = {
        "atoms": atoms,
        "origin": origin,
        "grid_position": grid_pos,
        "metadat": metadata,
        "translate": offset,
        "heavy_mask": heavy_mask
    }

    return res

def load_data_given_gridpos_v2(coords, atom_numbers, ori_num, grid_pos):
    # mol_addH = Chem.AddHs(mol, addCoords=True)
    # mol_addH = mol
    pt = Chem.GetPeriodicTable()
    symbols = [pt.GetElementSymbol(int(n)) for n in atom_numbers]
    # conf = mol_addH.GetConformer()
    # coords = conf.GetPositions()

    offset = coords.mean(0)
    coords = coords - offset
    grid_pos = grid_pos - offset
    origin = grid_pos.min(0)
    atoms = ase.Atoms(symbols, positions=[(coords[i][0], coords[i][1], coords[i][2]) 
                                        for i in range(coords.shape[0])])
    # addH_num = mol_addH.GetNumAtoms()
    addH_num = coords.shape[0]
    heavy_mask = np.zeros((addH_num), dtype=np.bool_)
    heavy_mask[:ori_num] = True

    metadata = {"filename": ""}
    res = {
        "atoms": atoms,
        "origin": origin,
        "grid_position": grid_pos,
        "metadat": metadata,
        "translate": offset,
        "heavy_mask": heavy_mask
    }

    return res

def atoms_to_graph(atoms, cutoff):
    atom_edges = []
    atom_edges_displacement = []

    inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    # Compute neighborlist
    if (
        np.any(atoms.get_cell().lengths() <= 0.0001)
        or (
            np.any(atoms.get_pbc())
            and np.any(_cell_heights(atoms.get_cell()) < cutoff)
        )
    ):
        neighborlist = AseNeigborListWrapper(cutoff, atoms)
    else:
        neighborlist = asap3.FullNeighborList(cutoff, atoms)

    atom_positions = atoms.get_positions()

    for i in range(len(atoms)):
        neigh_idx, neigh_vec, _ = neighborlist.get_neighbors(i, cutoff)

        self_index = np.ones_like(neigh_idx) * i
        edges = np.stack((neigh_idx, self_index), axis=1)

        neigh_pos = atom_positions[neigh_idx]
        this_pos = atom_positions[i]
        neigh_origin = neigh_vec + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        atom_edges.append(edges)
        atom_edges_displacement.append(neigh_origin_scaled)

    return atom_edges, atom_edges_displacement, neighborlist, inv_cell_T

def atoms_to_graph_dict(atoms, cutoff):
    atom_edges, atom_edges_displacement, _, _ = atoms_to_graph(atoms, cutoff)

    default_type = torch.get_default_dtype()

    # pylint: disable=E1102
    res = {
        "nodes": torch.tensor(atoms.get_atomic_numbers()),
        "atom_edges": torch.tensor(np.concatenate(atom_edges, axis=0)),
        "atom_edges_displacement": torch.tensor(
            np.concatenate(atom_edges_displacement, axis=0), dtype=default_type
        ),
    }
    res["num_nodes"] = torch.tensor(res["nodes"].shape[0])
    res["num_atom_edges"] = torch.tensor(res["atom_edges"].shape[0])
    res["atom_xyz"] = torch.tensor(atoms.get_positions(), dtype=default_type)
    res["cell"] = torch.tensor(np.array(atoms.get_cell()), dtype=default_type)
    return res

def probes_to_graph(atoms, probe_pos, cutoff, neighborlist=None, inv_cell_T=None):
    probe_edges = []
    probe_edges_displacement = []
    if inv_cell_T is None:
        inv_cell_T = np.linalg.inv(atoms.get_cell().complete().T)

    if hasattr(neighborlist, "get_neighbors_querypoint"):
        results = neighborlist.get_neighbors_querypoint(probe_pos, cutoff)
        atomic_numbers = atoms.get_atomic_numbers()
    else:
        # Insert probe atoms
        num_probes = probe_pos.shape[0]
        probe_atoms = ase.Atoms(numbers=[0] * num_probes, positions=probe_pos)
        atoms_with_probes = atoms.copy()
        atoms_with_probes.extend(probe_atoms)
        atomic_numbers = atoms_with_probes.get_atomic_numbers()

        if (
            np.any(atoms.get_cell().lengths() <= 0.0001)
            or (
                np.any(atoms.get_pbc())
                and np.any(_cell_heights(atoms.get_cell()) < cutoff)
            )
        ):
            neighborlist = AseNeigborListWrapper(cutoff, atoms_with_probes)
        else:
            neighborlist = asap3.FullNeighborList(cutoff, atoms_with_probes)

        results = [neighborlist.get_neighbors(i+len(atoms), cutoff) for i in range(num_probes)]

    atom_positions = atoms.get_positions()
    for i, (neigh_idx, neigh_vec, _) in enumerate(results):
        neigh_atomic_species = atomic_numbers[neigh_idx]

        neigh_is_atom = neigh_atomic_species != 0
        neigh_atoms = neigh_idx[neigh_is_atom]
        self_index = np.ones_like(neigh_atoms) * i
        edges = np.stack((neigh_atoms, self_index), axis=1)

        neigh_pos = atom_positions[neigh_atoms]
        this_pos = probe_pos[i]
        neigh_origin = neigh_vec[neigh_is_atom] + this_pos - neigh_pos
        neigh_origin_scaled = np.round(inv_cell_T.dot(neigh_origin.T).T)

        probe_edges.append(edges)
        probe_edges_displacement.append(neigh_origin_scaled)

    return probe_edges, probe_edges_displacement

def collate_list_of_dicts(list_of_dicts, pin_memory=False):
    # Convert from "list of dicts" to "dict of lists"
    dict_of_lists = {k: [dic[k] for dic in list_of_dicts] for k in list_of_dicts[0]}

    # Convert each list of tensors to single tensor with pad and stack
    if pin_memory:
        pin = lambda x: x.pin_memory()
    else:
        pin = lambda x: x

    collated = {k: pin(pad_and_stack(dict_of_lists[k])) for k in dict_of_lists}
    return collated

def insert_grid_pos(density_dict, probe_pos, atoms, cutoff):
    probe_edges, probe_edges_displacement = probes_to_graph(atoms, probe_pos, cutoff)
    if not probe_edges:
        probe_edges = [np.zeros((0,2), dtype=np.int)]
        probe_edges_displacement = [np.zeros((0,3), dtype=np.float32)]
    probe_edges = np.concatenate(probe_edges, axis=0)
    probe_edges_displacement = np.concatenate(probe_edges_displacement, axis=0).astype(np.float32)
    num_probe_edges = probe_edges.shape[0]
    num_probes = probe_pos.shape[0]
    probe_xyz = probe_pos.astype(np.float32)

    density_dict["probe_edges"] = probe_edges
    density_dict["probe_edges_displacement"] = probe_edges_displacement
    density_dict["probe_xyz"] = probe_xyz
    density_dict["num_probe_edges"] = num_probe_edges
    density_dict["num_probes"] = num_probes

    return density_dict







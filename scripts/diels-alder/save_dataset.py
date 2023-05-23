from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import pandas as pd
from rxnmapper import RXNMapper
import h5py as h5

from reaction_path_sampler.src.utils import read_trajectory_file


atomic_number_dict = {
    'H':  1,
    'O':  8,
    'N':  7,
    'Si': 14,
    'P': 15,
    'B': 5,
    'Se': 34,
    'C':  6,
    'S':  16,
    'Cl': 17,
    'Br': 35,
    'F':  9,
    'I':  53
}

class Geometry(object):
    def __init__(self, atomic_nums, cartesians) -> None:
        self.atomic_nums = atomic_nums
        self.cartesians = cartesians

def xyz_string_to_geometry(xyz_string: str) -> Geometry:
    lines = xyz_string.split('\n')
    n_atoms = float(lines[0].strip())
    atomic_nums, cartesians = [], []
    for line in lines[2:int(2+n_atoms)]:
        symbol, x, y, z = line.split()
        atomic_nums.append(atomic_number_dict[symbol])
        cartesians.append([float(x), float(y), float(z)])
    return Geometry(np.array(atomic_nums), np.array(cartesians))


if __name__ == "__main__":
    reaction_dataset_path = "./data/DA_test_no_solvent.txt"
    
    path = "./scratch/DA_test_no_solvent/"
    name = "DA_test_no_solvent"
    dataset_name = "diels_alder_reaxys_rps_dataset"


    if reaction_dataset_path.split('.')[-1] == 'csv':
        reaction_smiles_list = pd.read_csv(reaction_dataset_path)['reaction_smiles'].values
    elif reaction_dataset_path.split('.')[-1] == 'txt':
        with open(reaction_dataset_path, 'r') as f:
            reaction_smiles_list = [line.replace('\n', '') for line in f.readlines()]

    idx_list = []
    for root, dirs, files in os.walk(path):
        if len(root.split('/')) > 3 and root.split('/')[-2] == name:
            if os.path.exists(os.path.join(root, 'barrier.txt')):
                idx_list.append(int(root.split('/')[-1]))
    idx_list = sorted(idx_list)
    successfull_reaction_smiles = [reaction_smiles_list[i] for i in idx_list]

    # atom-mapped reaction smiles
    atom_mapped_reaction_smiles = []
    rxn_mapper = RXNMapper()
    for smiles in successfull_reaction_smiles:
        atom_mapped_reaction_smiles.append(rxn_mapper.get_attention_guided_atom_maps([smiles])[0]['mapped_rxn'])
    
    # activation energies & geometries
    reactant_geometries, ts_geometries, product_geometries = [], [], []
    activation_energies = []
    for i in idx_list:
        results_dir = os.path.join(path, f'{i}')
        reaction, _ = read_trajectory_file(os.path.join(results_dir, 'reaction.xyz'))
        reactant_geometries.append(xyz_string_to_geometry(reaction[0]))
        ts_geometries.append(xyz_string_to_geometry(reaction[1]))
        product_geometries.append(xyz_string_to_geometry(reaction[2]))

        with open(os.path.join(results_dir, 'barrier.txt'), 'r') as f:
            barrier = float(f.readlines()[0])
            activation_energies.append(barrier)

    # add padding to all geometries
    max_n_atoms = max([geom.atomic_nums.shape[0] for geom in reactant_geometries])
    print('max n atoms: ', max_n_atoms)
    for geom in reactant_geometries:
        n_pad = max_n_atoms - geom.atomic_nums.shape[0]
        geom.atomic_nums = np.concatenate([geom.atomic_nums, np.zeros(n_pad)], axis=0)
        geom.cartesians = np.concatenate([geom.cartesians, np.zeros((n_pad, 3))], axis=0)
    for geom in ts_geometries:
        n_pad = max_n_atoms - geom.atomic_nums.shape[0]
        geom.atomic_nums = np.concatenate([geom.atomic_nums, np.zeros(n_pad)], axis=0)
        geom.cartesians = np.concatenate([geom.cartesians, np.zeros((n_pad, 3))], axis=0)
    for geom in product_geometries:
        n_pad = max_n_atoms - geom.atomic_nums.shape[0]
        geom.atomic_nums = np.concatenate([geom.atomic_nums, np.zeros(n_pad)], axis=0)
        geom.cartesians = np.concatenate([geom.cartesians, np.zeros((n_pad, 3))], axis=0)

    # save all to hdf5 file
    with open(f'./data/{dataset_name}.txt', 'w') as f:
        f.writelines("\n".join(atom_mapped_reaction_smiles))

    file = h5.File(f'./data/{dataset_name}.h5', 'w')
    file.create_dataset('activation_energies', data=np.array(activation_energies))
    r_group = file.create_group('reactants')
    r_group.create_dataset('atomic_nums', data=np.stack([geom.atomic_nums for geom in reactant_geometries]))
    r_group.create_dataset('cartesians', data=np.stack([geom.cartesians for geom in reactant_geometries]))
    ts_group = file.create_group('ts')
    ts_group.create_dataset('atomic_nums', data=np.stack([geom.atomic_nums for geom in ts_geometries]))
    ts_group.create_dataset('cartesians', data=np.stack([geom.cartesians for geom in ts_geometries]))
    p_group = file.create_group('products')
    p_group.create_dataset('atomic_nums', data=np.stack([geom.atomic_nums for geom in product_geometries]))
    p_group.create_dataset('cartesians', data=np.stack([geom.cartesians for geom in product_geometries]))

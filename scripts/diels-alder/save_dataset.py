from concurrent.futures import ProcessPoolExecutor
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rxnmapper import RXNMapper
import h5py as h5

from src.interfaces.XTB import xtb_driver
from src.utils import read_trajectory_file


atomic_number_dict = {
    'H':  1,
    'O':  8,
    'N':  7,
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

def compute_activation_energy(args) -> float:
    r_geometry, ts_geometry, solvent = args
    r_energy = xtb_driver(
        xyz_string=r_geometry,
        charge=0,
        mult=1,
        job="sp",
        method="2",
        solvent=solvent,
        n_cores=4
    )
    ts_energy = xtb_driver(
        xyz_string=ts_geometry,
        charge=0,
        mult=1,
        job="sp",
        method="2",
        solvent=solvent,
        n_cores=4
    )
    return ts_energy - r_energy

if __name__ == "__main__":
    # reaction_cores_path = "./data/diels_alder_reaction_cores.txt"
    # path = "./scratch/diels_alder_reaction_cores/"
    # name = "diels_alder_reaction_cores"
    # dataset_name = "diels_alder_core_dataset"
    # n_processes = 25
    reaction_cores_path = "./data/test_da_reactions_reaction_core.csv"
    path = "./scratch/da_reaction_cores_test/"
    name = "da_reaction_cores_test"
    dataset_name = "diels_alder_test_dataset"
    n_processes = 25
    
    reaction_smiles_list = pd.read_csv(reaction_cores_path)['reaction_smiles'].values

    # with open(reaction_cores_path, 'r') as f:
    #     reaction_smiles_list = [line.replace('\n', '') for line in f.readlines()]

    idx_list = []
    for root, dirs, files in os.walk(path):
        if len(root.split('/')) > 3 and root.split('/')[-2] == name:
            if os.path.exists(os.path.join(root, 'reaction.xyz')):
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
    ea_computation_arguments = []
    for i in idx_list:
        results_dir = os.path.join(path, f'{i}')
        reaction, _ = read_trajectory_file(os.path.join(results_dir, 'reaction.xyz'))
        reactant_geometries.append(xyz_string_to_geometry(reaction[0]))
        ts_geometries.append(xyz_string_to_geometry(reaction[1]))
        product_geometries.append(xyz_string_to_geometry(reaction[2]))
        ea_computation_arguments.append((reaction[0], reaction[1], 'Methanol'))

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        activation_energies = list(tqdm(executor.map(compute_activation_energy, ea_computation_arguments), total=len(ea_computation_arguments), desc="Computing activation energies"))
    activation_energies = np.array(activation_energies)

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

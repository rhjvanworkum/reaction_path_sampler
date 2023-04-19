"""
Each reaction path needs a reactant + product complex
"""


from concurrent.futures import ProcessPoolExecutor
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
import os
from concurrent.futures import ProcessPoolExecutor
import time
from tqdm import tqdm

from autode.geom import calc_heavy_atom_rmsd, get_rot_mat_kabsch
from autode.conformers.conformer import Conformer


from src.reactive_complex_sampler import ReactiveComplexSampler
from src.utils import autode_conf_to_xyz_string, read_trajectory_file, remove_whitespaces_from_xyz_strings, xyz_string_to_autode_atoms


def generate_reactant_product_complexes(
    smiles_strings: List[str],
    solvent: str,
    settings: Any,
    save_path: str
) -> List[Conformer]:
    rps = ReactiveComplexSampler(
        smiles_strings=smiles_strings,
        solvent=solvent,
        settings=settings
    )

    complex = rps._get_ade_complex()

    if os.path.exists(save_path):
        conformers, _ = read_trajectory_file(save_path)
        conformer_list = [Conformer(
            atoms=xyz_string_to_autode_atoms(structure), 
            charge=complex.charge, 
            mult=complex.mult
        ) for structure in conformers]

    else:
        t = time.time()
        complexes = rps._sample_initial_complexes()
        print(f'time to do autode sampling: {time.time() - t}')

        # for idx, complex in enumerate(complexes):
        #     with open(f'complex_{idx}.xyz', 'w') as f:
        #         f.writelines(autode_conf_to_xyz_string(complex))
        # return None

        conformer_list = []
        conformer_xyz_list = []
        for complex in complexes:
            conformers = rps.sample_reaction_complexes(complex=complex)
            for conformer in conformers:
                conformer_xyz_list.append(conformer)
                conformer_list.append(Conformer(
                    atoms=xyz_string_to_autode_atoms(conformer), 
                    charge=complex.charge, 
                    mult=complex.mult
                ))

        with open(save_path, 'w') as f:
            f.writelines(remove_whitespaces_from_xyz_strings(conformer_xyz_list))

    return complex, conformer_list


def compute_optimal_rmsd(args):
    idxs, rc_coordinates, pc_coordinates = args
    new_rc_coordinates = compute_optimal_coordinates(
        rc_coordinates,
        pc_coordinates
    )
    return idxs, np.sqrt(np.mean((pc_coordinates - new_rc_coordinates)**2))

def compute_optimal_rmsd_bb(args):
    idxs, rc_coordinates, pc_coordinates, bonds = args
    new_rc_coordinates = compute_optimal_coordinates(
        rc_coordinates,
        pc_coordinates
    )
    # rmsd = np.sqrt(np.mean((pc_coordinates - new_rc_coordinates)**2))
    bb = sum([
        np.sqrt(np.mean((new_rc_coordinates[bond[0]] - new_rc_coordinates[bond[1]])**2)) for bond in bonds
    ]) 
    bb += sum([
        np.sqrt(np.mean((pc_coordinates[bond[0]] - pc_coordinates[bond[1]])**2)) for bond in bonds
    ])

    score = bb
    return idxs, score

def select_promising_reactant_product_pairs(
    rc_conformers: List[Conformer],
    pc_conformers: List[Conformer],
    bonds: List[Tuple[int, int]],
    settings: Any
) -> List[Tuple[int]]:
    """
    This function aims to find the set of most promising reactant + product complex
    pairs to find a reaction path between.

    The current formulation is based on a mixture between RMSD between the complexes &
    the distances between the atoms which bonds change during the reaction
    """
    k = settings['max_n_aligned_screening_pairs']
    n_reactant_product_pairs = settings['max_n_reactant_product_pairs']

    # TODO: parallelize this?
    indices = []
    scores = []
    t = time.time()
    for i in range(len(rc_conformers)):
        for j in range(len(pc_conformers)):
            indices.append((i, j))
            scores.append(
                sum([
                    np.sqrt(np.mean((rc_conformers[bond[0]].coordinates - rc_conformers[bond[1]].coordinates)**2)) for bond in bonds
                ]) + sum([
                    np.sqrt(np.mean((pc_conformers[bond[0]].coordinates - pc_conformers[bond[1]].coordinates)**2)) for bond in bonds
                ])
            )
    print(f'RMSD complexes compute time {time.time() - t}')
    indices = np.array(indices)
    scores = np.array(scores)    

    if len(scores) == 1:
        print(f'Only 1 reactant & product complex was found')
        opt_idxs = [indices[0]]
    elif len(scores) <= n_reactant_product_pairs:
        n_reactant_product_pairs = len(scores) - 1
        print(f'reduced amount of reactant & product pairs to {n_reactant_product_pairs}')
        opt_idxs = indices[np.argpartition(scores, n_reactant_product_pairs)[:n_reactant_product_pairs]]
    else:
        opt_idxs = indices[np.argpartition(scores, n_reactant_product_pairs)[:n_reactant_product_pairs]]
        # perform an extra screening round of top k
        # pre_opt_pairs = min(k, len(scores) - 1)
        # pre_opt_idxs = indices[np.argpartition(scores, pre_opt_pairs)[:pre_opt_pairs]]
        # args = [
        #     (idxs, rc_conformers[idxs[0]].coordinates, pc_conformers[idxs[1]].coordinates, bonds) for idxs in pre_opt_idxs
        # ]
        # with ProcessPoolExecutor(max_workers=int(settings['n_processes'] * settings['xtb_n_cores'])) as executor:
        #     results = list(tqdm(executor.map(compute_optimal_rmsd_bb, args), total=len(args), desc="Computing optimal complex RMSD scores"))
        # idxs = np.array([result[0] for result in results])
        # scores = np.array([result[1] for result in results])
        # opt_idxs = idxs[np.argpartition(scores, n_reactant_product_pairs)[:n_reactant_product_pairs]]

    return opt_idxs


    # # compute RMSDS for each pair
    # indices = []
    # rmsds = []
    # t = time.time()
    # for i in range(len(rc_conformers)):
    #     for j in range(len(pc_conformers)):
    #         indices.append((i, j))
    #         rmsds.append(np.sqrt(np.mean((pc_conformers[j].coordinates - rc_conformers[i].coordinates)**2)))
    # print(f'RMSD complexes compute time {time.time() - t}')
    # indices = np.array(indices)
    # rmsds = np.array(rmsds)


    # # select pairs
    # if len(rmsds) == 1:
    #     print(f'Only 1 reactant & product complex was found')
    #     opt_idxs = [indices[0]]
    # elif len(rmsds) <= n_reactant_product_pairs:
    #     n_reactant_product_pairs = len(rmsds) - 1
    #     print(f'reduced amount of reactant & product pairs to {n_reactant_product_pairs}')
    #     opt_idxs = indices[np.argpartition(rmsds, n_reactant_product_pairs)[:n_reactant_product_pairs]]
    # else:
    #     # perform an extra screening round of top k
    #     pre_opt_pairs = min(k, len(rmsds) - 1)
    #     pre_opt_idxs = indices[np.argpartition(rmsds, pre_opt_pairs)[:pre_opt_pairs]]
    #     args = [
    #         (idxs, rc_conformers[idxs[0]].coordinates, pc_conformers[idxs[1]].coordinates) for idxs in pre_opt_idxs
    #     ]
    #     with ProcessPoolExecutor(max_workers=int(settings['n_processes'] * settings['xtb_n_cores'])) as executor:
    #         results = list(tqdm(executor.map(compute_optimal_rmsd, args), total=len(args), desc="Computing optimal complex RMSD scores"))
    #     idxs = np.array([result[0] for result in results])
    #     rmsds = np.array([result[1] for result in results])
    #     opt_idxs = idxs[np.argpartition(rmsds, n_reactant_product_pairs)[:n_reactant_product_pairs]]

    # return opt_idxs

def compute_optimal_coordinates(
    rc_coordinates: np.ndarray,
    pc_coordinates: np.ndarray
) -> np.ndarray:
    assert rc_coordinates.shape == pc_coordinates.shape

    p_mat = np.array(rc_coordinates, copy=True)
    p_mat -= np.average(p_mat, axis=0)

    q_mat = np.array(pc_coordinates, copy=True)
    q_mat -= np.average(q_mat, axis=0)

    rot_mat = get_rot_mat_kabsch(p_mat, q_mat)

    fitted_coords = np.dot(rot_mat, p_mat.T).T
    return fitted_coords
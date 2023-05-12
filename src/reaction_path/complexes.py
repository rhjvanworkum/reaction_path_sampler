"""
Each reaction path needs a reactant + product complex
"""
import numpy as np
from typing import Dict, List, Tuple, Any
import os
import time

import autode as ade
from autode.geom import get_rot_mat_kabsch
from autode.conformers.conformer import Conformer

from src.reactive_complex_sampler import ReactiveComplexSampler
from src.utils import read_trajectory_file, remove_whitespaces_from_xyz_strings, xyz_string_to_autode_atoms


def generate_reaction_complexes(
    smiles_strings: List[str],
    solvent: str,
    settings: Any,
    save_path: str,
    compute_conformers: bool = False
) -> Tuple[List[Conformer]]:
    """
    Function that creates molecular complexes from provided SMILES strings
    :args smiles_strings: List of SMILES strings
    :args solvent: Solvent to be used
    :args settings: Settings object
    :args save_path: In which folder to save conformers of the molecular complex

    Returns:

    """
    rps = ReactiveComplexSampler(
        smiles_strings=smiles_strings,
        solvent=solvent,
        settings=settings
    )

    complex = rps._get_ade_complex()

    species_complex_mapping = None
    if len(smiles_strings) > 1:
        species_complex_mapping = {}
        mols = [ade.Molecule(smiles=smi) for smi in smiles_strings]
        tot_atoms = 0
        for idx, mol in enumerate(mols):
            species_complex_mapping[idx] = np.arange(tot_atoms, tot_atoms + len(mol.atoms))
            tot_atoms += len(mol.atoms)

    if compute_conformers:
        if os.path.exists(save_path):
            conformers, _ = read_trajectory_file(save_path)
            conformer_list = [Conformer(
                atoms=xyz_string_to_autode_atoms(structure), 
                charge=complex.charge, 
                mult=complex.mult
            ) for structure in conformers]

        else:
            t = time.time()
            complexes = rps._sample_initial_complexes(complex)

            print(f'time to do autode sampling: {time.time() - t}')
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

        return complex, conformer_list, len(smiles_strings), species_complex_mapping
    
    else:

        return complex, len(smiles_strings), species_complex_mapping



def select_promising_reactant_product_pairs(
    rc_conformers: List[Conformer],
    pc_conformers: List[Conformer],
    species_complex_mapping: Any,
    bonds: Any,
    settings: Any
) -> List[Tuple[int]]:
    """
    Selects the most "promising" reactant & product complex pairs based on 
    some scoring function
    """
    n_reactant_product_pairs = settings['max_n_reactant_product_pairs']

    indices, scores = [], []
    for i in range(len(rc_conformers)):
        for j in range(len(pc_conformers)):
            idx = (i, j)
            rc_coords = rc_conformers[i].coordinates
            pc_coords = pc_conformers[j].coordinates

            score = rmsd_score(rc_coords, pc_coords, align_complexes=True)
            # score = weighted_rmsd_score(rc_conformers[0].atomic_masses, rc_coords, pc_coords, align_complexes=True)
            # score = active_bond_distance_score(rc_coords, pc_coords, bonds, align_complexes=True)
            # score = subsystems_aligned_rmsd_score(rc_coords, pc_coords, species_complex_mapping, align_complexes=True)

            indices.append(idx)
            scores.append(score)
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
    
    return opt_idxs
    

""" Scoring Fucntions """
def rmsd_score(
    rc_coords: np.ndarray,
    pc_coords: np.ndarray,
    align_complexes: bool = True
) -> float:
    """
    Returns the RMSD between reactant & product complexes
    """
    if align_complexes:
        rc_coords_aligned = compute_optimal_coordinates(
            rc_coords, pc_coords
        )
    else:
        rc_coords_aligned = rc_coords
    score = np.sqrt(np.mean((pc_coords - rc_coords_aligned)**2))
    return score

def weighted_rmsd_score(
    rc_coords: np.ndarray,
    pc_coords: np.ndarray,
    weights: np.ndarray,
    align_complexes: bool = True
) -> float:
    """
    Returns a weighted RMSD between reactant & product complexes
    """
    if align_complexes:
        rc_coords_aligned = compute_optimal_coordinates(
            rc_coords, pc_coords
        )
    else:
        rc_coords_aligned = rc_coords
    score = np.sqrt(np.sum(weights * np.linalg.norm((pc_coords - rc_coords_aligned)**2, axis=-1)) / np.sum(weights))
    return score

def active_bond_distance_score(
    rc_coords: np.ndarray,
    pc_coords: np.ndarray,
    bonds: List[Tuple[int]],
    align_complexes: bool = True
) -> float:
    """
    Returns the bond lenghts of the active bonds in the reactant complexes
    """
    if align_complexes:
        rc_coords_aligned = compute_optimal_coordinates(
            rc_coords, pc_coords
        )
    else:
        rc_coords_aligned = rc_coords
    score = sum([
        np.sqrt(np.mean((rc_coords_aligned[bond[0], :] - rc_coords_aligned[bond[1], :])**2)) for bond in bonds
    ])
    return score

def subsystems_aligned_rmsd_score(
    rc_coords: np.ndarray,
    pc_coords: np.ndarray,
    atom_subsystem_mapping: Dict[int, List[int]],
    align_complexes: bool = True
) -> float:
    """
    Returns the RMSD between a set of subsytems of the 
    reactant & product complexes, e.g. the separate species.
    """
    score = 0
    for _, idxs in atom_subsystem_mapping.items():
        sub_system_rc_coords = rc_coords[idxs]
        sub_system_pc_coords = pc_coords[idxs]
        if align_complexes:
            sub_system_rc_coords_aligned = compute_optimal_coordinates(
                sub_system_rc_coords, sub_system_pc_coords
            )
        else:
            sub_system_rc_coords_aligned = sub_system_rc_coords
        score += np.sqrt(np.mean((sub_system_pc_coords - sub_system_rc_coords_aligned)**2))
    return score


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
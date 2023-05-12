"""
Each reaction path needs a mapping from the atom indexing in the reactant graph & product graph
"""

from concurrent.futures import ProcessPoolExecutor
import numpy as np
from typing import Dict, List, Tuple, Any
from tqdm import tqdm

import networkx as nx

import autode as ade
from autode.conformers.conformer import Conformer
from autode.bond_rearrangement import get_bond_rearrangs, BondRearrangement
from autode.mol_graphs import reac_graph_to_prod_graph

from src.reaction_path.complexes import compute_optimal_coordinates


def get_reaction_isomorphisms(
    rc_complex: ade.Species,
    pc_complex: ade.Species,
) -> Tuple[BondRearrangement, Dict[int, int], int]:
    """
    This function returns all possible isomorphisms between the reactant & product graphs
    """
    # TODO: I think there is some function here that can time out

    for idx, reaction_complexes in enumerate([
        [rc_complex, pc_complex],
        [pc_complex, rc_complex],
    ]):
        bond_rearrs = get_bond_rearrangs(reaction_complexes[1], reaction_complexes[0], name='test')
        if bond_rearrs is not None:
            for bond_rearr in bond_rearrs:
                graph1 = reaction_complexes[0].graph
                graph2 = reac_graph_to_prod_graph(reaction_complexes[1].graph, bond_rearr)
                mappings = []
                for isomorphism in nx.vf2pp_all_isomorphisms(
                    graph1, 
                    graph2, 
                    node_label="atom_label"
                ):
                    mappings.append(isomorphism)

                mappings = [dict(s) for s in set(frozenset(d.items()) for d in mappings)]

                if len(mappings) > 0:
                    return bond_rearr, mappings, idx


def compute_isomorphism_score(args) -> float:
    isomorphism, species_complex_mapping, coords1, coords2 = args

    # remap coords based on isomorphism
    ordering = np.array(sorted(isomorphism, key=isomorphism.get))
    coords2 = coords2[ordering, :]   

    # remap species mapping based on isomorphism
    for key, value in species_complex_mapping.items():
        species_complex_mapping[key] = np.array([isomorphism[idx] for idx in value])

    rmsd = 0
    for _, idxs in species_complex_mapping.items():
        sub_system_rc_coords = coords1[idxs, :]
        sub_system_pc_coords = coords2[idxs, :]
        sub_system_rc_coords_aligned = compute_optimal_coordinates(
            sub_system_rc_coords, sub_system_pc_coords
        )
        rmsd += np.sqrt(np.mean((sub_system_pc_coords - sub_system_rc_coords_aligned)**2))
        
    return rmsd

def select_ideal_isomorphism(
    rc_conformers: List[Conformer],
    pc_conformers: List[Conformer],
    rc_species_complex_mapping, 
    pc_species_complex_mapping,
    isomorphism_idx: int,
    isomorphisms: List[Dict[int, int]],
    settings: Any
) -> Dict[int, int]:
    """
    This function select the "true" graph isomorpism based on some RMSD calculations between
    product & reactant complex conformers
    """
    
    scores = []   
    if isomorphism_idx == 0:
        coords_no_remap = pc_conformers[0].coordinates
        coords_to_remap = rc_conformers[0].coordinates
    elif isomorphism_idx == 1:
        coords_no_remap = rc_conformers[0].coordinates
        coords_to_remap = pc_conformers[0].coordinates
    else:
        raise ValueError(f"isomorphism idx can not be {isomorphism_idx}")

    species_complex_mapping = [rc_species_complex_mapping, pc_species_complex_mapping][isomorphism_idx]

    args = [
        (isomorphism, species_complex_mapping, coords_no_remap, coords_to_remap) for isomorphism in isomorphisms
    ]
    with ProcessPoolExecutor(max_workers=int(settings['n_processes'] * settings['xtb_n_cores'])) as executor:
        scores = list(tqdm(executor.map(compute_isomorphism_score, args), total=len(args), desc="Computing isomorphisms score"))

    return isomorphisms[np.argmin(scores)]






"""
New isomorphism selection code
"""
def compute_isomorphism_score_single(args) -> float:
    isomorphism, coords1, coords2 = args

    # remap coords based on isomorphism
    ordering = np.array(sorted(isomorphism, key=isomorphism.get))
    coords2 = coords2[ordering, :]   

    rc_coords = coords1
    pc_coords = coords2
    rc_coords_aligned = compute_optimal_coordinates(
        rc_coords, pc_coords
    )
    score = np.sqrt(np.mean((pc_coords - rc_coords_aligned)**2))
    return score


def select_ideal_pair_isomorphism(
    rc_conformer: List[Conformer],
    pc_conformer: List[Conformer],
    isomorphism_idx: int,
    isomorphisms: List[Dict[int, int]],
    settings: Any
) -> Dict[int, int]:
    """
    This function select the "true" graph isomorpism based on some RMSD calculations between
    product & reactant complex conformers
    """
    
    scores = []   
    if isomorphism_idx == 0:
        coords_no_remap = pc_conformer.coordinates
        coords_to_remap = rc_conformer.coordinates
    elif isomorphism_idx == 1:
        coords_no_remap = rc_conformer.coordinates
        coords_to_remap = pc_conformer.coordinates
    else:
        raise ValueError(f"isomorphism idx can not be {isomorphism_idx}")

    args = [
        (isomorphism, coords_no_remap, coords_to_remap) for isomorphism in isomorphisms
    ]
    with ProcessPoolExecutor(max_workers=int(settings['n_processes'] * settings['xtb_n_cores'])) as executor:
        scores = list(tqdm(executor.map(compute_isomorphism_score_single, args), total=len(args), disable=True, desc="Computing isomorphisms score"))

    return isomorphisms[np.argmin(scores)]
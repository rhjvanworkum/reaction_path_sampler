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

def compute_rmsd(coords1, coords2):
    return np.sqrt(np.mean((coords2 - coords1)**2))

def compute_isomorphism_score(args) -> float:
    isomorphism, coords1, coords2 = args
    ordering = np.array(sorted(isomorphism, key=isomorphism.get))
    coords2 = coords2[:, ordering, :]    
    rmsds = []
    for i in range(coords1.shape[0]):
        for j in range(coords2.shape[0]):
            rmsds.append(compute_rmsd(
                coords1[i],
                coords2[j]
            ))
    return np.min(np.array(rmsds))

def select_ideal_isomorphism(
    rc_conformers: List[Conformer],
    pc_conformers: List[Conformer],
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
        coords_no_remap = np.stack([
            pc_conformers[i] .coordinates for i in np.random.choice(len(pc_conformers), size=min(100, len(pc_conformers)), replace=False)
        ])
        coords_to_remap = np.stack([
            rc_conformers[i] .coordinates for i in np.random.choice(len(rc_conformers), size=min(100, len(rc_conformers)), replace=False)
        ])
    elif isomorphism_idx == 1:
        coords_no_remap = np.stack([
            rc_conformers[i] .coordinates for i in np.random.choice(len(rc_conformers), size=min(100, len(rc_conformers)), replace=False)
        ])
        coords_to_remap = np.stack([
            pc_conformers[i] .coordinates for i in np.random.choice(len(pc_conformers), size=min(100, len(pc_conformers)), replace=False)
        ])
    else:
        raise ValueError(f"isomorphism idx can not be {isomorphism_idx}")
    
    args = [
        (isomorphism, coords_no_remap, coords_to_remap) for isomorphism in isomorphisms
    ]
    with ProcessPoolExecutor(max_workers=int(settings['n_processes'] * settings['xtb_n_cores'])) as executor:
        scores = list(tqdm(executor.map(compute_isomorphism_score, args), total=len(args), desc="Computing isomorphisms score"))

    return isomorphisms[np.argmin(scores)]
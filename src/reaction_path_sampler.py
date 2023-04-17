import numpy as np
from typing import Dict, List, Tuple, Any
import os
import time
import networkx as nx

import autode as ade
from autode.values import Distance
from autode.species import Complex
from autode.conformers.conformer import Conformer
from autode.atoms import Atoms
from autode.input_output import atoms_to_xyz_file
from autode.bond_rearrangement import get_bond_rearrangs
from autode.mol_graphs import reac_graph_to_prod_graph

from src.reactive_complex_sampler import ReactiveComplexSampler
from src.utils import read_trajectory_file, remove_whitespaces_from_xyz_strings, xyz_string_to_autode_atoms


class ReactionPathSampler:

    def __init__(
        self,
        settings: Dict[str, Any]
    ) -> None:
        self.settings = settings

    def sample_reaction_paths(self) -> None:
        pass

    def _generate_reactive_complexes(
        self,
        smiles_strings: List[str],
        solvent: str,
        save_path: str
    ) -> Tuple[Complex, List[Conformer]]:
        rcs = ReactiveComplexSampler(
            smiles_strings=smiles_strings,
            solvent=solvent,
            settings=self.settings
        )

        if os.path.exists(save_path):
            complex = rcs._get_ade_complex()
            conformers, _ = read_trajectory_file(save_path)
            conformer_list = [Conformer(
                atoms=xyz_string_to_autode_atoms(structure), 
                charge=complex.charge, 
                mult=complex.mult
            ) for structure in conformers]

        else:
            t = time.time()
            complexes = rcs._sample_initial_complexes()
            print(f'time to do autode sampling: {time.time() - t}')

            conformer_list = []
            conformer_xyz_list = []
            for complex in complexes:
                conformers = rcs.sample_reaction_complexes(complex=complex)
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
    
    def get_reaction_isomorphisms(
        self,
        rc_complex: ade.Species,
        pc_complex: ade.Species,
    ) -> Tuple[Dict[int, int], int]:
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

                    if len(mappings) > 0:
                        return mappings, idx


    
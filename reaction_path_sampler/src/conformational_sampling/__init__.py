
import logging
from typing import Any, List
import autode as ade
from autode.conformers.conformer import Conformer
import numpy as np

from reaction_path_sampler.src.interfaces.CREST import crest_driver
from reaction_path_sampler.src.interfaces.lewis import compute_adjacency_matrix
from reaction_path_sampler.src.molecule import Molecule, parse_geometry_from_xyz_string
from reaction_path_sampler.src.xyz2mol import get_canonical_smiles_from_xyz_string, get_canonical_smiles_from_xyz_string_ob
from reaction_path_sampler.src.utils import get_canonical_smiles

from reaction_path_sampler.src.xyz2mol import xyz2AC, __ATOM_LIST__

class ConformerSampler:

    def __init__(
        self,
        smiles_strings: List[str],
        settings: Any,
        solvent: str
    ) -> None:
        self.smiles_strings = smiles_strings
        self.settings = settings
        self.solvent = solvent
        
    def sample_conformers(self, initial_geometry: ade.Species) -> List[str]:
        raise NotImplementedError
    
    def _prune_conformers(
        self,
        initial_geometry: Molecule,
        conformers: List[str],
        use_graph_pruning: bool,
        use_cregen_pruning: bool,
        init: str = ""
    ) -> List[str]:
        """
        Prunes a set of conformers using CREST CREGEN
        """
        if use_cregen_pruning:
            conformers = crest_driver(
                ref_structure=initial_geometry.to_xyz_string(),
                ensemble_structures='\n'.join(conformers),
                ref_energy_threshold=self.settings[f"{init}ref_energy_threshold"][len(self.smiles_strings)],
                rmsd_threshold=self.settings[f"{init}rmsd_threshold"][len(self.smiles_strings)],
                conf_energy_threshold=self.settings[f"{init}conf_energy_threshold"][len(self.smiles_strings)],
                rotational_threshold=self.settings[f"{init}rotational_threshold"][len(self.smiles_strings)],
            )

        print(len(conformers))
        
        if use_graph_pruning:
            pruned_conformers = []

            # TODO: maybe we can do this by means of adjacency matrix?
            # # adcompute_adjacency_matrix
            if isinstance(initial_geometry, Molecule):
                symbols, coords = initial_geometry.to_geometry()
            elif isinstance(initial_geometry, Conformer):
                symbols, coords = initial_geometry.atomic_symbols, initial_geometry.coordinates
            
            symbols = [
                __ATOM_LIST__.index(s.lower()) + 1 for s in symbols
            ]
            ref_adj_matrix, _ = xyz2AC(symbols, coords, initial_geometry.charge, use_huckel=True)
            for conformer in conformers:
                try:
                    symbols, coords = parse_geometry_from_xyz_string(conformer)
                    symbols = [
                        __ATOM_LIST__.index(s.lower()) + 1 for s in symbols
                    ]
                    adj_matrix, _ = xyz2AC(symbols, coords, initial_geometry.charge, use_huckel=True)

                    if np.array_equal(ref_adj_matrix, adj_matrix):
                        pruned_conformers.append(conformer)

                except Exception as e:
                    print('Exception occured during XYZ -> adjecency matrix: \n', e)
                    continue


            # smiles_list = [get_canonical_smiles(smi) for smi in self.smiles_strings]
            # for conformer in conformers:
            #     try:
            #         conf_smiles_list = get_canonical_smiles_from_xyz_string(conformer, initial_geometry.charge)
            #         print(conf_smiles_list, smiles_list)
            #         if set(conf_smiles_list) == set(smiles_list):
            #             pruned_conformers.append(conformer)
            #     except Exception as e:
            #         print('Exception occured during XYZ -> SMILES: \n', e)
            #         continue
            # conformers = pruned_conformers
        
        return conformers
    
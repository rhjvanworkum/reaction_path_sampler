
import logging
from typing import Any, List
import autode as ade
from autode.conformers.conformer import Conformer
import networkx
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, Draw

from reaction_path_sampler.src.interfaces.CREST import crest_driver
from reaction_path_sampler.src.graphs.lewis import compute_adjacency_matrix
from reaction_path_sampler.src.interfaces.xtb_utils import comp_ad_mat_xtb
from reaction_path_sampler.src.molecular_system import MolecularSystem
from reaction_path_sampler.src.molecule import parse_geometry_from_xyz_string
from reaction_path_sampler.src.graphs.xyz2mol import get_canonical_smiles_from_xyz_string, get_canonical_smiles_from_xyz_string_ob
from reaction_path_sampler.src.utils import comp_adj_mat, get_canonical_smiles

from reaction_path_sampler.src.graphs.xyz2mol import xyz2AC, __ATOM_LIST__
from reaction_path_sampler.src.visualization.plotly import plot_networkx_mol_graph

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
        mol: MolecularSystem,
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
                ref_structure=mol.init_geometry_xyz_string,
                ensemble_structures='\n'.join(conformers),
                ref_energy_threshold=self.settings[f"{init}ref_energy_threshold"][len(self.smiles_strings)],
                rmsd_threshold=self.settings[f"{init}rmsd_threshold"][len(self.smiles_strings)],
                conf_energy_threshold=self.settings[f"{init}conf_energy_threshold"][len(self.smiles_strings)],
                rotational_threshold=self.settings[f"{init}rotational_threshold"][len(self.smiles_strings)],
            )
    
        if use_graph_pruning:
            pruned_conformers = []

            for conformer in conformers:
                try:
                    adj_matrix = comp_ad_mat_xtb(
                        xyz_string=conformer,
                        charge=mol.charge,
                        mult=mol.mult,
                        solvent=self.solvent,
                    )

                    # TODO: remove this
                    print(np.sum(np.abs(adj_matrix - mol.connectivity_matrix)))

                    if np.sum(np.abs(mol.connectivity_matrix - adj_matrix)) <= self.settings['graph_pruning_threshold']:
                        pruned_conformers.append(conformer)
                except:
                    continue
            
            conformers = pruned_conformers
        
        return conformers
    
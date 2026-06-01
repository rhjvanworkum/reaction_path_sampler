from typing import Any

import autode as ade
import numpy as np

from reaction_path_sampler.interfaces.CREST import crest_driver
from reaction_path_sampler.interfaces.xtb_utils import comp_ad_mat_xtb
from reaction_path_sampler.molecular_system import MolecularSystem


class ConformerSampler:
    def __init__(self, smiles_strings: list[str], settings: Any, solvent: str) -> None:
        self.smiles_strings = smiles_strings
        self.settings = settings
        self.solvent = solvent

    def sample_conformers(self, initial_geometry: ade.Species) -> list[str]:
        raise NotImplementedError

    def _prune_conformers(
        self,
        mol: MolecularSystem,
        conformers: list[str],
        use_graph_pruning: bool,
        use_cregen_pruning: bool,
        init: str = "",
    ) -> list[str]:
        """
        Prunes a set of conformers using CREST CREGEN
        """
        if use_cregen_pruning:
            conformers = crest_driver(
                ref_structure=mol.init_geometry_xyz_string,
                ensemble_structures="\n".join(conformers),
                ref_energy_threshold=self.settings[f"{init}ref_energy_threshold"][
                    len(self.smiles_strings)
                ],
                rmsd_threshold=self.settings[f"{init}rmsd_threshold"][len(self.smiles_strings)],
                conf_energy_threshold=self.settings[f"{init}conf_energy_threshold"][
                    len(self.smiles_strings)
                ],
                rotational_threshold=self.settings[f"{init}rotational_threshold"][
                    len(self.smiles_strings)
                ],
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

                    if (
                        np.sum(np.abs(mol.connectivity_matrix - adj_matrix))
                        <= self.settings["graph_pruning_threshold"]
                    ):
                        pruned_conformers.append(conformer)
                except Exception:
                    continue

            conformers = pruned_conformers

        return conformers

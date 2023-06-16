from typing import Dict, List, Tuple, Any
import os

import autode as ade
from autode.species import Complex

from reaction_path_sampler.src.reaction_path.complexes import generate_reaction_complex
from reaction_path_sampler.src.reaction_path.mapped_complex import generate_mapped_reaction_complexes
from reaction_path_sampler.src.utils import set_autode_settings


class ReactionSampler:

    def __init__(
        self,
        settings: Dict[str, Any]
    ) -> None:
        self.settings = settings

        # create output dir
        if not os.path.exists(self.settings['output_dir']):
            os.makedirs(self.settings['output_dir'])

        # set autode settings
        set_autode_settings(settings)

        self._rc_complex = None
        self._pc_complex = None

        self.solvent = self.settings['solvent']
        self._charge = None
        self._mult = None

        self._bond_rearr = None
        self._isomorphism_idx = None

    # @property
    # def rc_complex(self) -> Complex:
    #     if self._rc_complex is None:
    #         raise ValueError('Reactant complex not set, call generate_reaction_complexes() first')
    #     return self._rc_complex

    # @property
    # def pc_complex(self) -> Complex:
    #     if self._pc_complex is None:
    #         raise ValueError('Reactant complex not set, call generate_reaction_complexes() first')
    #     return self._pc_complex

    # @property
    # def charge(self) -> int:
    #     if self._charge is None:
    #         raise ValueError('Charge not set, call generate_reaction_complexes() first')
    #     return self._charge

    # @property
    # def mult(self) -> int:
    #     if self._mult is None:
    #         raise ValueError('Mult not set, call generate_reaction_complexes() first')
    #     return self._mult

    def generate_reaction_complexes(self) -> None:
        """
        Generate reactant and product complexes using autodE.
        """
        if self.settings['use_rxn_mapper']:
            rc_complex, pc_complex = generate_mapped_reaction_complexes(
                self.settings['reactant_smiles'],
                self.settings['product_smiles'],
                solvent=self.solvent
            )
        else:
            rc_complex = generate_reaction_complex(self.settings['reactant_smiles'])
            pc_complex = generate_reaction_complex(self.settings['product_smiles'])

        assert rc_complex.charge == pc_complex.charge
        assert pc_complex.mult == pc_complex.mult

        self._rc_complex = rc_complex
        self._pc_complex = pc_complex
        self._charge = rc_complex.charge
        self._mult = rc_complex.mult
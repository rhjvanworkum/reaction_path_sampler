from concurrent.futures import ProcessPoolExecutor
import logging
import autode as ade
from autode.values import Distance
from autode.species import Complex
from autode.conformers.conformer import Conformer

import time
from tqdm import tqdm
from typing import List, Any, Dict, Optional, Union


from reaction_path_sampler.src.conformational_sampling import ConformerSampler
from reaction_path_sampler.src.interfaces.XTB import xtb_driver
from reaction_path_sampler.src.interfaces.xtb_utils import compute_wall_radius, get_atom_constraints, get_fixing_constraints, get_metadynamics_settings, get_wall_constraint
from reaction_path_sampler.src.molecule import Molecule
from reaction_path_sampler.src.utils import get_tqdm_disable, xyz_string_to_autode_atoms

def optimize_autode_conformer(args):
    xyz_string, charge, mult, solvent, method, xcontrol_settings, cores = args
    opt_xyz = xtb_driver(
        xyz_string,
        charge,
        mult,
        "opt",
        method=method,
        solvent=solvent,
        xcontrol_settings=xcontrol_settings,
        n_cores=cores
    )
    try:
        return Conformer(
            atoms=xyz_string_to_autode_atoms(opt_xyz), 
            charge=charge, 
            mult=mult
        )
    except:
        return None


def optimize_conformer(args):
    conformer, complex, solvent, method, xcontrol_settings, cores = args
    return xtb_driver(
        conformer,
        complex.charge,
        complex.mult,
        "opt",
        method=method,
        solvent=solvent,
        xcontrol_settings=xcontrol_settings,
        n_cores=cores
    )

class MetadynConformerSampler(ConformerSampler):

    def __init__(
        self,
        smiles_strings: int,
        solvent: str,
        settings: Dict[str, Any]
    ) -> None:
        super().__init__(
            smiles_strings=smiles_strings,
            settings=settings,
            solvent=solvent
        )

    def sample_ts_conformers(
        self,
        complex: Molecule,
        fixed_atoms: Optional[List[int]] = None, 
    ) -> List[str]:     
        self.settings["wall_radius"] = compute_wall_radius(
            complex=complex,
            settings=self.settings
        )

        # 1. sample conformers
        t = time.time()
        confs = self._sample_metadynamics_conformers(complex=complex, post_fix="_ts", fixed_atoms=fixed_atoms)
        logging.info(f'metadyn sampling: {time.time() - t}')
        logging.info(f'metadyn sampled n conformers: {len(confs)}')

        # 2. optimize conformers
        t = time.time()
        confs = self._optimize_conformers(
            complex=complex,
            conformers=confs,
            fixed_atoms=fixed_atoms
        )
        logging.info(f'optimizing conformers: {time.time() - t}')

        # 3. prune conformer set
        t = time.time()
        confs = self._prune_conformers(
            initial_geometry=complex,
            conformers=confs,
            use_graph_pruning=False,
            use_cregen_pruning=True,
            init="ts_"
        )
        logging.info(f'pruning conformers: {time.time() - t}')
        logging.info(f'conformers after pruning: {len(confs)}\n\n')

        return confs

    def sample_conformers(
        self, 
        initial_geometry: Union[ade.Species, Molecule],
        fixed_atoms: Optional[List[int]] = None, 
    ) -> List[str]:
        if isinstance(initial_geometry, ade.Species):
            complex = Molecule.from_autode_mol(initial_geometry)      
        else:
            complex = initial_geometry
              
        self.settings["wall_radius"] = compute_wall_radius(
            complex=complex,
            settings=self.settings
        )

        # 1. sample conformers
        t = time.time()
        confs = self._sample_metadynamics_conformers(complex=complex, fixed_atoms=fixed_atoms)
        logging.info(f'metadyn sampling: {time.time() - t}')
        logging.info(f'metadyn sampled n conformers: {len(confs)}')

        if len(confs) < 10:
            confs = self._sample_metadynamics_conformers(complex=complex, post_fix="_tight", fixed_atoms=fixed_atoms)
            logging.info(f'metadyn sampling: {time.time() - t}')
            logging.info(f'metadyn sampled n conformers: {len(confs)}')

        # 2. prune conformer set
        t = time.time()
        confs = self._prune_conformers(
            initial_geometry=complex,
            conformers=confs,
            use_graph_pruning=False,
            use_cregen_pruning=True,
            init="init_"
        )
        logging.info(f'pruning conformers: {time.time() - t}')
        logging.info(f'conformers after pruning: {len(confs)}\n\n')


        # 3. optimize conformers
        t = time.time()
        confs = self._optimize_conformers(
            complex=complex,
            conformers=confs,
        )
        logging.info(f'optimizing conformers: {time.time() - t}')


        # 4. prune conformer set
        t = time.time()
        confs = self._prune_conformers(
            initial_geometry=complex,
            conformers=confs,
            use_graph_pruning=True,
            use_cregen_pruning=self.settings['use_cregen_pruning']
        )
        logging.info(f'pruning conformers: {time.time() - t}')
        logging.info(f'conformers after pruning: {len(confs)}\n\n')

        return confs
    
    def _sample_metadynamics_conformers(
        self, 
        complex: Molecule, 
        post_fix: str = "",
        fixed_atoms: Optional[List[int]] = None
    ) -> List[str]:
        """
        Sample conformers of a Molecule object complex.
        Returns a list of xyz strings containing conformers
        """
        xcontrol_settings = get_wall_constraint(
            self.settings["wall_radius"]
        )
        xcontrol_settings += get_metadynamics_settings(
            complex,
            self.settings,
            len(self.smiles_strings),
            post_fix=post_fix
        )
        if fixed_atoms is not None:
            xcontrol_settings += get_atom_constraints(
                atom_idxs=fixed_atoms,
                force_constant=0.5,
                reference_file='ref.xyz'
            )

        structures, _ = xtb_driver(
            complex.to_xyz_string(),
            complex.charge,
            complex.mult,
            "metadyn",
            method=self.settings['xtb_method'],
            solvent=self.solvent,
            xcontrol_settings=xcontrol_settings,
            n_cores=self.settings['xtb_n_cores_metadyn']
        )
        return structures

    def _optimize_conformers(
        self,
        complex: Molecule,
        conformers: List[str],
        fixed_atoms: Optional[List[int]] = None
    ) -> List[str]:
        """
        Optimizes a set of conformers
        """
        xcontrol_settings = get_wall_constraint(
            self.settings["wall_radius"]
        )
        if fixed_atoms is not None:
            xcontrol_settings += get_fixing_constraints(
                atom_idxs=fixed_atoms,
            )
        
        arguments = [
            (conf, complex, self.solvent, self.settings['xtb_method'], xcontrol_settings, self.settings['xtb_n_cores']) for conf in conformers
        ]
        with ProcessPoolExecutor(max_workers=self.settings['n_processes']) as executor:
            opt_conformers = list(tqdm(executor.map(optimize_conformer, arguments), total=len(arguments), desc="Optimizing conformers", disable=get_tqdm_disable()))

        opt_conformers = list(filter(lambda x: x is not None, opt_conformers))
        return opt_conformers


if __name__ == "__main__":
    import yaml
    import autode as ade

    ade.Config.n_cores = 4
    ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'
    ade.Config.rmsd_threshold = Distance(0.5, units="Å")

    with open('/home/ruard/code/test_reactions/settings.yaml', "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)
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
from reaction_path_sampler.src.molecular_system import MolecularSystem
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
    conformer, mol, solvent, method, xcontrol_settings, cores = args
    return xtb_driver(
        conformer,
        mol.charge,
        mol.mult,
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
        mol: MolecularSystem,
        fixed_atoms: Optional[List[int]] = None, 
    ) -> List[str]:     
        self.settings["wall_radius"] = compute_wall_radius(
            mol=mol,
            settings=self.settings
        )

        # 1. sample conformers
        t = time.time()
        confs = self._sample_metadynamics_conformers(mol=mol, post_fix="", fixed_atoms=fixed_atoms)
        print(f'metadyn sampling: {time.time() - t}')
        print(f'metadyn sampled n conformers: {len(confs)}')

        # 2. optimize conformers
        t = time.time()
        confs = self._optimize_conformers(
            mol=mol,
            conformers=confs,
            fixed_atoms=fixed_atoms
        )
        print(f'optimizing conformers: {time.time() - t}')

        # 3. prune conformer set
        t = time.time()
        confs = self._prune_conformers(
            mol=mol,
            conformers=confs,
            use_graph_pruning=False,
            use_cregen_pruning=True,
            init="ts_"
        )
        print(f'pruning conformers: {time.time() - t}')
        print(f'conformers after pruning: {len(confs)}\n\n')

        return confs

    def sample_conformers(
        self, 
        mol: MolecularSystem,
        fixed_atoms: Optional[List[int]] = None, 
    ) -> List[str]:    
        self.settings["wall_radius"] = compute_wall_radius(
            mol=mol,
            settings=self.settings
        )

        # 1. sample conformers
        t = time.time()
        confs = self._sample_metadynamics_conformers(mol=mol, fixed_atoms=fixed_atoms)
        print(f'metadyn sampling: {time.time() - t}')
        print(f'metadyn sampled n conformers: {len(confs)}')

        if len(confs) < 10:
            confs = self._sample_metadynamics_conformers(mol=mol, post_fix="_tight", fixed_atoms=fixed_atoms)
            print(f'metadyn sampling: {time.time() - t}')
            print(f'metadyn sampled n conformers: {len(confs)}')

        # 2. prune conformer set
        t = time.time()
        confs = self._prune_conformers(
            mol=mol,
            conformers=confs,
            use_graph_pruning=False,
            use_cregen_pruning=True,
            init="init_"
        )
        print(f'pruning conformers: {time.time() - t}')
        print(f'conformers after pruning: {len(confs)}\n\n')


        # 3. optimize conformers
        t = time.time()
        confs = self._optimize_conformers(
            mol=mol,
            conformers=confs,
        )
        print(f'optimizing conformers: {time.time() - t}')

        # 4. prune conformer set
        t = time.time()
        confs = self._prune_conformers(
            mol=mol,
            conformers=confs,
            use_graph_pruning=True,
            use_cregen_pruning=self.settings['use_cregen_pruning']
        )
        print(f'pruning conformers: {time.time() - t}')
        print(f'conformers after pruning: {len(confs)}\n\n')

        return confs
    
    def _sample_metadynamics_conformers(
        self, 
        mol: MolecularSystem, 
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
            mol,
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
            mol.init_geometry_xyz_string,
            mol.charge,
            mol.mult,
            "metadyn",
            method=self.settings['xtb_method'],
            solvent=self.solvent,
            xcontrol_settings=xcontrol_settings,
            n_cores=self.settings['xtb_n_cores_metadyn']
        )
        return structures

    def _optimize_conformers(
        self,
        mol: MolecularSystem,
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
            (conf, mol, self.solvent, self.settings['xtb_method'], xcontrol_settings, self.settings['xtb_n_cores']) for conf in conformers
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
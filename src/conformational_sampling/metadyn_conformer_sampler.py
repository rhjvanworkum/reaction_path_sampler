from concurrent.futures import ProcessPoolExecutor
import autode as ade
from autode.values import Distance
from autode.species import Complex
from autode.conformers.conformer import Conformer


import time
from tqdm import tqdm
from typing import List, Literal, Any, Dict
import numpy as np


from src.conformational_sampling import ConformerSampler
from src.interfaces.CREST import crest_driver
from src.interfaces.XTB import xtb_driver
from src.interfaces.xtb_utils import compute_wall_radius, get_metadynamics_constraint, get_wall_constraint
from src.molecule import Molecule
from src.utils import autode_conf_to_xyz_string, get_canonical_smiles, xyz_string_to_autode_atoms, sort_complex_conformers_on_distance
from src.xyz2mol import get_canonical_smiles_from_xyz_string_ob

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
        super().__init__()
        self.smiles_strings = smiles_strings
        self.solvent = solvent
        self.settings = settings

    def sample_conformers(self, mol: ade.Species) -> List[str]:
        complex = Molecule.from_autode_mol(mol)
        self.settings["wall_radius"] = compute_wall_radius(
            complex=complex,
            settings=self.settings
        )

        # 1. sample conformers
        t = time.time()
        confs = self._sample_metadynamics_conformers(complex=complex)
        print(f'metadyn sampling: {time.time() - t}')
        print(f'metadyn sampled n conformers: {len(confs)}')

        if len(confs) < 10:
            confs = self._sample_metadynamics_conformers(complex=complex, post_fix="_tight")
            print(f'metadyn sampling: {time.time() - t}')
            print(f'metadyn sampled n conformers: {len(confs)}')


        # 2. prune conformer set
        t = time.time()
        confs = self._prune_conformers(
            complex=complex,
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
            complex=complex,
            conformers=confs,
        )
        print(f'optimizing conformers: {time.time() - t}')


        # 4. prune conformer set
        t = time.time()
        confs = self._prune_conformers(
            complex=complex,
            conformers=confs,
            use_graph_pruning=False,
            use_cregen_pruning=self.settings['use_cregen_pruning']
        )
        print(f'pruning conformers: {time.time() - t}')
        print(f'conformers after pruning: {len(confs)}\n\n')

        return confs
    
    def _sample_metadynamics_conformers(self, complex: Molecule, post_fix: str = "") -> List[str]:
        """
        Sample conformers of a Molecule object complex.
        Returns a list of xyz strings containing conformers
        """
        xcontrol_settings = get_wall_constraint(
            self.settings["wall_radius"]
        )
        xcontrol_settings += get_metadynamics_constraint(
            complex,
            self.settings,
            len(self.smiles_strings),
            post_fix=post_fix
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
    ) -> List[str]:
        """
        Optimizes a set of conformers
        """
        xcontrol_settings = get_wall_constraint(
            self.settings["wall_radius"]
        )
        
        arguments = [
            (conf, complex, self.solvent, self.settings['xtb_method'], xcontrol_settings, self.settings['xtb_n_cores']) for conf in conformers
        ]
        with ProcessPoolExecutor(max_workers=self.settings['n_processes']) as executor:
            opt_conformers = list(tqdm(executor.map(optimize_conformer, arguments), total=len(arguments), desc="Optimizing conformers"))

        opt_conformers = list(filter(lambda x: x is not None, opt_conformers))
        return opt_conformers
        
    def _prune_conformers(
        self,
        complex: Molecule,
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
                ref_structure=complex.to_xyz_string(),
                ensemble_structures='\n'.join(conformers),
                ref_energy_threshold=self.settings[f"{init}ref_energy_threshold"][len(self.smiles_strings)],
                rmsd_threshold=self.settings[f"{init}rmsd_threshold"][len(self.smiles_strings)],
                conf_energy_threshold=self.settings[f"{init}conf_energy_threshold"][len(self.smiles_strings)],
                rotational_threshold=self.settings[f"{init}rotational_threshold"][len(self.smiles_strings)],
            )
        

        # TODO: maybe we can do this by means of adjacency matrix?
        if use_graph_pruning:
            pruned_conformers = []
            smiles_list = [get_canonical_smiles(smi) for smi in self.smiles_strings]
            
            for conformer in conformers:
                try:
                    conf_smiles_list = get_canonical_smiles_from_xyz_string_ob(conformer)
                    if set(conf_smiles_list) == set(smiles_list):
                        pruned_conformers.append(conformer)
                except Exception as e:
                    continue
            conformers = pruned_conformers
        
        return conformers


if __name__ == "__main__":
    import yaml
    import autode as ade

    ade.Config.n_cores = 4
    ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'
    ade.Config.rmsd_threshold = Distance(0.5, units="Å")

    with open('/home/ruard/code/test_reactions/settings.yaml', "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    rps = ReactiveComplexSampler(
        smiles_strings=["[O]=[N+]([O-])CCCl", "[F-]"],
        settings=settings,
        n_initial_complexes=1
    )

    structures = rps.sample_reaction_complexes(
        reactive_coordinate=[4,5],
        min=0.5,
        max=2.5,
        n_points=10
    )
    print(len(structures))
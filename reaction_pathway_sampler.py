from concurrent.futures import ProcessPoolExecutor
import autode as ade
from autode.values import Distance
from autode.species import Complex
from autode.wrappers.XTB import XTB
from autode.conformers.conformer import Conformer
from autode.utils import work_in_tmp_dir

from rdkit import Chem

import os
import time
from tqdm import tqdm
from typing import List, Literal, Any, Dict
import numpy as np
from openbabel import pybel
import openbabel as ob
from CREST import crest_driver

from XTB import xtb_driver
from utils import Molecule, conf_to_xyz_string, xyz_string_to_autode_atoms
from constants import bohr_ang
from xyz2mol import canonical_smiles_from_xyz_string, get_canonical_smiles_from_xyz_string_ob

def get_canonical_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def get_reactive_coordinate_value(
    mol: pybel.Molecule,
    reactive_coordinate: List[int]
) -> float:
    atoms = [mol.GetAtom(i) for i in reactive_coordinate]
    if len(atoms)==2:
        return atoms[0].GetDistance(atoms[1])
    if len(atoms)==3:
        return mol.GetAngle(*atoms)
    if len(atoms)==4:
        return mol.GetTorsion(*atoms)
    
def get_geometry_constraints(
    force: float,
    reactive_coordinate: List[int],
    curr_coordinate_val: float
):
    names = ['', '', 'distance', 'angle', 'dihedral']
    string = "$constrain\n"
    string += "  force constant=%f \n" % force
    if len(reactive_coordinate) == 2:
        string += "  distance: %i, %i, %f \n" % (reactive_coordinate[0], reactive_coordinate[1], curr_coordinate_val)
    return string

def get_scan_constraint(
    start: float,
    end: float,
    nsteps: int
):
    string = "$scan\n"
    string += '  1: %f, %f, %i \n' % (start, end, nsteps)
    return string

def get_wall_constraint(
    wall_radius: float
):
    string = "$wall\n"
    string += "  potential=logfermi"
    string += "  sphere:%f, all" % wall_radius
    return string

def get_metadynamics_constraint(
    mol: Molecule,
    settings: Dict[str, Any],
    mode: Literal["wide", "narrow"],
):
    string = "$md\n"
    for k, v in settings[f"md_settings_{mode}"].items():
        string += f"  {k}={v}\n"
    string += f"  time: {mol.n_atoms * settings[f'md_time_per_atom_{mode}']}\n"

    string += "$metadyn\n"
    for k, v in settings[f"metadyn_settings_{mode}"].items():
        string += f"  {k}={v}\n"
    return string    



def compute_wall_radius(
    complex: Molecule, 
    settings: Dict[str, Any],
) -> float:
    # Compute all interatomic distances
    distances = []
    for i in range(complex.n_atoms):
        for j in range(i):
            distances.append(
                np.sqrt(np.sum((complex.geometry[i].coordinates - complex.geometry[j].coordinates)**2))
            )

    # Cavity is 1.5 x maximum distance in diameter
    radius_bohr = 0.5 * max(distances) * settings["cavity_scale"] + 0.5 * settings["cavity_offset"]
    radius_bohr /= bohr_ang
    return radius_bohr

def compute_force_constant(
    complex: Molecule,
    settings: Dict[str, Any],
    reactive_coordinate: List[int],
    curr_coordinate_val: float,
):
    if len(reactive_coordinate) == 2:
        xcontrol_settings = get_geometry_constraints(
            settings['force_constant'],
            reactive_coordinate,
            curr_coordinate_val
        )
        xcontrol_settings += get_scan_constraint(
            curr_coordinate_val - 0.05, 
            curr_coordinate_val + 0.05, 
            5
        )

        structures, energies = xtb_driver(
            complex.to_xyz_string(),
            complex.charge,
            complex.mult,
            "scan",
            method='2',
            xcontrol_settings=xcontrol_settings,
            n_cores=2
        )

        mols = [pybel.readstring("xyz", s.lower()).OBMol for s in structures]
        x = [abs(get_reactive_coordinate_value(mol, reactive_coordinate)) for mol in mols]
        x = np.array(x)

        y = np.array(energies)
        p = np.polyfit(x, y, 2)
        k = 2*p[0]
        force_constant = float(k * bohr_ang)
    else:
        force_constant = 1.0
    return force_constant


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

class ReactionPathwaySampler:

    def __init__(
        self,
        smiles_strings: List[str],
        solvent: str,
        settings: Dict[str, Any],
        n_initial_complexes: int = 1,
    ) -> None:
        self.smiles_strings = smiles_strings
        self.solvent = solvent
        self.settings = settings
        self.n_initial_complexes = n_initial_complexes

        self.ade_complex = None

    def sample_reaction_complexes(
        self,
        complex: ade.Species,
        reactive_coordinate: List[int],
    ) -> List[str]:
        complex = Molecule.from_autode_mol(complex)
        equi_coordinate_value = get_reactive_coordinate_value(complex.to_pybel(), reactive_coordinate)

        self.settings["wall_radius"] = compute_wall_radius(
            complex=complex,
            settings=self.settings
        )
        self.settings["force_constant"] = compute_force_constant(
            complex=complex,
            settings=self.settings,
            reactive_coordinate=reactive_coordinate,
            curr_coordinate_val=equi_coordinate_value
        )

        # TODO: why can the metadynamics fail sometimes?

        t = time.time()
        # 1. sample conformers
        confs = self._sample_metadynamics_conformers(
            complex=complex,
            reactive_coordinate=reactive_coordinate,
            curr_coordinate_val=equi_coordinate_value, 
            mode="wide"
        )
        print(f'metadyn sampling: {time.time() - t}')
        print(f'metadyn sampled n conformers: {len(confs)}')

        # 2. optimize conformers
        t = time.time()
        confs = self._optimize_conformers(
            complex=complex,
            conformers=confs,
            reactive_coordinate=reactive_coordinate,
            curr_coordinate_val=equi_coordinate_value
        )
        print(f'optimizing conformers: {time.time() - t}')

        # TODO: we can not have changed the mol graph during sampling (but maybe RMSD selecting already filters this out)

        # 3. prune conformer set
        # if self.settings['use_pruning']:
        t = time.time()
        confs = self._prune_conformers(
            complex=complex,
            conformers=confs
        )
        print(f'pruning conformers: {time.time() - t}')
        print(f'conformers after pruning: {len(confs)}\n\n')

        return confs

    def sample_trajectories(
        self,
        reactive_coordinate: List[int],
        min: float,
        max: float,
        n_points: int
    ):
        pass
        # coordinate_interval = np.linspace(min, max, n_points)
        # trajectories = [
        #     self._relaxed_scan(conf, coordinate_interval) for conf in confs
        # ]   


        # # repeat for each intermediate interval
        # for curr_idx, curr_coord in enumerate(coordinate_interval):
        #     # 2. do constrained opt of structures at this coordinate
        #     constrained_confs = self._constrained_opt_confs(
        #         conformers=confs,
        #         reactive_coordinate=reactive_coordinate,
        #         curr_val=curr_coord
        #     )

        #     # 3. do metadynamics sampling of the structures at this coordinate
        #     new_constrained_confs = []
        #     for conf in constrained_confs:
        #         confs = self._sample_metadynamics_conformers(conf)
        #         confs = self._prune_conformers(confs)
        #         new_constrained_confs.append(confs)
        #     new_constrained_confs = self._prune_conformers(new_constrained_confs)

        #     # 4. create bck/fwd trajectories for each structure 
        #     trajectories += [
        #         self._simulate_trajectory(conf, coordinate_interval, curr_idx)
        #     ]
        
        # return trajectories

    def _get_ade_complex(self) -> Complex:
        if len(self.smiles_strings) == 1:
            self.ade_complex = ade.Molecule(smiles=self.smiles_strings[0])
        else:
            self.ade_complex = Complex(*[ade.Molecule(smiles=smi) for smi in self.smiles_strings])
        return self.ade_complex

    def _sample_initial_complexes(
        self
    ) -> List[Conformer]:
        """
        Sample initial complexes using autodE from the SMILES string
        """
        self.ade_complex = self._get_ade_complex()
        self.ade_complex._generate_conformers()
        self.ade_complex.conformers.prune_on_rmsd()

        arguments = [
            (
                conf_to_xyz_string(conformer), conformer.charge, conformer.mult,
                self.solvent, 
                self.settings['xtb_method'], "", self.settings['xtb_n_cores']
            ) for conformer in self.ade_complex.conformers
        ]
        with ProcessPoolExecutor(max_workers=self.settings['n_processes']) as executor:
            conformers = list(tqdm(executor.map(optimize_autode_conformer, arguments), total=len(arguments), desc="Optimizing init complex conformers"))
        
        self.ade_complex.conformers = list(filter(lambda x: x != None, conformers))
        self.ade_complex.conformers.prune_on_rmsd()

        print(f'sampled {len(self.ade_complex.conformers)} different complex confomers')

        def sort_complex_conformers_on_distance(
            conformers: List[Conformer],
            mols: List[ade.Molecule] 
        ) -> List[Conformer]:
            distances = []

            for conformer in conformers:
                if len(mols) == 2:
                    centroid_1 = np.mean(np.array([atom.coord for atom in conformer.atoms[:len(mols[0].atoms)]]), axis=0)
                    centroid_2 = np.mean(np.array([atom.coord for atom in conformer.atoms[len(mols[0].atoms):]]), axis=0)
                    distances.append(np.linalg.norm(centroid_2 - centroid_1))
                elif len(mols) == 3:
                    centroid_1 = np.mean(np.array([atom.coord for atom in conformer.atoms[:len(mols[0].atoms)]]), axis=0)
                    centroid_2 = np.mean(np.array([atom.coord for atom in conformer.atoms[len(mols[0].atoms):len(mols[0].atoms) + len(mols[1].atoms)]]), axis=0)
                    centroid_3 = np.mean(np.array([atom.coord for atom in conformer.atoms[len(mols[0].atoms) + len(mols[1].atoms):]]), axis=0)
                    distances.append(np.linalg.norm(centroid_2 - centroid_1) + np.linalg.norm(centroid_3 - centroid_1) + np.linalg.norm(centroid_3 - centroid_2))
            return [conformers[i] for i in np.argsort(np.array(distances))]

        complexes = sort_complex_conformers_on_distance(
            self.ade_complex.conformers,
            [ade.Molecule(smiles=smi) for smi in self.smiles_strings]
        )[:self.n_initial_complexes]

        complexes = self.ade_complex.conformers[:self.n_initial_complexes]
        return complexes
    
    def _sample_metadynamics_conformers(
        self,
        complex: Molecule,
        reactive_coordinate: List[int],
        curr_coordinate_val: float,
        mode: Literal["wide", "narrow"] = "wide"
    ) -> List[str]:
        """
        Sample conformers of a Molecule object complex.
        Returns a list of xyz strings containing conformers
        """
        # xcontrol_settings = get_geometry_constraints(
        #     self.settings['force_constant'],
        #     reactive_coordinate,
        #     curr_coordinate_val
        # )
        xcontrol_settings = get_wall_constraint(
            self.settings["wall_radius"]
        )
        xcontrol_settings += get_metadynamics_constraint(
            complex,
            self.settings,
            mode
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
        reactive_coordinate: List[int],
        curr_coordinate_val: float
    ) -> List[str]:
        """
        Optimizes a set of conformers
        """
        # xcontrol_settings = get_geometry_constraints(
        #     self.settings['force_constant'],
        #     reactive_coordinate,
        #     curr_coordinate_val
        # )
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
    ) -> List[str]:
        """
        Prunes a set of conformers using CREST CREGEN
        """
        # structures = crest_driver(
        #     ref_structure=complex.to_xyz_string(),
        #     ensemble_structures='\n'.join(conformers),
        #     ref_energy_threshold=self.settings["ref_energy_threshold"],
        #     rmsd_threshold=self.settings["rmsd_threshold"],
        #     conf_energy_threshold=self.settings["conf_energy_threshold"],
        #     rotational_threshold=self.settings["rotational_threshold"],
        # )
        # return structures
        
        pruned_conformers = []
        smiles_list = [get_canonical_smiles(smi) for smi in self.smiles_strings]
        # TODO: replace charges in a more structured way?
        smiles_list = [smi if len(smi) > 6 else smi.replace('+', '').replace('-', '')  for smi in smiles_list]
        for conformer in conformers:
            conf_smiles_list = get_canonical_smiles_from_xyz_string_ob(conformer)
            if set(conf_smiles_list) == set(smiles_list):
                pruned_conformers.append(conformer)
        
        return pruned_conformers



    def _relaxed_scan(
        self,
        structure: Any,
        coordinate_interval,  
    ):
        pass


if __name__ == "__main__":
    import yaml
    import autode as ade

    ade.Config.n_cores = 4
    ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'
    ade.Config.rmsd_threshold = Distance(0.5, units="Å")

    with open('/home/ruard/code/test_reactions/settings.yaml', "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    rps = ReactionPathwaySampler(
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
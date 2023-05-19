
from concurrent.futures import ProcessPoolExecutor
from typing import Any, List
import autode as ade
import os
from tqdm import tqdm
import numpy as np
from openbabel import openbabel as ob
import time
from src.conformational_sampling import ConformerSampler
from src.interfaces.XTB import xtb_driver

from src.interfaces.lewis import compute_adjacency_matrix, mol_write
from src.interfaces.xtb_utils import compute_wall_radius, get_wall_constraint
from src.molecule import Molecule
from src.utils import geom_to_xyz_string, xyz_string_to_geom

def optimize_conformer(args):
    conformer, charge, mult, solvent, method, xcontrol_settings, cores = args
    return xtb_driver(
        conformer,
        charge,
        mult,
        "opt",
        method=method,
        solvent=solvent,
        xcontrol_settings=xcontrol_settings,
        n_cores=cores
    )

def optimize_conformer_ff(args):
    atomic_symbols, coordinates, adj_mat = args
    new_coords = compute_ff_optimized_coords(
        atomic_symbols, coordinates, adj_mat
    )
    return geom_to_xyz_string(atomic_symbols, new_coords)

def compute_ff_optimized_coords(
    atomic_symbols: List[str],
    coordinates: Any,
    adj_mat: np.array,
    ff_name: str = 'UFF', 
    fixed_atoms: List[int] = [], 
    n_steps: int = 500
) -> None:
    # create a mol file object for ob
    mol_file_name = 'obff.mol'

    mol_write(
        name=mol_file_name,
        elements=atomic_symbols,
        geo=coordinates,
        adj_mat=adj_mat,
        q=0,
        append_opt=False
    )

    # load in ob mol
    conv = ob.OBConversion()
    conv.SetInAndOutFormats('mol','xyz')
    mol = ob.OBMol()    
    conv.ReadFile(mol, mol_file_name)

    # Define constraints  
    constraints= ob.OBFFConstraints()
    if len(fixed_atoms) > 0:
        for atom in fixed_atoms: 
            constraints.AddAtomConstraint(int(atom)) 
            
    # Setup the force field with the constraints 
    forcefield = ob.OBForceField.FindForceField(ff_name)
    forcefield.Setup(mol, constraints)     
    forcefield.SetConstraints(constraints) 
    
    # Do a conjugate gradient minimiazation
    forcefield.ConjugateGradients(n_steps)
    forcefield.GetCoordinates(mol) 

    # cleanup
    try:
        os.remove(mol_file_name)
    except:
        pass

    # read coordinates
    coordinates = []
    for atom in ob.OBMolAtomIter(mol):
        coordinates.append([atom.GetX(), atom.GetY(), atom.GetZ()])

    return np.array(coordinates)

class TopologyConformerSampler(ConformerSampler):

    def __init__(
        self,
        smiles_strings: List[str],
        settings: Any,
        solvent: str,
        mol: ade.Species,
    ) -> None:
        super().__init__(
            smiles_strings=smiles_strings,
            settings=settings,
            solvent=solvent
        )
        self.mol = mol
        self.adjacency_matrix = compute_adjacency_matrix(
            elements=[a.atomic_symbol for a in self.mol.atoms],
            geometry=self.mol.coordinates
        )

        self.settings["wall_radius"] = compute_wall_radius(
            complex=Molecule.from_autode_mol(self.mol),
            settings=self.settings
        )

    def sample_conformers(self, conformers: List[str]) -> List[str]:
        # 1. Find similar conformers
        t = time.time()
        confs = self._compute_ff_optimised_conformers(
            conformers=conformers
        )
        print(f'optimizing conformers with FF: {time.time() - t}')

        # 2. Optimize conformers
        t = time.time()
        confs = self._optimize_conformers(
            conformers=confs,
        )
        print(f'optimizing conformers: {time.time() - t}')

        # 3. prune conformer set
        t = time.time()
        confs = self._prune_conformers(
            initial_geometry=self.mol,
            conformers=confs,
            use_graph_pruning=True,
            use_cregen_pruning=False
        )
        print(f'pruning conformers: {time.time() - t}')
        print(f'conformers after pruning: {len(confs)}\n\n')

        return confs
    
    def _compute_ff_optimised_conformers(
        self,
        conformers: List[str]
    ):  
        arguments = []
        for conformer in conformers:
            atomic_symbols, coordinates = xyz_string_to_geom(conformer)
            arguments.append((atomic_symbols, coordinates, self.adjacency_matrix))
 
        with ProcessPoolExecutor(max_workers=self.settings['n_processes']) as executor:
            pre_opt_conformers = list(tqdm(executor.map(optimize_conformer_ff, arguments), total=len(arguments), desc="Optimizing conformers FF"))

        pre_opt_conformers = list(filter(lambda x: x is not None, pre_opt_conformers))
        return pre_opt_conformers

    def _optimize_conformers(
        self,
        conformers: List[str],
    ) -> List[str]:
        """
        Optimizes a set of conformers
        """
        xcontrol_settings = get_wall_constraint(
            self.settings["wall_radius"]
        )
        
        arguments = [
            (conf, self.mol.charge, self.mol.mult, self.solvent, self.settings['xtb_method'], xcontrol_settings, self.settings['xtb_n_cores']) for conf in conformers
        ]
        with ProcessPoolExecutor(max_workers=self.settings['n_processes']) as executor:
            opt_conformers = list(tqdm(executor.map(optimize_conformer, arguments), total=len(arguments), desc="Optimizing conformers"))

        opt_conformers = list(filter(lambda x: x is not None, opt_conformers))
        return opt_conformers

from concurrent.futures import ProcessPoolExecutor
from typing import Any, List
import autode as ade
import os
from tqdm import tqdm
import numpy as np
from openbabel import openbabel as ob
import time
from src.interfaces.XTB import xtb_driver

from src.interfaces.lewis import compute_adjacency_matrix, mol_write
from src.interfaces.xtb_utils import get_wall_constraint
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

class TopologyConformerSampler:

    def __init__(
        self,
        mol: ade.Species,
        solvent: str,
        settings: Any
    ) -> None:
        self.mol = mol
        self.adjacency_matrix = compute_adjacency_matrix(
            elements=[a.atomic_symbol for a in self.mol.atoms],
            geometry=self.mol.coordinates
        )

        self.solvent = solvent
        self.settings = settings

    def sample_conformers(self, conformers: List[str]) -> List[str]:
        pre_opt_conformers = []
        for conformer in conformers:
            atomic_symbols, coordinates = xyz_string_to_geom(conformer)
            new_coords = compute_ff_optimized_coords(
                atomic_symbols,
                coordinates,
                self.adjacency_matrix
            )
            conformer = geom_to_xyz_string(atomic_symbols, new_coords)
            pre_opt_conformers.append(conformer)

        t = time.time()
        confs = self._optimize_conformers(
            conformers=pre_opt_conformers,
        )
        print(f'optimizing conformers: {time.time() - t}')

        return confs
    
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
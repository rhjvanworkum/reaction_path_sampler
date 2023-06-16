
from concurrent.futures import ProcessPoolExecutor
import logging
import tempfile
from typing import Any, List
import autode as ade
import os
from tqdm import tqdm
import numpy as np
from openbabel import openbabel as ob
import time
from reaction_path_sampler.src.conformational_sampling import ConformerSampler
from reaction_path_sampler.src.interfaces.XTB import xtb_driver

from reaction_path_sampler.src.graphs.lewis import compute_adjacency_matrix, mol_write, mol_write_geom
from reaction_path_sampler.src.interfaces.xtb_utils import compute_wall_radius, get_wall_constraint
from reaction_path_sampler.src.molecular_system import MolecularSystem
from reaction_path_sampler.src.utils import geom_to_xyz_string, get_tqdm_disable, remove_whitespaces_from_xyz_strings, xyz_string_to_geom
from reaction_path_sampler.src.visualization.plotly import plot_networkx_mol_graph

from reaction_path_sampler.src.graphs.xyz2mol import xyz2AC, __ATOM_LIST__

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

def optimize_conformer_ff(args):
    atomic_symbols, coordinates, adj_mat, bond_mat = args
    new_coords = compute_ff_optimized_coords(
        atomic_symbols, coordinates, adj_mat, bond_mat
    )
    return geom_to_xyz_string(atomic_symbols, new_coords)

def compute_ff_optimized_coords(
    atomic_symbols: List[str],
    coordinates: Any,
    adj_mat: np.ndarray,
    bond_mat: np.ndarray,
    ff_name: str = 'UFF', 
    fixed_atoms: List[int] = [], 
    n_steps: int = 500
) -> None:
    with tempfile.NamedTemporaryFile(suffix='.mol') as tmp:
        # write mol file
        mol_write_geom(
            name=tmp.name,
            elements=atomic_symbols,
            geo=coordinates,
            adj_mat=adj_mat,
            bond_mat=bond_mat,
            append_opt=False
        )

        # load in ob mol
        conv = ob.OBConversion()
        conv.SetInAndOutFormats('mol','xyz')
        mol = ob.OBMol()
        conv.ReadFile(mol, tmp.name)

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
        mol: MolecularSystem
    ) -> None:
        super().__init__(
            smiles_strings=smiles_strings,
            settings=settings,
            solvent=solvent
        )
        self.mol = mol

        self.settings["wall_radius"] = compute_wall_radius(
            mol=self.mol,
            settings=self.settings
        )

    def sample_conformers(self, conformers: List[str]) -> List[str]:
        # 1. Find similar conformers
        t = time.time()
        confs = self._compute_ff_optimised_conformers(
            conformers=conformers
        )
        print(f'optimizing conformers with FF: {time.time() - t}')

        with open('ff_optimized_confs.', 'w') as f:
            f.writelines(confs)

        # 2. Optimize conformers
        t = time.time()
        confs = self._optimize_conformers(
            conformers=confs,
        )
        print(f'optimizing conformers: {time.time() - t}')

        with open('xtb_optimized_confs.', 'w') as f:
            f.writelines(confs)

        # 3. prune conformer set
        t = time.time()
        confs = self._prune_conformers(
            mol=self.mol,
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
            arguments.append((atomic_symbols, coordinates, self.mol.connectivity_matrix, self.mol.bond_order_matrix))
 
        with ProcessPoolExecutor(max_workers=self.settings['n_processes']) as executor:
            pre_opt_conformers = list(tqdm(executor.map(optimize_conformer_ff, arguments), total=len(arguments), desc="Optimizing conformers FF", disable=get_tqdm_disable()))

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
            (conf, self.mol, self.solvent, self.settings['xtb_method'], xcontrol_settings, self.settings['xtb_n_cores']) for conf in conformers
        ]
        with ProcessPoolExecutor(max_workers=self.settings['n_processes']) as executor:
            opt_conformers = list(tqdm(executor.map(optimize_conformer, arguments), total=len(arguments), desc="Optimizing conformers", disable=get_tqdm_disable()))

        opt_conformers = list(filter(lambda x: x is not None, opt_conformers))
        return opt_conformers
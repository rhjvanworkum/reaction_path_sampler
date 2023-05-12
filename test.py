# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import networkx as nx
# import yaml

# from src.reaction_path.complexes import generate_reactant_product_complexes
# from src.reaction_path.reaction_graph import get_reaction_isomorphisms
# from src.ts_template import TStemplate

# from autode.bond_rearrangement import get_bond_rearrangs
# from autode.mol_graphs import (
#     get_mapping_ts_template,
#     get_truncated_active_mol_graph,
# )
# import autode as ade
# from autode.species import Complex
# from autode.mol_graphs import reac_graph_to_prod_graph


# if __name__ == "__main__":
#     img_name = 'test2.png'

#     """ RC's / PC's """
#     with open('./systems/ac_base.yaml', "r") as f:
#         settings = yaml.load(f, Loader=yaml.Loader)
#     output_dir = settings["output_dir"]
#     reactant_smiles = settings["reactant_smiles"]
#     product_smiles = settings["product_smiles"]
#     solvent = settings["solvent"]
    
#     rc_complex, _rc_conformers, rc_n_species, rc_species_complex_mapping = generate_reactant_product_complexes(
#         reactant_smiles, 
#         solvent, 
#         settings, 
#         f'{output_dir}/rcs.xyz'
#     )
#     pc_complex, _pc_conformers, pc_n_species, pc_species_complex_mapping = generate_reactant_product_complexes(
#         product_smiles, 
#         solvent, 
#         settings, 
#         f'{output_dir}/pcs.xyz'
#     )   
    
#     # bond_rearr, reaction_isomorphisms, isomorphism_idx = get_reaction_isomorphisms(rc_complex, pc_complex)
#     # # graph = reac_graph_to_prod_graph(pc_complex.graph, bond_rearr)
#     # graph = rc_complex.graph

#     for idx, atom in enumerate(rc_complex.atoms):
#         print(idx, atom.atomic_symbol)




from typing import List, Optional
import os
import yaml
import numpy as np
import autode as ade
from autode.input_output import atoms_to_xyz_file
from openbabel import openbabel as ob
from lewis import Table_generator, mol_write
from geodesic_interpolate.fileio import write_xyz
from src.reaction_path.complexes import compute_optimal_coordinates, generate_reaction_complexes
from src.reaction_path.path_interpolation import interpolate_geodesic
from src.reaction_path.reaction_graph import map_reaction_complexes

from src.reactive_complex_sampler import ReactiveComplexSampler
from src.utils import set_autode_settings

def compute_ff_optimized_coords(
    conformer: ade.Species,
    adj_mat: Optional[np.array] = None,
    ff_name: str = 'UFF', 
    fixed_atoms: List[int] = [], 
    n_steps: int = 500
) -> None:
    # create a mol file object for ob
    mol_file_name = 'obff.mol'

    if adj_mat is None:
        adj_mat = Table_generator(
            Elements=[a.atomic_symbol for a in conformer.atoms],
            Geometry=conformer.coordinates
        )

    mol_write(
        name=mol_file_name,
        elements=[a.atomic_symbol for a in conformer.atoms],
        geo=conformer.coordinates,
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


if __name__ == "__main__":
    # 1. get product conformer
    with open('./systems/ac_base.yaml', "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    # set autode settings
    set_autode_settings(settings)

    output_dir = settings["output_dir"]
    reactant_smiles = settings["reactant_smiles"]
    product_smiles = settings["product_smiles"]
    solvent = settings["solvent"]

    complex, _, pc_species_complex_mapping = generate_reaction_complexes(
        smiles_strings=product_smiles,
        solvent=solvent,
        settings=settings,
        save_path=None
    )
    complex._generate_conformers()
    pc = complex.conformers[0]

    complex, _, rc_species_complex_mapping = generate_reaction_complexes(
        smiles_strings=reactant_smiles,
        solvent=solvent,
        settings=settings,
        save_path=None
    )
    complex._generate_conformers()
    rc = complex.conformers[0]

    rcs, pcs = map_reaction_complexes(
        _rc_conformers=[rc], 
        _pc_conformers=[pc], 
        settings=settings,
        rc_species_complex_mapping=rc_species_complex_mapping,
        pc_species_complex_mapping=pc_species_complex_mapping
    )
    rc, pc = rcs[0], pcs[0]


    # 2. optimize product conformer
    new_coords = compute_ff_optimized_coords(pc)
    pc._coordinates = new_coords

    # 3. Find corresponding reactant conformer
    adj_matrix = Table_generator(
        Elements=[a.atomic_symbol for a in rc.atoms],
        Geometry=rc.coordinates
    )
    new_coords = compute_ff_optimized_coords(rc, adj_mat=adj_matrix)
    rc._coordinates = new_coords
    rc._coordinates = compute_optimal_coordinates(rc.coordinates, pc.coordinates)

    # 4. export xyz
    atoms_to_xyz_file(rc.atoms, f'test_rc.xyz')
    atoms_to_xyz_file(pc.atoms, f'test_pc.xyz')

    # t = time.time()
    curve = interpolate_geodesic(
        rc.atomic_symbols, 
        rc.coordinates, 
        pc.coordinates,
        settings
    )

    write_xyz(f'geodesic_path.xyz', rc.atomic_symbols, curve.path)
    # print(f'geodesic interpolation: {time.time() - t}')
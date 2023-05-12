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




from typing import List
import os
import numpy as np

def xyz_parse(input,read_types=False):

    # Commands for reading only the coordinates and the elements
    if read_types is False:
        
        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count]=fields[0]
                            Geometry[count,:]=np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry

    # Commands for reading the atomtypes from the fourth column
    if read_types is True:

        # Iterate over the remainder of contents and read the
        # geometry and elements into variable. Note that the
        # first two lines are considered a header
        with open(input,'r') as f:
            for lc,lines in enumerate(f):
                fields=lines.split()

                # Parse header
                if lc == 0:
                    if len(fields) < 1:
                        print("ERROR in xyz_parse: {} is missing atom number information".format(input))
                        quit()
                    else:
                        N_atoms = int(fields[0])
                        Elements = ["X"]*N_atoms
                        Geometry = np.zeros([N_atoms,3])
                        Atom_types = [None]*N_atoms
                        count = 0

                # Parse body
                if lc > 1:

                    # Skip empty lines
                    if len(fields) == 0:
                        continue            

                    # Write geometry containing lines to variable
                    if len(fields) > 3:

                        # Consistency check
                        if count == N_atoms:
                            print("ERROR in xyz_parse: {} has more coordinates than indicated by the header.".format(input))
                            quit()

                        # Parse commands
                        else:
                            Elements[count] = fields[0]
                            Geometry[count,:] = np.array([float(fields[1]),float(fields[2]),float(fields[3])])
                            if len(fields) > 4:
                                Atom_types[count] = fields[4]
                            count = count + 1

        # Consistency check
        if count != len(Elements):
            print("ERROR in xyz_parse: {} has less coordinates than indicated by the header.".format(input))

        return Elements,Geometry,Atom_types

def ob_geo_opt(
    E,
    G,
    adj_mat,
    ff_name: str = 'UFF', 
    fixed_atoms: List[int] = [], 
    n_steps: int = 500
) -> None:
    import openbabel as ob

    # load in openbabel modules
    conv = ob.OBConversion()
    conv.SetInAndOutFormats('mol','xyz')
    mol = ob.OBMol()

    # create a mol file for geo_opt
    opt_file = 'obff.mol'
    # mol_write(opt_file,E,G,adj_mat,q=q,append_opt=False)
    
    conv.ReadFile(mol,opt_file)
    
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
    
    # write into xyz file
    conv.WriteFile(mol,'result.xyz')
    
    # parse output xyz file
    element, geo = xyz_parse("result.xyz")

    # remove mol and xyz file
    try:
        os.remove(opt_file)
        os.remove('result.xyz')
    except:
        pass

    return geo



if __name__ == "__main__":
    # 1. get product conformer

    # so you wanna take the product elements & geometry but use the adjacency_matrix of the reactants

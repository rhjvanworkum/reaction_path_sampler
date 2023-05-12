import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import yaml

from src.reaction_path.complexes import generate_reactant_product_complexes
from src.reaction_path.reaction_graph import get_reaction_isomorphisms
from src.ts_template import TStemplate

from autode.bond_rearrangement import get_bond_rearrangs
from autode.mol_graphs import (
    get_mapping_ts_template,
    get_truncated_active_mol_graph,
)
import autode as ade
from autode.species import Complex
from autode.mol_graphs import reac_graph_to_prod_graph


if __name__ == "__main__":
    img_name = 'test2.png'

    """ RC's / PC's """
    with open('./systems/ac_base.yaml', "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)
    output_dir = settings["output_dir"]
    reactant_smiles = settings["reactant_smiles"]
    product_smiles = settings["product_smiles"]
    solvent = settings["solvent"]
    
    rc_complex, _rc_conformers, rc_n_species, rc_species_complex_mapping = generate_reactant_product_complexes(
        reactant_smiles, 
        solvent, 
        settings, 
        f'{output_dir}/rcs.xyz'
    )
    pc_complex, _pc_conformers, pc_n_species, pc_species_complex_mapping = generate_reactant_product_complexes(
        product_smiles, 
        solvent, 
        settings, 
        f'{output_dir}/pcs.xyz'
    )   
    
    # bond_rearr, reaction_isomorphisms, isomorphism_idx = get_reaction_isomorphisms(rc_complex, pc_complex)
    # # graph = reac_graph_to_prod_graph(pc_complex.graph, bond_rearr)
    # graph = rc_complex.graph

    for idx, atom in enumerate(rc_complex.atoms):
        print(idx, atom.atomic_symbol)
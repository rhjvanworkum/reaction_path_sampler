import os
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

from reaction_path_sampler.src.ts_template import TStemplate

from autode.bond_rearrangement import get_bond_rearrangs
from autode.mol_graphs import (
    get_mapping_ts_template,
    get_truncated_active_mol_graph,
)
from autode.transition_states.templates import template_matches
import autode as ade
from autode.species import Complex

if __name__ == "__main__":
    """ Template """
    template = TStemplate(filename="./scratch/templates/da_cores_new/template94.txt")
    # template_graph = template.graph

    """ Reaction """
    df = pd.read_csv('./data/test_da_reactions_new.csv')
    reaction_smiles = df['reaction_smiles'].values[335]
    print(reaction_smiles)
    reactants, products = reaction_smiles.split('>>')
    reactant_smiles = reactants.split('.')
    product_smiles = products.split('.')

    if len(reactant_smiles) == 1:
        reactants = ade.Molecule(smiles=reactant_smiles[0])
    else:
        reactants = Complex(*[ade.Molecule(smiles=smi) for smi in reactant_smiles])
    if len(product_smiles) == 1:
        products = ade.Molecule(smiles=product_smiles[0])
    else:
        products = Complex(*[ade.Molecule(smiles=smi) for smi in product_smiles])

    bond_rearr = get_bond_rearrangs(products, reactants, name='test')[0]
    reaction_graph = get_truncated_active_mol_graph(graph=products.graph, active_bonds=bond_rearr.all)
    # # products.populate_conformers()
    # nx.set_node_attributes(truncated_graph, {node: products.coordinates[node] for idx, node in enumerate(truncated_graph.nodes)}, 'cartesian')
    # reaction_graph = truncated_graph

    match, ignore_active_bonds = template_matches(products, reaction_graph, template)
    print(match, ignore_active_bonds)
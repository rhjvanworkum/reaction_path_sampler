import os
import pandas as pd

import matplotlib.pyplot as plt
import networkx as nx

from src.ts_template import TStemplate

from autode.bond_rearrangement import get_bond_rearrangs
from autode.mol_graphs import (
    get_mapping_ts_template,
    get_truncated_active_mol_graph,
)
import autode as ade
from autode.species import Complex


if __name__ == "__main__":
    img_name = 'test.png'

    """ Template """
    template = TStemplate(filename="./scratch/templates/da_cores_new/template94.txt")
    graph = template.graph

    """ Reaction """
    # df = pd.read_csv('./data/test_da_reactions_new.csv')
    # reaction_smiles = df['reaction_smiles'].values[335]
    # reactants, products = reaction_smiles.split('>>')
    # reactant_smiles = reactants.split('.')
    # product_smiles = products.split('.')

    # if len(reactant_smiles) == 1:
    #     reactants = ade.Molecule(smiles=reactant_smiles[0])
    # else:
    #     reactants = Complex(*[ade.Molecule(smiles=smi) for smi in reactant_smiles])
    # if len(product_smiles) == 1:
    #     products = ade.Molecule(smiles=product_smiles[0])
    # else:
    #     products = Complex(*[ade.Molecule(smiles=smi) for smi in product_smiles])

    # bond_rearr = get_bond_rearrangs(products, reactants, name='test')[0]
    # truncated_graph = get_truncated_active_mol_graph(graph=products.graph, active_bonds=bond_rearr.all)
    # # products.populate_conformers()
    # nx.set_node_attributes(truncated_graph, {node: products.coordinates[node] for idx, node in enumerate(truncated_graph.nodes)}, 'cartesian')
    # graph = truncated_graph



    color_df = pd.read_csv('./data/jmol_colors.csv')
    plt.rcParams.update({
        "figure.facecolor": "black",
        "figure.edgecolor": "black",
        "savefig.facecolor": "black",
        "savefig.edgecolor": "black"
    })

    atoms_set = set([data["atom_label"] for _, data in graph.nodes(data=True)])
    atoms_dict = {k: [] for k in atoms_set}
    pos_dict = {}

    for idx, data in graph.nodes(data=True):
        atoms_dict[data['atom_label']].append(idx)
        pos_dict[idx] = data['cartesian'][:2]

    for key, values in atoms_dict.items():
        nx.draw_networkx_nodes(
            graph,
            # nx.spectral_layout(graph),
            pos_dict,
            nodelist=values,
            node_color=(
                float(color_df[color_df['atom'] == key]['R'].values[0]) / 255,
                float(color_df[color_df['atom'] == key]['G'].values[0]) / 255,
                float(color_df[color_df['atom'] == key]['B'].values[0]) / 255
            )
        )


    bonds_dict = {
        'active': [],
        'single': [],
        'double': []
    }
    style_dict = {
        'active': 'dashed',
        'double': 'solid',
        'single': 'solid'
    }

    for idx, data in enumerate(graph.edges(data=True)):
        if data[2]['active']:
            bonds_dict['active'].append((data[0], data[1]))
        elif data[2]['pi']:
            bonds_dict['double'].append((data[0], data[1]))
        else:
            bonds_dict['single'].append((data[0], data[1]))

    for key, values in bonds_dict.items():
        nx.draw_networkx_edges(
            graph,
            # nx.spectral_layout(graph),
            pos_dict,
            edgelist=values,
            edge_color=(1,1,1),
            style=style_dict[key]
        )


    plt.axis('off')
    plt.savefig(img_name)
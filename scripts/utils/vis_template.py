from reaction_path_sampler.src.ts_template import TStemplate
from reaction_path_sampler.src.visualization.plotly import plot_networkx_mol_graph


if __name__ == "__main__":
    """ RC's / PC's """
    # with open('./systems/ac_base.yaml', "r") as f:
    #     settings = yaml.load(f, Loader=yaml.Loader)
    # output_dir = settings["output_dir"]
    # reactant_smiles = settings["reactant_smiles"]
    # product_smiles = settings["product_smiles"]
    # solvent = settings["solvent"]
    
    # rc_complex, _rc_conformers, rc_n_species, rc_species_complex_mapping = generate_reactant_product_complexes(
    #     reactant_smiles, 
    #     solvent, 
    #     settings, 
    #     f'{output_dir}/rcs.xyz'
    # )
    # pc_complex, _pc_conformers, pc_n_species, pc_species_complex_mapping = generate_reactant_product_complexes(
    #     product_smiles, 
    #     solvent, 
    #     settings, 
    #     f'{output_dir}/pcs.xyz'
    # )   
    
    # bond_rearr, reaction_isomorphisms, isomorphism_idx = get_reaction_isomorphisms(rc_complex, pc_complex)
    # # graph = reac_graph_to_prod_graph(pc_complex.graph, bond_rearr)
    # graph = rc_complex.graph

    """ Template """
    template = TStemplate(filename="./scratch/ac2/template0.txt")
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

    plot_networkx_mol_graph(graph)
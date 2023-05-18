# import numpy as np
# import networkx as nx
# import autode as ade
# import shutil

# from autode.mol_graphs import (
#     get_truncated_active_mol_graph,
# )
# from src.reaction_path.reaction_ends import check_reaction_ends

# from src.ts_template import TStemplate
# from src.xyz2mol import read_xyz_string, get_canonical_smiles_from_xyz_string
# from src.utils import get_canonical_smiles

# def save_ts_template(
#     tsopt: str,
#     rc_complex: ade.Species,
#     pc_complex: ade.Species,
#     isomorphism_idx: int,
#     bond_rearr: ade.bond_rearr,
#     output_dir: str
# ):
#     # save as a template here
#     base_complex = [rc_complex, pc_complex][1 - isomorphism_idx].copy()
#     coords = np.array([
#         [a.x, a.y, a.z] for a in read_xyz_string(tsopt)
#     ])
#     base_complex.coordinates = coords

#     for bond in bond_rearr.all:
#         base_complex.graph.add_active_edge(*bond)
#     truncated_graph = get_truncated_active_mol_graph(graph=base_complex.graph, active_bonds=bond_rearr.all)
#     # bonds
#     for bond in bond_rearr.all:
#         truncated_graph.edges[bond]["distance"] = base_complex.distance(*bond)
#     # cartesians
#     nx.set_node_attributes(truncated_graph, {node: base_complex.coordinates[node] for idx, node in enumerate(truncated_graph.nodes)}, 'cartesian')

#     ts_template = TStemplate(truncated_graph, species=base_complex)
#     ts_template.save(folder_path=f'{output_dir}/')


# def finish_reaction(
#     rc_complex: ade.Species,
#     pc_complex: ade.Species, 
#     isomorphism_idx: int,
#     bond_rearr: ade.bond_rearr,
#     output_dir: str
# ):
#     true_rc_smi_list = [get_canonical_smiles(smi) for smi in reactant_smiles]
#     true_pc_smi_list = [get_canonical_smiles(smi) for smi in product_smiles]
#     pred_rc_smi_list = get_canonical_smiles_from_xyz_string("".join(backward_end), pc_complex.charge)
#     pred_pc_smi_list = get_canonical_smiles_from_xyz_string("".join(forward_end), pc_complex.charge)

#     print(true_rc_smi_list, pred_rc_smi_list)
#     print(true_pc_smi_list, pred_pc_smi_list)
#     print('\n\n')

#     if check_reaction_ends(
#         true_rc_smi_list,
#         true_pc_smi_list,
#         pred_rc_smi_list,
#         pred_pc_smi_list,
#     ):
#         save_ts_template(
#             tsopt, rc_complex, pc_complex, isomorphism_idx, bond_rearr, output_dir
#         )

#         # also save tsopt + reaction + irc path
#         shutil.copy2(f'{output_dir}/{idx}/ts_opt.xyz', f'{output_dir}/ts_opt.xyz')
#         shutil.copy2(f'{output_dir}/{idx}/reaction.xyz', f'{output_dir}/reaction.xyz')
#         shutil.copy2(f'{output_dir}/{idx}/irc_path.xyz', f'{output_dir}/irc_path.xyz')
        
#         print('finshed reaction \n\n')
#         break
import numpy as np
from typing import Callable, Dict, List, Any
import os
import time
import yaml
import networkx as nx
import argparse
import shutil

from geodesic_interpolate.fileio import write_xyz

from autode.input_output import atoms_to_xyz_file
from autode.mol_graphs import (
    get_truncated_active_mol_graph,
)

from src.conformational_sampling.sample_conformers import sample_reactant_and_product_conformers
from src.reaction_path.complexes import compute_optimal_coordinates, generate_reaction_complex, select_promising_reactant_product_pairs
from src.reaction_path.path_interpolation import interpolate_geodesic
from src.reaction_path.reaction_ends import check_reaction_ends
from src.reaction_path.reaction_graph import get_reaction_graph_isomorphism
from src.visualization.plotly import plot_networkx_mol_graph
from src.ts_template import TStemplate
from src.interfaces.PYSISYPHUS import pysisyphus_driver
from src.molecule import read_xyz_string
from src.utils import get_canonical_smiles, read_trajectory_file, remap_conformer, remove_whitespaces_from_xyz_strings, set_autode_settings, xyz_string_to_autode_atoms
from src.xyz2mol import get_canonical_smiles_from_xyz_string, get_canonical_smiles_from_xyz_string_ob


def write_output_file(variable, name):
    if variable is not None:
        with open(name, 'w') as f:
            f.writelines(variable)


def search_reaction_path(settings: Dict[str, Any]) -> None:
    output_dir = settings["output_dir"]
    reactant_smiles = settings["reactant_smiles"]
    product_smiles = settings["product_smiles"]
    solvent = settings["solvent"]

    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set autode settings
    set_autode_settings(settings)

    # 1. Get reaction complexes 
    rc_complex = generate_reaction_complex(reactant_smiles)
    pc_complex = generate_reaction_complex(product_smiles)

    # 2. Remap the reaction
    bond_rearr, isomorphism, isomorphism_idx = get_reaction_graph_isomorphism(rc_complex, pc_complex, settings)
    if isomorphism_idx == 0:
        rc_complex.conformers = [remap_conformer(conf, isomorphism) for conf in rc_complex.conformers]
    elif isomorphism_idx == 1:
        pc_complex.conformers = [remap_conformer(conf, isomorphism) for conf in pc_complex.conformers]
    
    # 3. Sample conformers
    rc_conformers, pc_conformers = sample_reactant_and_product_conformers(
        rc_complex,
        pc_complex,
        settings
    )
    
    # select closest pairs of reactants & products
    t = time.time()
    print('Selecting most promising Reactant-Product complexes now...')
    closest_pairs = select_promising_reactant_product_pairs(
        rc_conformers=rc_conformers,
        pc_conformers=pc_conformers,
        species_complex_mapping=None,       # currently unused
        bonds=None,                         # currently unused
        settings=settings
    )
    print(f'Selecting most promising Reactant-Product Complex pairs took: {time.time() - t}\n\n')
    print(closest_pairs)

    for idx, opt_idx in enumerate(closest_pairs):
        if not os.path.exists(f'{output_dir}/{idx}'):
            os.makedirs(f'{output_dir}/{idx}/')

        print(f'Working on Reactant-Product Complex pair {idx}')
        rc_conformer = rc_conformers[opt_idx[0]]
        pc_conformer = pc_conformers[opt_idx[1]] 

        # 1. Optimally align the 2 conformers using kabsh algorithm
        t = time.time()
        rc_conformer._coordinates = compute_optimal_coordinates(rc_conformer.coordinates, pc_conformer.coordinates)
        atoms_to_xyz_file(rc_conformer.atoms, f'{output_dir}/{idx}/selected_rc.xyz')
        atoms_to_xyz_file(pc_conformer.atoms, f'{output_dir}/{idx}/selected_pc.xyz')
        print(f'aligning complexes: {time.time() - t}')

        # 2. Create a geodesic interpolation between 2 optimal conformers
        t = time.time()
        curve = interpolate_geodesic(
            rc_conformer.atomic_symbols, 
            rc_conformer.coordinates, 
            pc_conformer.coordinates,
            settings
        )
        write_xyz(f'{output_dir}/{idx}/geodesic_path.trj', rc_conformer.atomic_symbols, curve.path)
        write_xyz(f'{output_dir}/{idx}/geodesic_path.xyz', rc_conformer.atomic_symbols, curve.path)
        print(f'geodesic interpolation: {time.time() - t}')

        # 3. Perform NEB-CI + TS opt in pysisyphus
        t = time.time()
        output, cos_final_traj, tsopt, imaginary_freq = pysisyphus_driver(
            geometry_files=[f'{output_dir}/{idx}/geodesic_path.trj'],
            charge=pc_complex.charge,
            mult=pc_complex.mult,
            job="ts_search",
            solvent=solvent
        )
        if os.path.exists(f'{output_dir}/{idx}/geodesic_path.trj'):
            os.remove(f'{output_dir}/{idx}/geodesic_path.trj')
        print(f'TS search time: {time.time() - t}, imaginary freq: {imaginary_freq}')

        write_output_file(output, f'{output_dir}/{idx}/ts_search.out')
        write_output_file(cos_final_traj, f'{output_dir}/{idx}/cos_final_traj.xyz')

        if tsopt is not None and imaginary_freq is not None:
            tsopt[1] = f"{imaginary_freq} \n"
            write_output_file(tsopt, f'{output_dir}/{idx}/ts_opt.xyz')

            if imaginary_freq < settings['min_ts_imaginary_freq'] and imaginary_freq > settings['max_ts_imaginary_freq']:
                # 4. IRC calculation in pysisphus
                t = time.time()
                output, forward_irc, backward_irc, forward_end, backward_end = pysisyphus_driver(
                    geometry_files=[f'{output_dir}/{idx}/ts_opt.xyz'],
                    charge=pc_complex.charge,
                    mult=pc_complex.mult,
                    job="irc",
                    solvent=solvent
                )
                print(f'IRC time: {time.time() - t} \n\n')
                write_output_file(output, f'{output_dir}/{idx}/irc.out')

                if None not in [backward_irc, forward_irc]:
                    backward_irc.reverse()
                    write_output_file(backward_irc + 10 * (tsopt + ["\n"]) + forward_irc, f'{output_dir}/{idx}/irc_path.xyz')
                    
                    if None not in [backward_end, forward_end]:
                        write_output_file(backward_end + ["\n"] + tsopt + ["\n"] + forward_end, f'{output_dir}/{idx}/reaction.xyz')

                        try:

                            finish_reaction()

                            true_rc_smi_list = [get_canonical_smiles(smi) for smi in reactant_smiles]
                            true_pc_smi_list = [get_canonical_smiles(smi) for smi in product_smiles]
                            pred_rc_smi_list = get_canonical_smiles_from_xyz_string("".join(backward_end), pc_complex.charge)
                            pred_pc_smi_list = get_canonical_smiles_from_xyz_string("".join(forward_end), pc_complex.charge)

                            print(true_rc_smi_list, pred_rc_smi_list)
                            print(true_pc_smi_list, pred_pc_smi_list)
                            print('\n\n')

                            if check_reaction_ends(
                                true_rc_smi_list,
                                true_pc_smi_list,
                                pred_rc_smi_list,
                                pred_pc_smi_list,
                            ):
                                # save as a template here
                                base_complex = [rc_complex, pc_complex][1 - isomorphism_idx].copy()
                                coords = np.array([
                                    [a.x, a.y, a.z] for a in read_xyz_string(tsopt)
                                ])
                                base_complex.coordinates = coords

                                for bond in bond_rearr.all:
                                    base_complex.graph.add_active_edge(*bond)
                                truncated_graph = get_truncated_active_mol_graph(graph=base_complex.graph, active_bonds=bond_rearr.all)
                                # bonds
                                for bond in bond_rearr.all:
                                    truncated_graph.edges[bond]["distance"] = base_complex.distance(*bond)
                                # cartesians
                                nx.set_node_attributes(truncated_graph, {node: base_complex.coordinates[node] for idx, node in enumerate(truncated_graph.nodes)}, 'cartesian')

                                ts_template = TStemplate(truncated_graph, species=base_complex)
                                ts_template.save(folder_path=f'{output_dir}/')

                                # also save tsopt + reaction + irc path
                                shutil.copy2(f'{output_dir}/{idx}/ts_opt.xyz', f'{output_dir}/ts_opt.xyz')
                                shutil.copy2(f'{output_dir}/{idx}/reaction.xyz', f'{output_dir}/reaction.xyz')
                                shutil.copy2(f'{output_dir}/{idx}/irc_path.xyz', f'{output_dir}/irc_path.xyz')
                                
                                print('finshed reaction \n\n')
                                break

                        except Exception as e:
                            print(e)
                            print('Failed to retrieve SMILES from IRC ends \n\n')

                    else:
                        print("IRC end opt failed\n\n")
                
                else:
                    print("IRC failed\n\n")
            
            else:
                print(f"TS curvature, ({imaginary_freq} cm-1), is not within allowed interval \n\n")
        
        else:
            print("TS optimization failed\n\n")

    # cleanup
    if os.path.exists('test_BR.txt'):
        os.remove('test_BR.txt')
    if os.path.exists('nul'):
        os.remove('nul')
    if os.path.exists('run.out'):
        os.remove('run.out')
        


# TODO: for now complexes must be specified with
# main substrate first & smaller substrates later
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "settings_file_path",
        help="Path to file containing the settings",
        type=str
    )
    args = parser.parse_args()

    # open yaml settings
    with open(args.settings_file_path, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    search_reaction_path(settings)
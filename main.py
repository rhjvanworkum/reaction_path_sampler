import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
import os
import time
import yaml
import argparse

from geodesic_interpolate.fileio import write_xyz
from geodesic_interpolate.interpolation import redistribute
from geodesic_interpolate.geodesic import Geodesic

import networkx as nx

import autode as ade
from autode.values import Distance
from autode.conformers.conformer import Conformer
from autode.atoms import Atoms
from autode.input_output import atoms_to_xyz_file

from autode.bond_rearrangement import get_bond_rearrangs, BondRearrangement
from autode.mol_graphs import reac_graph_to_prod_graph
from autode.geom import calc_heavy_atom_rmsd, get_rot_mat_kabsch
from autode.mol_graphs import (
    get_mapping_ts_template,
    get_truncated_active_mol_graph,
)

from src.ts_template import TStemplate
from src.interfaces.PYSISYPHUS import pysisyphus_driver
from src.molecule import read_xyz_string
from src.reactive_complex_sampler import ReactiveComplexSampler
from src.utils import read_trajectory_file, remove_whitespaces_from_xyz_strings, xyz_string_to_autode_atoms
from src.xyz2mol import get_canonical_smiles_from_xyz_string_ob

def set_autode_settings(settings):
    ade.Config.n_cores = settings['xtb_n_cores']
    ade.Config.XTB.path = os.environ["XTB_PATH"]
    ade.Config.rmsd_threshold = Distance(0.3, units="Å")
    ade.Config.num_complex_sphere_points = 10
    ade.Config.num_complex_random_rotations = 10

def generate_reactant_product_complexes(
    smiles_strings: List[str],
    solvent: str,
    settings: Any,
    save_path: str
) -> List[Conformer]:
    rps = ReactiveComplexSampler(
        smiles_strings=smiles_strings,
        solvent=solvent,
        settings=settings
    )

    complex = rps._get_ade_complex()

    if os.path.exists(save_path):
        conformers, _ = read_trajectory_file(save_path)
        conformer_list = [Conformer(
            atoms=xyz_string_to_autode_atoms(structure), 
            charge=complex.charge, 
            mult=complex.mult
        ) for structure in conformers]

    else:
        t = time.time()
        complexes = rps._sample_initial_complexes()
        print(f'time to do autode sampling: {time.time() - t}')

        conformer_list = []
        conformer_xyz_list = []
        for complex in complexes:
            conformers = rps.sample_reaction_complexes(complex=complex)
            for conformer in conformers:
                conformer_xyz_list.append(conformer)
                conformer_list.append(Conformer(
                    atoms=xyz_string_to_autode_atoms(conformer), 
                    charge=complex.charge, 
                    mult=complex.mult
                ))

        with open(save_path, 'w') as f:
            f.writelines(remove_whitespaces_from_xyz_strings(conformer_xyz_list))

    return complex, conformer_list

def get_reaction_isomorphisms(
    rc_complex: ade.Species,
    pc_complex: ade.Species,
) -> Tuple[BondRearrangement, Dict[int, int], int]:
    for idx, reaction_complexes in enumerate([
        [rc_complex, pc_complex],
        [pc_complex, rc_complex],
    ]):
        bond_rearrs = get_bond_rearrangs(reaction_complexes[1], reaction_complexes[0], name='test')
        if bond_rearrs is not None:
            for bond_rearr in bond_rearrs:
                graph1 = reaction_complexes[0].graph
                graph2 = reac_graph_to_prod_graph(reaction_complexes[1].graph, bond_rearr)
                mappings = []
                for isomorphism in nx.vf2pp_all_isomorphisms(
                    graph1, 
                    graph2, 
                    node_label="atom_label"
                ):
                    mappings.append(isomorphism)

                if len(mappings) > 0:
                    return bond_rearr, mappings, idx

def select_closest_reactant_product_pairs(
    n_reactant_product_pairs: int,
    rc_conformers: List[Conformer],
    pc_conformers: List[Conformer]
) -> List[Tuple[int]]:
    indices = []
    rmsds = []
    for i in range(len(rc_conformers)):
        for j in range(len(pc_conformers)):
            indices.append((i, j))
            rmsds.append(np.sqrt(np.mean((pc_conformers[j].coordinates - rc_conformers[i].coordinates)**2)))
    
    indices = np.array(indices)
    rmsds = np.array(rmsds)

    if len(rmsds) == 1:
        opt_idxs = [indices[0]]
    else:
        if len(rmsds) <= n_reactant_product_pairs:
            n_reactant_product_pairs = len(rmsds) - 1
            print(f'reduced amount of reactant & product pairs to {n_reactant_product_pairs}')
        opt_idxs = indices[np.argpartition(rmsds, n_reactant_product_pairs)[:n_reactant_product_pairs]]
    
    return opt_idxs

def compute_optimal_coordinates(
    rc_coordinates: np.ndarray,
    pc_coordinates: np.ndarray
) -> np.ndarray:
    assert rc_coordinates.shape == pc_coordinates.shape

    p_mat = np.array(rc_coordinates, copy=True)
    p_mat -= np.average(p_mat, axis=0)

    q_mat = np.array(pc_coordinates, copy=True)
    q_mat -= np.average(q_mat, axis=0)

    rot_mat = get_rot_mat_kabsch(p_mat, q_mat)

    fitted_coords = np.dot(rot_mat, p_mat.T).T
    return fitted_coords

def interpolate_geodesic(
    symbols: List[str],
    rc_coordinates: np.ndarray,
    pc_coordinates: np.ndarray,
    settings: Any
) -> Geodesic:
    X = [rc_coordinates, pc_coordinates]
    raw = redistribute(symbols, X, settings['nimages'], tol=settings['tol'])
    smoother = Geodesic(symbols, raw, settings['scaling'], threshold=settings['dist_cutoff'], friction=settings['friction'])
    try:
        smoother.smooth(tol=settings['tol'], max_iter=settings['maxiter'])
    except Exception as e:
        print(e)

    return smoother

def write_output_file(variable, name):
    if variable is not None:
        with open(name, 'w') as f:
            f.writelines(variable)

def remap_conformer(
    conformer: Conformer, 
    mapping: Dict[int, int]
) -> None:
    conformer._parent_atoms = Atoms(
        [conformer.atoms[i] for i in sorted(mapping, key=mapping.get)]
    )
    conformer._coordinates = np.array(
        [conformer._coordinates[i] for i in sorted(mapping, key=mapping.get)]
    )

def select_ideal_isomorphism(
    get_rc_fn: Callable,
    get_pc_fn: Callable,
    isomorphism_idx: int,
    isomorphisms: List[Dict[int, int]]
) -> Dict[int, int]:
    scores = []
    for isomorphism in isomorphisms:
        rc_conformers = get_rc_fn()
        pc_conformers = get_pc_fn()

        if isomorphism_idx == 0:
            for conformer in rc_conformers:
                remap_conformer(conformer, isomorphism)
        elif isomorphism_idx == 1:
            for conformer in pc_conformers:
                remap_conformer(conformer, isomorphism)
        else:
            raise ValueError(f"isomorphism idx can not be {isomorphism_idx}")
    
        rc_coords = np.stack([conf.coordinates for conf in rc_conformers])
        pc_coords = np.stack([conf.coordinates for conf in pc_conformers])
        score = (pc_coords[np.newaxis, :, :] - rc_coords[:, np.newaxis, :])**2
        scores.append(np.mean(np.sqrt(np.mean(score, axis=-1))))

    return isomorphisms[np.argmin(scores)]


def main(settings: Dict[str, Any]) -> None:
    output_dir = settings["output_dir"]
    reactant_smiles = settings["reactant_smiles"]
    product_smiles = settings["product_smiles"]
    solvent = settings["solvent"]

    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # set autode settings
    set_autode_settings(settings)

    # generate rc/pc complexes & reaction isomorphisms
    rc_complex, rc_conformers = generate_reactant_product_complexes(reactant_smiles, solvent, settings, f'{output_dir}/rcs.xyz')
    pc_complex, pc_conformers = generate_reactant_product_complexes(product_smiles, solvent, settings, f'{output_dir}/pcs.xyz')     
    bond_rearr, reaction_isomorphisms, isomorphism_idx = get_reaction_isomorphisms(rc_complex, pc_complex)

    # select best reaction isomorphism & remap reaction
    t = time.time()
    get_reactant_conformers = lambda: generate_reactant_product_complexes(reactant_smiles, solvent, settings, f'{output_dir}/rcs.xyz')[1]
    get_product_conformers = lambda: generate_reactant_product_complexes(product_smiles, solvent, settings, f'{output_dir}/pcs.xyz')[1]

    isomorphism = select_ideal_isomorphism(
        get_rc_fn=get_reactant_conformers,
        get_pc_fn=get_product_conformers,
        isomorphism_idx=isomorphism_idx,
        isomorphisms=reaction_isomorphisms
    )

    for conformer in [rc_conformers, pc_conformers][isomorphism_idx]:
        remap_conformer(conformer, isomorphism)
    print(f'Selecting best isomorphism: {time.time() - t}')


    # select closest pairs of reactants & products
    t = time.time()
    closest_pairs = select_closest_reactant_product_pairs(
        n_reactant_product_pairs=settings["n_reactant_product_pairs"],
        rc_conformers=rc_conformers,
        pc_conformers=pc_conformers
    )
    print(f'comparing RMSDs: {time.time() - t}\n\n')


    for idx, opt_idx in enumerate(closest_pairs):
        if not os.path.exists(f'{output_dir}/{idx}'):
            os.makedirs(f'{output_dir}/{idx}/')

        print(f'Working on Reactant-Complex pair {idx}')

        # 1. Optimally align the 2 conformers using kabsh algorithm
        t = time.time()
        rc_conformer = rc_conformers[opt_idx[0]]
        pc_conformer = pc_conformers[opt_idx[1]]
        rc_conformer._coordinates = compute_optimal_coordinates(rc_conformer.coordinates, pc_conformer.coordinates)
        atoms_to_xyz_file(rc_conformer.atoms, f'{output_dir}/{idx}/selected_rc.xyz')
        atoms_to_xyz_file(pc_conformer.atoms, f'{output_dir}/{idx}/selected_pc.xyz')
        print(f'aligning complexes: {time.time() - t}')


        # 2. Create a geodesic interpolation between 2 optimal conformers
        t = time.time()
        curve = interpolate_geodesic(
            pc_complex.atomic_symbols, 
            rc_conformer.coordinates, 
            pc_conformer.coordinates,
            settings
        )
        write_xyz(f'{output_dir}/{idx}/geodesic_path.trj', pc_complex.atomic_symbols, curve.path)
        write_xyz(f'{output_dir}/{idx}/geodesic_path.xyz', pc_complex.atomic_symbols, curve.path)
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

            if imaginary_freq < settings['min_ts_imaginary_freq']:
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
                            rc_smiles = get_canonical_smiles_from_xyz_string_ob("".join(backward_end))
                            pc_smiles = get_canonical_smiles_from_xyz_string_ob("".join(forward_end))
                            print(reactant_smiles, rc_smiles)
                            print(product_smiles, pc_smiles)

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
                            ts_template.save(folder_path=f'{output_dir}/{idx}/')
                            # ts_template.save(folder_path='./templates/')

                            print('\n\n')

                        except Exception as e:
                            print('Failed to retrieve SMILES from IRC ends \n\n')

                    else:
                        print("IRC end opt failed\n\n")
                
                else:
                    print("IRC failed\n\n")
            
            else:
                print(f"TS curvature is too low ({imaginary_freq} cm-1)\n\n")
        
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

    main(settings)
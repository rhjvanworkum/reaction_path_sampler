import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Any
import os
import time
import yaml

from geodesic_interpolate.fileio import write_xyz
from geodesic_interpolate.interpolation import redistribute
from geodesic_interpolate.geodesic import Geodesic

import networkx as nx

import autode as ade
from autode.values import Distance
from autode.conformers.conformer import Conformer
from autode.atoms import Atoms
from autode.input_output import atoms_to_xyz_file

from autode.bond_rearrangement import get_bond_rearrangs
from autode.mol_graphs import reac_graph_to_prod_graph
from autode.geom import calc_heavy_atom_rmsd, get_rot_mat_kabsch

from PYSISYPHUS import pysisyphus_driver
from reaction_pathway_sampler import ReactionPathwaySampler
from utils import traj2str, xyz_string_to_autode_atoms
from xyz2mol import canonical_smiles_from_xyz_string

def set_autode_settings(settings):
    ade.Config.n_cores = settings['xtb_n_cores']
    ade.Config.XTB.path = os.environ["XTB_PATH"]
    ade.Config.rmsd_threshold = Distance(0.3, units="Å")
    ade.Config.num_complex_sphere_points = 10
    ade.Config.num_complex_random_rotations = 10

def remove_whitespace(
    _lines: List[str]
) -> str:
    lines = []
    for line in _lines:
        lines += list(filter(lambda x: len(x) > 0, line.split('\n')))

    for i in range(len(lines)):
        j = 0
        while j < len(lines[i]) and lines[i][j].isspace():
            j += 1
        lines[i] = lines[i][j:]
    output_text = '\n'.join(lines)
    return output_text


def generate_reactant_product_complexes(
    smiles_strings: List[str],
    solvent: str,
    reactive_coordinate: List[int],
    settings: Any,
    save_path: str
) -> List[Conformer]:
    rps = ReactionPathwaySampler(
        smiles_strings=smiles_strings,
        solvent=solvent,
        settings=settings,
        n_initial_complexes=settings['n_initial_complexes']
    )

    if os.path.exists(save_path):
        complex = rps._get_ade_complex()
        conformers, _ = traj2str(save_path)
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
            conformers = rps.sample_reaction_complexes(
                complex=complex,
                reactive_coordinate=reactive_coordinate,
            )
            for conformer in conformers:
                conformer_xyz_list.append(conformer)
                conformer_list.append(Conformer(
                    atoms=xyz_string_to_autode_atoms(conformer), 
                    charge=complex.charge, 
                    mult=complex.mult
                ))

        with open(save_path, 'w') as f:
            f.writelines(remove_whitespace(conformer_xyz_list))

    return complex, conformer_list


def get_reaction_isomorphisms(
    rc_complex: ade.Species,
    pc_complex: ade.Species,
) -> None:
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
                    return mappings, idx

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
            # rmsds.append(calc_heavy_atom_rmsd(pc_conformers[j].atoms, rc_conformers[i].atoms)) 
    
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


def main(
    output_dir: str,
    settings_file_path: str,
    n_reactant_product_pairs: int,
    solvent: str,
    reactant_smiles: List[str],
    rc_rc: List[int],
    product_smiles: List[str],
    pc_rc: List[int],
):
    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # open yaml settings
    with open(settings_file_path, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    # set autode settings
    set_autode_settings(settings)


    # generate rc/pc complexes & reaction isomorphisms
    rc_complex, rc_conformers = generate_reactant_product_complexes(reactant_smiles, solvent, rc_rc, settings, f'{output_dir}/rcs.xyz')
    pc_complex, pc_conformers = generate_reactant_product_complexes(product_smiles, solvent, pc_rc, settings, f'{output_dir}/pcs.xyz')     
    reaction_isomorphisms, isomorphism_idx = get_reaction_isomorphisms(rc_complex, pc_complex)

    # select best reaction isomorphism & remap reaction
    t = time.time()
    get_reactant_conformers = lambda: generate_reactant_product_complexes(reactant_smiles, solvent, rc_rc, settings, f'{output_dir}/rcs.xyz')[1]
    get_product_conformers = lambda: generate_reactant_product_complexes(product_smiles, solvent, pc_rc, settings, f'{output_dir}/pcs.xyz')[1]

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
        n_reactant_product_pairs,
        rc_conformers,
        pc_conformers
    )
    print(f'comparing RMSDs: {time.time() - t}\n\n')


    for idx, opt_idx in enumerate(closest_pairs):
        if not os.path.exists(f'{output_dir}/{idx}'):
            os.makedirs(f'{output_dir}/{idx}/')

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
                            rc_smiles = canonical_smiles_from_xyz_string("\n".join(backward_end), rc_complex.charge)
                            pc_smiles = canonical_smiles_from_xyz_string("\n".join(forward_end), pc_complex.charge)
                            print(reactant_smiles, rc_smiles)
                            print(product_smiles, pc_smiles)
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
        

if __name__ == "__main__":    
    # TODO: for now complexes must be specified with
    # main substrate first & smaller substrates later

    output_dir = './scratch/claissen'
    settings_file_path = './scratch/settings.yaml'

    n_reactant_product_pairs = 5
    n_geodesic_retries = 3
    
    # reactant_smiles = ["[O-][N+](=O)CCCl", "[F-]"]
    # product_smiles = ["[O-][N+](=O)C=C", "F", "[Cl-]"]
    # rc_rc = [3, 6]
    # pc_rc = [3, 9]

    # reactant_smiles = ["[O-][N+](=O)CCCl", "[F-]"]
    # rc_rc = [4, 5]
    # product_smiles = ["[O-][N+](=O)CCF", "[Cl-]"]
    # pc_rc = [4, 10]

    # reactant_smiles = ["C1=CC=CO1", "C=C"]
    # product_smiles = ["C1=CC(O2)CCC12"]
    # rc_rc = [9, 11, 13, 14]
    # pc_rc = [4, 5, 12, 13]

    reactant_smiles = ["c1cccc(OCC=C)c1"]
    product_smiles = ["C1=CC=CC(CC=C)C1(=O)"]
    solvent = "Methanol"
    rc_rc = [5, 6]
    pc_rc = [7, 9]

    # reactant_smiles = ["C1=C(C(=O)O)C(Cl)=CO1", "C=CCNO"]
    # product_smiles = ["C(Cl)1=C(C(=O)O)C(O2)CC(CNO)C12"]
    # rc_rc = [13, 12, 17, 18]
    # pc_rc = [9, 8, 16, 17]


    main(
        output_dir,
        settings_file_path,
        n_reactant_product_pairs,
        solvent,
        reactant_smiles,
        rc_rc,
        product_smiles,
        pc_rc
    )
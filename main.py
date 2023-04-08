import numpy as np
from typing import List
import os
import yaml

from geodesic_interpolate.fileio import write_xyz
from geodesic_interpolate.interpolation import redistribute
from geodesic_interpolate.geodesic import Geodesic

import autode as ade
from autode.values import Distance
from autode.conformers.conformer import Conformer
from autode.atoms import Atoms
from autode.wrappers.XTB import XTB
from autode.input_output import atoms_to_xyz_file

from autode.bond_rearrangement import get_bond_rearrangs
from autode.mol_graphs import get_mapping
from autode.mol_graphs import reac_graph_to_prod_graph
from autode.geom import calc_heavy_atom_rmsd, get_rot_mat_kabsch

from PYSISYPHUS import pysisyphus_driver
from reaction_pathway_sampler import ReactionPathwaySampler
from utils import traj2str, xyz_string_to_autode_atoms

def interpolate_geodesic(
    symbols: List[str],
    rc_coordinates: np.ndarray,
    pc_coordinates: np.ndarray
):
    nimages = 20
    tol = 1e-2
    scaling = 1.7
    dist_cutoff = 3
    friction = 1e-2
    maxiter = 15

    X = [rc_coordinates, pc_coordinates]
    raw = redistribute(symbols, X, nimages, tol=tol * 5)
    smoother = Geodesic(symbols, raw, scaling, threshold=dist_cutoff, friction=friction)
    try:
        smoother.smooth(tol=tol, max_iter=maxiter)
    except Exception as e:
        print(e)

    return smoother

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


if __name__ == "__main__":
    ade.Config.n_cores = 4
    ade.Config.XTB.path = '/home/ruard/Programs/xtb-6.5.1/bin/xtb'
    ade.Config.rmsd_threshold = Distance(0.5, units="Å")
    method = XTB()

    output_dir = './scratch/da/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open('/home/ruard/code/test_reactions/settings.yaml', "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)


    # rc
    rps = ReactionPathwaySampler(
        smiles_strings=["C1=CC=CO1", "C=C"],
        settings=settings,
        n_initial_complexes=1
    )
    rc_conformers = rps.sample_reaction_complexes(
        reactive_coordinate=[9, 11, 13, 14],
        min=0.5,
        max=2.5,
        n_points=10
    )
    with open(f'{output_dir}rcs.xyz', 'w') as f:
        f.writelines('\n'.join(rc_conformers))
    # rc_conformers, _ = traj2str('rcs.xyz')

    rc_conformers = [Conformer(
        atoms=xyz_string_to_autode_atoms(structure), 
        charge=0, 
        mult=0
    ) for structure in rc_conformers]


    # pc
    pps = ReactionPathwaySampler(
        smiles_strings=["C1=CC(O2)CCC12"],
        settings=settings,
        n_initial_complexes=1
    )
    pc_conformers = pps.sample_reaction_complexes(
        reactive_coordinate=[4, 5, 12, 13],
        min=0.5,
        max=2.5,
        n_points=10
    )
    with open(f'{output_dir}pcs.xyz', 'w') as f:
        f.writelines('\n'.join(pc_conformers))
    # pc_conformers, _ = traj2str('pcs.xyz')

    pc_conformers = [Conformer(
        atoms=xyz_string_to_autode_atoms(structure), 
        charge=0, 
        mult=0
    ) for structure in pc_conformers]


    # TODO: should try this for both [reactant, product] & [product, reactant]
    # now compute atom ordering mapping & rearrange RC conformers to match product
    bond_rearr = get_bond_rearrangs(pps.ade_complex, rps.ade_complex, name='test')[0]
    mapping = get_mapping(
        graph1=rps.ade_complex.graph,
        graph2=reac_graph_to_prod_graph(pps.ade_complex.graph, bond_rearr),
    )
    for conformer in rc_conformers:
        conformer._parent_atoms = Atoms(
            [conformer.atoms[i] for i in sorted(mapping, key=mapping.get)]
        )
        conformer._coordinates = np.array(
            [conformer._coordinates[i] for i in sorted(mapping, key=mapping.get)]
        )


    # compute conformers closest in RMSD & rotate complexes to align optimally
    indices = []
    rmsds = []
    for i in range(len(rc_conformers)):
        for j in range(len(pc_conformers)):
            indices.append((i, j))
            rmsds.append(calc_heavy_atom_rmsd(pc_conformers[j].atoms, rc_conformers[i].atoms))
    opt_idx = indices[np.argmin(np.array(rmsds))]

    rc_conformer = rc_conformers[opt_idx[0]]
    pc_conformer = pc_conformers[opt_idx[1]]
    rc_conformer._coordinates = compute_optimal_coordinates(rc_conformer.coordinates, pc_conformer.coordinates)
    atoms_to_xyz_file(rc_conformer.atoms, f'{output_dir}selected_rc.xyz')
    atoms_to_xyz_file(pc_conformer.atoms, f'{output_dir}selected_pc.xyz')
    

    # interpolation
    curve = interpolate_geodesic(pps.ade_complex.atomic_symbols, rc_conformer.coordinates, pc_conformer.coordinates)
    write_xyz(f'{output_dir}geodesic_path.trj', pps.ade_complex.atomic_symbols, curve.path)
    write_xyz(f'{output_dir}geodesic_path.xyz', pps.ade_complex.atomic_symbols, curve.path)


    # pysisyphus
    cos_final_traj, tsopt, forward_irc, backward_irc = pysisyphus_driver(
        geometry_files=[f'{output_dir}geodesic_path.trj'],
        charge=0,
        mult=0,
        jobs=["cos", "tsopt", "irc"]
    )
    os.remove(f'{output_dir}geodesic_path.trj')

    if cos_final_traj is not None:
        with open(f'{output_dir}cos_final_traj.xyz', 'w') as f:
            f.writelines(cos_final_traj)
    if tsopt is not None:
        with open(f'{output_dir}ts_opt.xyz', 'w') as f:
            f.writelines(tsopt)
    if forward_irc is not None:
        with open(f'{output_dir}forward_irc.xyz', 'w') as f:
            f.writelines(forward_irc)
    if backward_irc is not None:
        with open(f'{output_dir}backward_irc.xyz', 'w') as f:
            f.writelines(backward_irc)

    
    # TODO: check if actual species are created at backword & forward pass
    # TODO: optimize ends of IRC for real minima
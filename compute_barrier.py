import argparse
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import os
import tempfile
import yaml
import numpy as np
from tqdm import tqdm

from autode.mol_graphs import (
    get_mapping_ts_template,
    get_truncated_active_mol_graph,
)

from src.conformational_sampling.metadyn_conformer_sampler import MetadynConformerSampler
from src.interfaces.ORCA import orca_driver
from src.interfaces.PYSISYPHUS import pysisyphus_driver
from src.interfaces.XTB import xtb_driver
from src.molecule import Molecule
from src.reaction_path.complexes import generate_reaction_complex
from src.reaction_path.reaction_graph import get_reaction_graph_isomorphism
from src.ts_template import TStemplate
from src.utils import read_trajectory_file

def get_constraints_from_template(
    path: str,
    reactant_smiles: str,
    product_smiles: str,
    settings
):
    rc_complex = generate_reaction_complex(reactant_smiles)
    pc_complex = generate_reaction_complex(product_smiles)
    bond_rearr, isomorphism, isomorphism_idx = get_reaction_graph_isomorphism(rc_complex, pc_complex, settings)
    truncated_graph = get_truncated_active_mol_graph(graph=pc_complex.graph, active_bonds=bond_rearr.all)

    ts_template = TStemplate(filename=os.path.join(path, 'template0.txt'))
    cartesian_constraints = {}
    mapping = get_mapping_ts_template(
        larger_graph=truncated_graph, smaller_graph=ts_template.graph
    )
    for node in truncated_graph.nodes:
        try:
            coords = ts_template.graph.nodes[mapping[node]]["cartesian"]
            cartesian_constraints[node] = coords
        except KeyError:
            print(f"Couldn't find a mapping for atom {node}")

    return cartesian_constraints

def compute_activation_energy_ensemble_xtb(args) -> float:
    r_geometry, ts_geometry, results_dir, solvent, settings = args

    r_geometry = r_geometry.split('\n')
    ts_geometry = ts_geometry.split('\n')
    cartesian_constraints = get_constraints_from_template(results_dir, settings['reactant_smiles'], settings['product_smiles'], settings)

    """ Compute energy of lowest reactant energy conformer """
    r_confs, _ = read_trajectory_file(os.path.join(results_dir, 'rcs.xyz'))
    energies = [
        xtb_driver(
            xyz_string=reactant,
            charge=0,
            mult=1,
            job="sp",
            method="2",
            solvent=solvent,
            n_cores=4
        ) for reactant in r_confs
    ]
    min_r_conf = r_confs[np.argmin(np.array(energies))]
    r_energy = orca_driver(
        xyz_string=min_r_conf,
        charge=0,
        mult=1,
        xc_functional="B3LYP",
        basis_set="6-31G",
        job="sp",
        solvent=solvent,
        n_cores=1
    )

    print(r_energy)


    """ Compute energy of lowest TS energy conformer """
    ts_conf_sampler = MetadynConformerSampler([], solvent, settings)
    ts_confs = ts_conf_sampler.sample_ts_conformers(
        complex=Molecule.from_xyz_string(ts_geometry, 0, 1),
        fixed_atoms=[str(k + 1) for k in sorted(cartesian_constraints.keys())]
    )

    # TS optimization loop thing
    max_tries = 5
    ts = None
    for i in range(max_tries):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xyz')
        try:
            with open(tmp.name, 'w') as f:
                f.write(ts_confs[i])
      
            _, tsopt, imaginary_freq = pysisyphus_driver(
                geometry_files=[tmp.name],
                charge=0,
                mult=1,
                job="ts_opt",
                solvent=settings['solvent'],
                n_mins_timeout=5 # settings['n_mins_timeout']
            )
            if tsopt is not None and imaginary_freq is not None:
                if imaginary_freq < -50 and imaginary_freq > -1000:
                    ts = tsopt
                    break
        
        finally:
            tmp.close()
            os.unlink(tmp.name)

    if ts is None:
        ts = "\n".join(ts_geometry)

    # ts_energy = xtb_driver(
    #     xyz_string=ts,
    #     charge=0,
    #     mult=1,
    #     job="sp",
    #     method="2",
    #     solvent=solvent,
    #     n_cores=4
    # )
    ts_energy = orca_driver(
        xyz_string=ts,
        charge=0,
        mult=1,
        xc_functional="B3LYP",
        basis_set="6-31G",
        job="sp",
        solvent=solvent,
        n_cores=1
    )

    print(ts_energy)


    if ts_energy is None or r_energy is None:
        return np.nan   
    else:
        return ts_energy - r_energy

def compute_barrier(
    base_dir: str,
    i: int
):
    results_dir = os.path.join(base_dir, f'{i}')
    reaction, _ = read_trajectory_file(os.path.join(results_dir, 'reaction.xyz'))
    
    r_geometry, ts_geometry, solvent = reaction[0], reaction[1], 'Methanol'
    with open(os.path.join(base_dir, f'{i}.yaml'), 'r') as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    settings['ts_ref_energy_threshold'] = [0, 100, 100]
    settings['ts_rmsd_threshold'] = [0, 2.0, 2.0]
    settings['ts_conf_energy_threshold'] = [0, 0.00001, 0.00001]
    settings['ts_rotational_threshold'] = [0, 0.04, 0.04]
    
    settings['metadyn_settings']['save'] = 100
    settings['md_time_per_atom'] = 0.5
    settings['md_settings']['dump'] = 100
    
    settings['n_processes'] = 4

    barrier = compute_activation_energy_ensemble_xtb(
        args=(r_geometry, ts_geometry, results_dir, solvent, settings)
    )

    return barrier


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "base_dir",
        type=str
    )
    parser.add_argument(
        "i",
        type=int
    )
    args = parser.parse_args()

    barrier = compute_barrier(args.base_dir, args.i)
    print('barrier: ', barrier)
    with open(os.path.join(args.base_dir, f'{args.i}/dft_barrier.txt'), 'w') as f:
        f.write(str(barrier))
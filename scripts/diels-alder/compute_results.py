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


def compute_activation_energy_xtb(args) -> float:
    r_geometry, ts_geometry, solvent = args
    r_energy = xtb_driver(
        xyz_string=r_geometry,
        charge=0,
        mult=1,
        job="sp",
        method="2",
        solvent=solvent,
        n_cores=4
    )
    ts_energy = xtb_driver(
        xyz_string=ts_geometry,
        charge=0,
        mult=1,
        job="sp",
        method="2",
        solvent=solvent,
        n_cores=4
    )

    if ts_energy is None or r_energy is None:
        return np.nan   
    else:
        return ts_energy - r_energy

def compute_activation_energy_orca(args) -> float:
    r_geometry, ts_geometry, solvent = args
    r_energy = orca_driver(
        xyz_string=r_geometry,
        charge=0,
        mult=1,
        xc_functional="B3LYP",
        basis_set="6-31G",
        job="sp",
        solvent=solvent,
        n_cores=1
    )
    ts_energy = orca_driver(
        xyz_string=ts_geometry,
        charge=0,
        mult=1,
        xc_functional="B3LYP",
        basis_set="6-31G",
        job="sp",
        solvent=solvent,
        n_cores=1
    )

    if ts_energy is None or r_energy is None:
        return np.nan   
    else:
        return ts_energy - r_energy


def compute_activation_energy_ensemble_xtb(args) -> float:
    r_geometry, ts_geometry, results_dir, solvent, settings = args

    r_geometry = r_geometry.split('\n')
    ts_geometry = ts_geometry.split('\n')
    cartesian_constraints = get_constraints_from_template(results_dir, settings['reactant_smiles'], settings['product_smiles'])

    r_conf_sampler = MetadynConformerSampler(settings['reactant_smiles'], solvent, settings)
    r_confs = r_conf_sampler.sample_conformers(
        initial_geometry=Molecule.from_xyz_string(r_geometry, 0, 1),
    )
    if len(r_confs) == 0:
        reactant = "\n".join(r_geometry)
    else:
        reactant = r_confs[0]

    r_energy = xtb_driver(
        xyz_string=reactant,
        charge=0,
        mult=1,
        job="sp",
        method="2",
        solvent=solvent,
        n_cores=4
    )
    
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

    ts_energy = xtb_driver(
        xyz_string=ts,
        charge=0,
        mult=1,
        job="sp",
        method="2",
        solvent=solvent,
        n_cores=4
    )

    if ts_energy is None or r_energy is None:
        return np.nan   
    else:
        return ts_energy - r_energy


if __name__ == "__main__":
    dataset_path = "./data/DA_regio_solvent_success.csv"
    dataset = pd.read_csv(dataset_path)
    base_dir = "./scratch/DA_test_solvent/"
    n_processes = 25

    # for i in dataset['uid'].values:
    #     bash_file_name = os.path.join(base_dir, f"{i}/job.sh")
    #     with open(bash_file_name, 'w') as f:
    #         f.writelines([
    #             '#!/bin/bash \n',
    #             'source env.sh \n',
    #             f'python -u compute_barrier.py {base_dir} {i}'
    #         ])
    #     os.system(f'sbatch --cpus-per-task=4 --time=02:00:00 --qos=cpus100 --output={base_dir}{i}/job_%A.out {os.path.join(base_dir, f"{i}/job.sh")}')


    """ Using single ponits"""
    ea_computation_arguments = []
    for i, solvent in zip(dataset['uid'].values, dataset['orca_solvent'].values):
        results_dir = os.path.join(base_dir, f'{i}')
        reaction, _ = read_trajectory_file(os.path.join(results_dir, 'reaction.xyz'))
        ea_computation_arguments.append((reaction[0], reaction[1], solvent))

    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        activation_energies = list(tqdm(executor.map(compute_activation_energy_orca, ea_computation_arguments), total=len(ea_computation_arguments), desc="Computing activation energies"))
    
    activation_energies = np.array(activation_energies)
    dataset['barrier'] = activation_energies

    # compute labels
    labels = []
    for _, row in dataset.iterrows():
        barrier = row['barrier']
        other_barriers = dataset[dataset['substrates'] == row['substrates']]['barrier']

        if np.isnan(barrier) or True in [np.isnan(val) for val in other_barriers.values]:
            labels.append(np.nan)
        else:
            label = int((barrier <= other_barriers).all())
            labels.append(label)
    dataset['orca_labels'] = labels

    # save dataset
    dataset.to_csv(dataset_path)
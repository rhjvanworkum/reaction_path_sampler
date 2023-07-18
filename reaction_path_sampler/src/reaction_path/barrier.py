import os
import tempfile
from typing import Callable, List
import numpy as np

from reaction_path_sampler.src.conformational_sampling.metadyn_conformer_sampler import MetadynConformerSampler
from reaction_path_sampler.src.interfaces.PYSISYPHUS import pysisyphus_driver
from reaction_path_sampler.src.interfaces.methods import xtb_single_point_method
from reaction_path_sampler.src.molecular_system import MolecularSystem
from reaction_path_sampler.src.utils import read_trajectory_file


def compute_barrier(
    reactant_conformers_file_path: str,
    ts_geometry: List[str],
    charge: int,
    mult: int,
    solvent: str,
    settings: dict,
    constraints: dict,
    method: Callable = xtb_single_point_method
) -> float:
    """ Compute energy of lowest reactant energy conformer """
    r_confs, _ = read_trajectory_file(reactant_conformers_file_path)
    energies = [method(reactant, charge, mult, solvent, n_cores=2) for reactant in r_confs]
    energies = [e for e in energies if e is not None]
    if len(energies) > 0:
        r_energy = min(energies)
    else:
        print('Warning: All DFT calcs failed for reactant conformers. Using XTB instead.')
        energies = [xtb_single_point_method(reactant, charge, mult, solvent, n_cores=2) for reactant in r_confs]
        energies = [e for e in energies if e is not None]
        r_energy = min(energies)

    if settings['sample_ts_conformers']:
        ts_conf_sampler = MetadynConformerSampler(
            smiles_strings=[], 
            solvent=solvent, 
            settings=settings
        )
        ts_confs = ts_conf_sampler.sample_ts_conformers(
            mol=MolecularSystem(
                smiles=None,
                rdkit_mol=None,
                geometry="".join(ts_geometry),
                charge=charge,
                mult=mult
            ),
            fixed_atoms=[str(k + 1) for k in sorted(constraints.keys())]
        )

        # TS optimization loop 
        # here ts conformers already come in ranked by energy from the sampler
        # therefore we are just trying top 5 to optimize & find lowest conformer
        max_tries = 5
        ts = None
        for i in range(min(len(ts_confs), max_tries)):
        # try:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.xyz')
            try:
                with open(tmp.name, 'w') as f:
                    f.write(ts_confs[i])
        
                _, tsopt, imaginary_freq = pysisyphus_driver(
                    geometry_files=[tmp.name],
                    charge=charge,
                    mult=mult,
                    job="ts_opt",
                    solvent=solvent,
                    n_mins_timeout=5 # settings['n_mins_timeout']
                )
                if tsopt is not None and imaginary_freq is not None:
                    if imaginary_freq < -50 and imaginary_freq > -1000:
                        ts = tsopt
                        break
            
            finally:
                tmp.close()
                os.unlink(tmp.name)


    if not settings['sample_ts_conformers'] or (settings['sample_ts_conformers'] and ts is None):
        ts = "".join(ts_geometry)

    ts_energy = method(ts, charge, mult, solvent, n_cores=2)
    print('computed TS energy: ', ts_energy)

    if ts_energy is None or r_energy is None:
        return np.nan   
    else:
        return ts_energy - r_energy
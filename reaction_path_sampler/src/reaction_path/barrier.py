import os
import tempfile
from typing import Callable, List
import numpy as np

from reaction_path_sampler.src.conformational_sampling.metadyn_conformer_sampler import MetadynConformerSampler
from reaction_path_sampler.src.interfaces.PYSISYPHUS import pysisyphus_driver
from reaction_path_sampler.src.interfaces.methods import xtb_single_point_method
# from reaction_path_sampler.src.molecule import Molecule
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
    r_energy = min(energies)

    if settings['sample_ts_conformers']:
        ts_conf_sampler = MetadynConformerSampler([], solvent, settings)
        # ts_confs = ts_conf_sampler.sample_ts_conformers(
        #     complex=Molecule.from_xyz_string(ts_geometry, 0, 1),
        #     fixed_atoms=[str(k + 1) for k in sorted(constraints.keys())]
        # )
        ts_confs = None

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

    if ts_energy is None or r_energy is None:
        return np.nan   
    else:
        return ts_energy - r_energy
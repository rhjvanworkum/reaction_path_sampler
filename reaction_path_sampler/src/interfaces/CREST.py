"""
File containing interface to CREST program
"""
import os
import subprocess

from autode.utils import run_in_tmp_environment, work_in_tmp_dir

from reaction_path_sampler.src.utils import read_trajectory_file


def crest_driver(
    ref_structure: str,
    ensemble_structures: str,
    ref_energy_threshold: float,
    rmsd_threshold: float,
    conf_energy_threshold: float,
    rotational_threshold: float,
    n_cores: int = 2
):
    flags = [
        f"-ewin {ref_energy_threshold}",
        f"-rthr {rmsd_threshold}",
        f"-ethr {conf_energy_threshold}",
        f"-bthr {rotational_threshold}"
    ]

    @work_in_tmp_dir(
        filenames_to_copy=[],
        kept_file_exts=(),
    )
    @run_in_tmp_environment(
        OMP_NUM_THREADS=n_cores, 
        MKL_NUM_THREADS=n_cores,
        GFORTRAN_UNBUFFERED_ALL=1
    )
    def execute_crest():
        with open('ref_struct.xyz', 'w') as f:
            f.writelines(ref_structure)
        with open('ensemble.xyz', 'w') as f:
            f.writelines(ensemble_structures)
        cmd = f'{os.environ["CREST_PATH"]} ref_struct.xyz --cregen ensemble.xyz {" ".join(flags)}'
        proc = subprocess.Popen(
            cmd.split(), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL, 
            text=True,
        )
        output = proc.communicate()[0]

        if os.path.exists('crest_ensemble.xyz'):
            structures, energies = read_trajectory_file("crest_ensemble.xyz")

        return structures
    
    structures = execute_crest()
    return structures


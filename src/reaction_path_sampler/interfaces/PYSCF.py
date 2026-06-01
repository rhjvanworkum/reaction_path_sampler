from typing import Literal, Optional
import os

from autode.utils import run_in_tmp_environment, work_in_tmp_dir

SOLVENT_CONSTANT_DICT = {
    'Methanol': 32.613
}

def pyscf_driver(
    xyz_string: str,
    charge: int,
    spin: int,
    job: Literal["sp"] = "sp",
    solvent: Optional[str] = None,
    basis_set: str = "6-31G",
    xc_functional: str = "B3LYP",
    n_cores: int = 8,
    max_cpus: Literal["50", "100", "150"] = "100"
):  
    @work_in_tmp_dir(
        filenames_to_copy=['./src/interfaces/run_pyscf.py'],
        kept_file_exts=(),
    )
    @run_in_tmp_environment(
        OMP_NUM_THREADS=n_cores, 
        MKL_NUM_THREADS=n_cores,
        GFORTRAN_UNBUFFERED_ALL=1
    )
    def execute_pyscf():
        with open('temp.xyz', 'w') as f:
            f.writelines(xyz_string)
        
        os.system(f'srun --cpus-per-task={n_cores} --time=01:00:00 --qos=cpus{max_cpus} python3 -u run_pyscf.py --xyz_file temp.xyz --charge {charge} --spin {spin} --solvent {solvent}')
        
        if os.path.exists('output.txt'):
            try:
                with open('output.txt', 'r') as f:
                    energy = float(f.readlines()[0])
            except:
                energy = None
        else:
            energy = None

        return energy
    
    return execute_pyscf()
"""
File containing interface to XTB
"""
import os
import subprocess
from typing import Any, Dict, List, Literal, Optional
import time

from autode.utils import run_in_tmp_environment, work_in_tmp_dir

from src.utils import read_trajectory_file


def xtb_driver(
    xyz_string: str,
    charge: int,
    mult: int,
    job: Literal["sp", "opt", "scan", "metadyn"] = "sp",
    method: str = '1',
    solvent: Optional[str] = None,
    xcontrol_settings: Optional[str] = None,
    n_cores: int = 4
):
    if mult != 1:
        print(f'WARNING: multiplicity is {mult}')

    flags = [
        "--chrg",
        str(charge),
        "--uhf",
        str(mult - 1),
        "--gfn",
        str(method)
    ]

    if solvent is not None:
        flags += ["--gbsa", solvent]

    if xcontrol_settings is not None:
        flags += ["--input", 'mol.input']

    if job == "opt" or job == "scan":
        flags += ["--opt"]
    elif job == "metadyn":
        flags += ["--md"]

    @work_in_tmp_dir(
        filenames_to_copy=[],
        kept_file_exts=(),
    )
    @run_in_tmp_environment(
        OMP_NUM_THREADS=n_cores, 
        MKL_NUM_THREADS=n_cores,
        GFORTRAN_UNBUFFERED_ALL=1
    )
    def execute_xtb():
        # xcontrol file
        if xcontrol_settings is not None:
            with open('mol.input', 'w') as f:
                f.write(xcontrol_settings)
        
        # xyz file
        xyz_file_name = f'mol-{time.time()}.xyz'
        with open(xyz_file_name, 'w') as f:
            f.writelines(xyz_string)
        
        # execute xtb
        cmd = f'{os.environ["XTB_PATH"]} {xyz_file_name} {" ".join(flags)}'
        proc = subprocess.Popen(
            cmd.split(), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.DEVNULL, 
            text=True,
        )
        output = proc.communicate()[0]

        # with open
        energy = None
        for line in reversed(output.split('\n')):
            if "total E" in line:
                energy = float(line.split()[-1])
            if "TOTAL ENERGY" in line:
                energy = float(line.split()[-3])

        opt_structure = None
        if job == "opt" and os.path.exists('xtbopt.xyz'):
            opt_structures, _ = read_trajectory_file('xtbopt.xyz')
            opt_structure = opt_structures[0]

        md_structures, md_energies = None, None
        if job == "metadyn" and os.path.exists('xtb.trj'):
            md_structures, md_energies = read_trajectory_file("xtb.trj")

        scan_structures, scan_energies = None, None
        if job == "scan" and os.path.exists('xtbscan.log'):
            scan_structures, scan_energies  = read_trajectory_file("xtbscan.log")
   
        return energy, opt_structure, md_structures, md_energies, scan_structures, scan_energies
    
    energy, opt_structure, md_structures, md_energies, scan_structures, scan_energies = execute_xtb()
    
    if job == "sp":
        return energy
    elif job == "opt":
        return opt_structure
    elif job == "metadyn":
        return md_structures, md_energies
    elif job == "scan":
        return scan_structures, scan_energies

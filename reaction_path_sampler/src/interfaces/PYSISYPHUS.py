"""
File containing interface to Pysisyphus module
"""
import logging
from typing import Optional, Union, Literal, List, Any
import subprocess
import os
import shutil
import numpy as np
import h5py

from autode.utils import run_in_tmp_environment, work_in_tmp_dir

from reaction_path_sampler.src.utils import read_trajectory_file


def construct_geometry_block(
    files: Union[str, List[str]],
    type: Literal["cart", "dlc"] = "cart"
) -> str:
    string = "geom:\n"
    string += f" type: {type}\n"
    if isinstance(files, str):
        string += f" fn: {files}\n\n"
    else:
        string += f" fn: [{','.join(files)}]\n\n"
    return string

def construct_xtb_calculation_block(
    charge: int, 
    mult: int,
    solvent: Optional[str] = None
) -> str:
    string = f"calc:\n" 
    string += " type: xtb\n"
    string += f" charge: {charge}\n"
    string += f" mult: {mult}\n"
    string += " pal: 4\n"
    if solvent is not None:
        string += f" gbsa: {solvent}\n\n"
    else:
        string += "\n"
    return string

def construct_orca_calculation_block(
    charge: int, 
    mult: int,
    method: str = 'B3LYP',
    basis: str = '6-31G',
    solvent: Optional[str] = None
) -> str:
    string = f"calc:\n" 
    string += " type: orca\n"
    string += f" keywords: {method} {basis}\n"
    string += f" charge: {charge}\n"
    string += f" mult: {mult}\n"
    string += " pal: 4\n"
    string += "\n"
    return string

def construct_cos_block() -> str:
    return "cos:\n type: neb\n climb: True\n\n"

def construct_opt_block() -> str:
    return " ".join([
        "opt:\n",
        "type: lbfgs\n",
        "align: True\n",
        "rms_force: 0.01\n",
        "max_step: 0.04\n\n"
    ])

def construct_tsopt_block() -> str:
    string =  f"tsopt:\n" 
    string += f" type: rsirfo\n"
    string += f" do_hess: True\n"
    string += f" max_cycles: 75\n"
    string += f" thresh: gau_tight\n"
    string += f" hessian_recalc: 5\n"
    # string += f" geom:\n"
    # string += f"  type: cart\n"
    string += "\n"
    return string

def construct_irc_block() -> str:
    return "irc:\n type: eulerpc\n rms_grad_thresh: 0.0005\n\n"

def construct_endopt_block() -> str:
    return "endopt: \n\n"


def pysisyphus_driver(
    geometry_files: Any,
    charge: int,
    mult: int,
    job: Literal["ts_opt", "ts_search", "irc"],
    n_cores: int = 2,
    n_mins_timeout: int = 5,
    solvent: Optional[str] = None,
    method: Literal["xtb", "orca"] = "xtb"
):
    if mult != 1:
        logging.info(f'WARNING: multiplicity is {mult}')

    settings_string = construct_geometry_block(
        files=[file.split('/')[-1] for file in geometry_files],
        type="cart"
    )
    if method == "xtb":
        settings_string += construct_xtb_calculation_block(
            charge=charge,
            mult=mult,
            solvent=solvent
        )
    elif method == "orca":
        settings_string += construct_orca_calculation_block(
            charge=charge,
            mult=mult,
            solvent=solvent
        )

    if job == "ts_opt":
        settings_string += construct_tsopt_block()

    if job == "ts_search":
        settings_string += construct_opt_block()
        settings_string += construct_cos_block()
        settings_string += construct_tsopt_block()

    if job == "irc":
        settings_string += construct_irc_block()
        settings_string += construct_endopt_block()

    @work_in_tmp_dir(
        filenames_to_copy=geometry_files,
        kept_file_exts=(),
    )
    @run_in_tmp_environment(
        OMP_NUM_THREADS=n_cores, 
        MKL_NUM_THREADS=n_cores,
        GFORTRAN_UNBUFFERED_ALL=1
    )
    def execute_pysisyphus():
        # for file in geometry_files:
        #     shutil.copy(file, f'./temp/{os.path.basename(file)}')

        # os.chdir('./temp/')

        with open('settings.yaml', 'w') as f:
            f.writelines(settings_string)

        cmd = f'pysis settings.yaml'

        try:
            output = subprocess.check_output(
                cmd.split(),
                text=True,
                timeout=n_mins_timeout * 60
            )
        except Exception as e:
            output = ''
            logging.debug(e)
            if isinstance(e, subprocess.TimeoutExpired):
                logging.debug('PYSISPHUS PROCESS TIMED OUT')
        
        if job == "ts_opt":
            tsopt = None
            if os.path.exists('ts_opt.xyz'):
                with open('ts_opt.xyz', 'r') as f:
                    tsopt = f.readlines()
            else:
                if os.path.exists('ts_final_geometry.xyz'):
                    with open('ts_final_geometry.xyz', 'r') as f:
                        tsopt = f.readlines()

            imaginary_freq = None
            if os.path.exists('ts_final_hessian.h5'):
                f = h5py.File('./ts_final_hessian.h5')
                freqs = f['vibfreqs'][:]
                imaginary_freq = np.min(freqs)

            return output, tsopt, imaginary_freq

        if job == "ts_search":
            cos_final_traj = None
            try:
                file_list = []
                for _, _, files in os.walk(os.getcwd()):
                    for file in files:
                        file_list.append(file)
                cycle_files = list(filter(lambda x: 'cycle_' in x, file_list))
                file = cycle_files[max([int(file.split('_')[-1].split('.')[0]) for file in cycle_files])]
                cos_final_traj, _ = read_trajectory_file(file)
            except Exception as e:
                logging.debug(e)
        
            tsopt = None
            if os.path.exists('ts_opt.xyz'):
                with open('ts_opt.xyz', 'r') as f:
                    tsopt = f.readlines()

            imaginary_freq = None
            if os.path.exists('ts_final_hessian.h5'):
                f = h5py.File('./ts_final_hessian.h5')
                freqs = f['vibfreqs'][:]
                imaginary_freq = np.min(freqs)

            return output, cos_final_traj, tsopt, imaginary_freq  
        
        elif job == "irc":
            forward_irc, backward_irc = None, None
            forward_end, backward_end = None, None
            if os.path.exists('forward_irc.trj'):
                forward_irc, _ = read_trajectory_file('forward_irc.trj')
            if os.path.exists('backward_irc.trj'):
                backward_irc, _ = read_trajectory_file('backward_irc.trj')
            if os.path.exists('forward_end_opt.xyz'):
                with open('forward_end_opt.xyz', 'r') as f:
                    forward_end = f.readlines()   
            if os.path.exists('backward_end_opt.xyz'):
                with open('backward_end_opt.xyz', 'r') as f:
                    backward_end = f.readlines()   

            return output, forward_irc, backward_irc, forward_end, backward_end
    
    return execute_pysisyphus()

if __name__ == "__main__":
    out = pysisyphus_driver(
        geometry_files="geodesic_path.xyz",
        charge=0,
        mult=0,
        jobs=["cos", "tsopt", "irc"]
    )

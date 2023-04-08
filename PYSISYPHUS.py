from typing import Union, Literal, List, Any
import subprocess
import os

from autode.utils import run_in_tmp_environment, work_in_tmp_dir

from utils import traj2str

def construct_geometry_block(
    files: Union[str, List[str]],
    type: Literal["cart"] = "cart"
) -> str:
    string = "geom:\n"
    string += f" type: {type}\n"
    if isinstance(files, str):
        string += f" fn: {files}\n\n"
    else:
        string += f" fn: [{','.join(files)}]\n\n"
    return string

def construct_calculation_block(
    charge: int, 
    mult: int
) -> str:
    return f"calc:\n type: xtb\n charge: {charge}\n mult: {mult}\n pal: 4\n\n"

def construct_cos_block() -> str:
    return "cos:\n type: neb\n climb: True\n\n"

def construct_opt_block() -> str:
    return "opt:\n type: lbfgs\n align: True\n rms_force: 0.01\n max_step: 0.04\n\n"

def construct_tsopt_block() -> str:
    return "tsopt:\n type: rsirfo\n do_hess: True\n max_cycles: 75\n thresh: gau_tight\n hessian_recalc: 7\n\n"

def construct_irc_block() -> str:
    return "irc:\n type: eulerpc\n rms_grad_thresh: 0.0005\n\n"


def pysisyphus_driver(
    geometry_files: Any,
    charge: int,
    mult: int,
    jobs: List[str] = ["cos"],
    n_cores: int = 2
):
    settings_string = construct_geometry_block(
        files=[file.split('/')[-1] for file in geometry_files],
        type="cart"
    )
    settings_string += construct_calculation_block(
        charge=charge,
        mult=mult
    )
    settings_string += construct_opt_block()

    if "cos" in jobs:
        settings_string += construct_cos_block()
    if "tsopt" in jobs:
        settings_string += construct_tsopt_block()
    if "irc" in jobs:
        settings_string += construct_irc_block()

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
        with open('settings.yaml', 'w') as f:
            f.writelines(settings_string)

        cmd = f'pysis settings.yaml'
        proc = subprocess.Popen(
            cmd.split(), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
        )
        output = proc.communicate()
        
        cos_final_traj = None
        if "cos" in jobs:
            try:
                file_list = []
                for _, _, files in os.walk(os.getcwd()):
                    for file in files:
                        file_list.append(file)
                cycle_files = list(filter(lambda x: 'cycle_' in x, file_list))
                file = cycle_files[max([int(file.split('_')[-1].split('.')[0]) for file in cycle_files])]
                cos_final_traj, _ = traj2str(file)
            except Exception as e:
                print(e)
        
        tsopt, imaginary_mode = None, None
        if "tsopt" in jobs:
            if os.path.exists('ts_opt.xyz'):
                with open('ts_opt.xyz', 'r') as f:
                    tsopt = f.readlines()        
        
        forward_irc, backward_irc = None, None
        if "irc" in jobs:
            if os.path.exists('forward_irc.trj'):
                forward_irc, _ = traj2str('forward_irc.trj')
            if os.path.exists('backward_irc.trj'):
                backward_irc, _ = traj2str('backward_irc.trj')
        
        return cos_final_traj, tsopt, forward_irc, backward_irc
    
    cos_final_traj, tsopt, forward_irc, backward_irc = execute_pysisyphus()
    return cos_final_traj, tsopt, forward_irc, backward_irc

if __name__ == "__main__":
    out = pysisyphus_driver(
        geometry_files="geodesic_path.xyz",
        charge=0,
        mult=0,
        jobs=["cos", "tsopt", "irc"]
    )

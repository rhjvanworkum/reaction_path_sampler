from typing import Literal, Optional
import time
import autode as ade
from autode.atoms import Atom

import os
from autode.wrappers.ORCA import ORCA
from autode.utils import run_in_tmp_environment, work_in_tmp_dir

def get_autode_species(
    xyz_string: str,
    charge: int,
    mult: int,
    solvent: str
) -> ade.Species:
    atom_lines = xyz_string.split('\n')
    atoms = []

    for line in atom_lines:
        splits = line.split()
        if len(splits) == 4:
            symbol, x, y, z = splits
            atoms.append(Atom(
                symbol, float(x), float(y), float(z)
            ))

    return ade.Species(
        name=str(time.time()),
        atoms=atoms,
        charge=charge,
        mult=mult,
        solvent_name=solvent
    )

def orca_driver(
    xyz_string: str,
    charge: int,
    mult: int,
    job: Literal["sp"] = "sp",
    solvent: Optional[str] = None,
    basis_set: str = "6-31+G**",
    xc_functional: str = "B3LYP",  
    n_cores: int = 1
):  
    ade.Config.n_cores = n_cores
    ade.Config.ORCA.path = "/home/rhjvanworkum/orca/orca"
    ade.Config.ORCA.keywords.sp = [
        xc_functional, basis_set
    ]   

    @work_in_tmp_dir(
        filenames_to_copy=[],
        kept_file_exts=(),
    )
    def compute():
        ade_species = get_autode_species(xyz_string, charge, mult, solvent)
        try:
            ade_species.single_point(method=ORCA())
            return ade_species.energy
        except:
            return None
    
    return compute()

# def orca_driver(
#     xyz_string: str,
#     charge: int,
#     mult: int,
#     job: Literal["sp"] = "sp",
#     solvent: Optional[str] = None,
#     basis_set: str = "6-31G",
#     xc_functional: str = "B3LYP",  
#     n_cores: int = 8
# ):  
#     ade.Config.n_cores = n_cores
#     ade.Config.ORCA.path = "/home/rhjvanworkum/orca/orca"
#     ade.Config.ORCA.keywords.sp = [
#         xc_functional, basis_set
#     ]   

#     @work_in_tmp_dir(
#         filenames_to_copy=[],
#         kept_file_exts=(),
#     )
#     def compute():
#         ade_species = get_autode_species(xyz_string, charge, mult, solvent)
#         orca_method = ORCA()
#         keywords = ade.SinglePointKeywords(ade.Config.ORCA.keywords.sp)
#         calc = ade.Calculation(
#             name=f'{ade_species.name}-Calc',
#             molecule=ade_species,
#             method=orca_method,
#             keywords=keywords,
#             n_cores=n_cores
#         )
#         calc._executor.generate_input()

#         os.system(f'srun --nodes=1 --ntasks-per-node={n_cores} /home/rhjvanworkum/orca/orca {calc.input.filename}')

#         # do something
#     compute()
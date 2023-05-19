from typing import Union, List
from src.interfaces.ORCA import orca_driver

from src.interfaces.XTB import xtb_driver

def xtb_single_point_method(
    geometry: Union[str, List[str]],
    charge: int,
    mult: int,
    solvent: str,
    n_cores: int
) -> float:
    return xtb_driver(
        xyz_string=geometry,
        charge=charge,
        mult=mult,
        job="sp",
        method="2",
        solvent=solvent,
        n_cores=n_cores,
    )


def orca_single_point_method(
    geometry: Union[str, List[str]],
    charge: int,
    mult: int,
    solvent: str,
    n_cores: int
) -> float:
    return orca_driver(
        xyz_string=geometry,
        charge=charge,
        mult=mult,
        xc_functional="B3LYP",
        basis_set="6-31G",
        job="sp",
        solvent=solvent,
        n_cores=1
    )


barrier_calculation_methods_dict = {
    "xtb": xtb_single_point_method,
    "orca_B3LYP": orca_single_point_method
}
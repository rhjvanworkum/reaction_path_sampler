from typing import List, Dict, Any
from openbabel import pybel
import numpy as np

from src.interfaces.XTB import xtb_driver
from src.molecule import Molecule
from src.constants import bohr_ang
from src.utils import get_reactive_coordinate_value


def get_geometry_constraints(
    force: float,
    reactive_coordinate: List[int],
    curr_coordinate_val: float
):
    names = ['', '', 'distance', 'angle', 'dihedral']
    string = "$constrain\n"
    string += "  force constant=%f \n" % force
    if len(reactive_coordinate) == 2:
        string += "  distance: %i, %i, %f \n" % (reactive_coordinate[0], reactive_coordinate[1], curr_coordinate_val)
    return string

def get_scan_constraint(
    start: float,
    end: float,
    nsteps: int
):
    string = "$scan\n"
    string += '  1: %f, %f, %i \n' % (start, end, nsteps)
    return string

def get_wall_constraint(
    wall_radius: float
):
    string = "$wall\n"
    string += "  potential=logfermi\n"
    string += "  sphere:%f, all\n" % wall_radius
    return string

def get_metadynamics_constraint(
    mol: Molecule,
    settings: Dict[str, Any],
    n_mols: int,
    post_fix: str = ""
):
    string = "$md\n"
    for k, v in settings[f"md_settings{post_fix}"].items():
        string += f"  {k}={v}\n"
    string += f"  time: {mol.n_atoms * n_mols * settings[f'md_time_per_atom']}\n"

    string += "$metadyn\n"
    for k, v in settings[f"metadyn_settings"].items():
        string += f"  {k}={v}\n"
    return string   



def compute_wall_radius(
    complex: Molecule, 
    settings: Dict[str, Any],
) -> float:
    # Compute all interatomic distances
    distances = []
    for i in range(complex.n_atoms):
        for j in range(i):
            distances.append(
                np.sqrt(np.sum((complex.geometry[i].coordinates - complex.geometry[j].coordinates)**2))
            )

    # Cavity is 1.5 x maximum distance in diameter
    radius_bohr = 0.5 * max(distances) * settings["cavity_scale"] + 0.5 * settings["cavity_offset"]
    radius_bohr /= bohr_ang
    return radius_bohr

def compute_force_constant(
    complex: Molecule,
    settings: Dict[str, Any],
    reactive_coordinate: List[int],
    curr_coordinate_val: float,
):
    if len(reactive_coordinate) == 2:
        xcontrol_settings = get_geometry_constraints(
            settings['force_constant'],
            reactive_coordinate,
            curr_coordinate_val
        )
        xcontrol_settings += get_scan_constraint(
            curr_coordinate_val - 0.05, 
            curr_coordinate_val + 0.05, 
            5
        )

        structures, energies = xtb_driver(
            complex.to_xyz_string(),
            complex.charge,
            complex.mult,
            "scan",
            method='2',
            xcontrol_settings=xcontrol_settings,
            n_cores=2
        )

        mols = [pybel.readstring("xyz", s.lower()).OBMol for s in structures]
        x = [abs(get_reactive_coordinate_value(mol, reactive_coordinate)) for mol in mols]
        x = np.array(x)

        y = np.array(energies)
        p = np.polyfit(x, y, 2)
        k = 2*p[0]
        force_constant = float(k * bohr_ang)
    else:
        force_constant = 1.0
    return force_constant
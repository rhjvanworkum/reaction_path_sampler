from typing import Any

import numpy as np
from openbabel import pybel

from reaction_path_sampler.constants import bohr_ang
from reaction_path_sampler.interfaces.XTB import xtb_driver
from reaction_path_sampler.molecular_system import MolecularSystem
from reaction_path_sampler.utils import (
    get_adj_mat_from_mol_block_string,
    get_reactive_coordinate_value,
)


def comp_ad_mat_xtb(xyz_string: str, charge: int, mult: int, solvent: str):
    mol_block_string = xtb_driver(
        xyz_string=xyz_string, charge=charge, mult=mult, job="mol", solvent=solvent
    )
    pred_adj_mat = get_adj_mat_from_mol_block_string(mol_block_string)
    return pred_adj_mat


def get_geometry_constraints(
    force: float, reactive_coordinate: list[int], curr_coordinate_val: float
):
    string = "$constrain\n"
    string += f"  force constant={force:f} \n"
    if len(reactive_coordinate) == 2:
        string += (
            f"  distance: {reactive_coordinate[0]:d}, "
            f"{reactive_coordinate[1]:d}, {curr_coordinate_val:f} \n"
        )
    string += "$end\n"
    return string


def get_scan_constraint(start: float, end: float, nsteps: int):
    string = "$scan\n"
    string += f"  1: {start:f}, {end:f}, {nsteps:d} \n"
    string += "$end\n"
    return string


def get_wall_constraint(wall_radius: float):
    string = "$wall\n"
    string += "  potential=logfermi\n"
    string += f"  sphere:{wall_radius:f}, all\n"
    string += "$end\n"
    return string


def get_fixing_constraints(atom_idxs: list[int]) -> str:
    string = "$fix\n"
    string += f"  atoms: {','.join(atom_idxs)}\n"
    string += "$end\n"
    return string


def get_atom_constraints(atom_idxs: list[int], force_constant: float, reference_file: str):
    string = "$constrain\n"
    string += f"  atoms: {','.join(atom_idxs)}\n"
    string += f"  force constant={force_constant}\n"
    string += f"  reference={reference_file}\n"
    string += "$end\n"
    return string


def get_metadynamics_settings(
    mol: MolecularSystem, settings: dict[str, Any], n_mols: int, post_fix: str = ""
):
    string = "$md\n"
    for k, v in settings[f"md_settings{post_fix}"].items():
        string += f"  {k}={v}\n"
    string += f"  time: {mol.n_atoms * n_mols * settings['md_time_per_atom']}\n"
    string += "$end\n"

    string += "$metadyn\n"
    for k, v in settings["metadyn_settings"].items():
        string += f"  {k}={v}\n"
    string += "$end\n"
    return string


def compute_wall_radius(
    mol: MolecularSystem,
    settings: dict[str, Any],
) -> float:
    # Compute all interatomic distances
    distances = []
    for i in range(mol.n_atoms):
        for j in range(i):
            distances.append(
                np.sqrt(
                    np.sum(
                        (
                            mol.init_geometry_autode.coordinates[i]
                            - mol.init_geometry_autode.coordinates[j]
                        )
                        ** 2
                    )
                )
            )

    # Cavity is 1.5 x maximum distance in diameter
    radius_bohr = 0.5 * max(distances) * settings["cavity_scale"] + 0.5 * settings["cavity_offset"]
    radius_bohr /= bohr_ang
    return radius_bohr


def compute_force_constant(
    mol: MolecularSystem,
    settings: dict[str, Any],
    reactive_coordinate: list[int],
    curr_coordinate_val: float,
):
    if len(reactive_coordinate) == 2:
        xcontrol_settings = get_geometry_constraints(
            settings["force_constant"], reactive_coordinate, curr_coordinate_val
        )
        xcontrol_settings += get_scan_constraint(
            curr_coordinate_val - 0.05, curr_coordinate_val + 0.05, 5
        )

        structures, energies = xtb_driver(
            mol.init_geometry_xyz_string(),
            mol.charge,
            mol.mult,
            "scan",
            method="2",
            xcontrol_settings=xcontrol_settings,
            n_cores=2,
        )

        mols = [pybel.readstring("xyz", s.lower()).OBMol for s in structures]
        x = [abs(get_reactive_coordinate_value(mol, reactive_coordinate)) for mol in mols]
        x = np.array(x)

        y = np.array(energies)
        p = np.polyfit(x, y, 2)
        k = 2 * p[0]
        force_constant = float(k * bohr_ang)
    else:
        force_constant = 1.0
    return force_constant

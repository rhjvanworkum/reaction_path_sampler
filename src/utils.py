"""
general utils
"""
from typing import Dict, List, Tuple
from rdkit import Chem
import numpy as np
import os
import re
from openbabel import pybel

def get_canonical_smiles(smiles: str) -> str:
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))

def get_reactive_coordinate_value(
    mol: pybel.Molecule,
    reactive_coordinate: List[int]
) -> float:
    atoms = [mol.GetAtom(i) for i in reactive_coordinate]
    if len(atoms)==2:
        return atoms[0].GetDistance(atoms[1])
    if len(atoms)==3:
        return mol.GetAngle(*atoms)
    if len(atoms)==4:
        return mol.GetTorsion(*atoms)


def comment_line_energy(comment_line):
    m = re.search('-?[0-9]*\.[0-9]*', comment_line)
    if m:
        E = float(m.group())
    else:
        E = np.nan
    return E

def read_trajectory_file(filepath: str, index=None, as_list=False) -> Tuple[List[str], List[float]]:
    """Read an xyz file containing a trajectory."""
    structures = []
    energies = []
    k = 0
    with open(filepath, 'r') as f:
        while True:
            first_line = f.readline()
            # EOF -> blank line
            if not first_line:
                break

            this_mol = first_line
            if len("".join(first_line.split())) == 0:
                first_line = f.readline()
                this_mol = first_line
            # print(first_line)
            # print(len("".join(first_line.split())))
            # print('hello')
            natoms = int("".join(first_line.split()))

            comment_line = f.readline()
            this_mol += comment_line
            E = comment_line_energy(comment_line)

            for i in range(natoms):
                this_mol += f.readline()

            if index is None:
                structures += [this_mol]
                energies += [E]
            
            else:
                if k == index:
                    if as_list:
                        return [this_mol], [E]
                    else:
                        return this_mol, E

            k += 1
    return structures, energies

def remove_whitespaces_from_xyz_strings(
    xyz_string: List[str]
) -> str:
    lines = []
    for line in xyz_string:
        lines += list(filter(lambda x: len(x) > 0, line.split('\n')))

    for i in range(len(lines)):
        j = 0
        while j < len(lines[i]) and lines[i][j].isspace():
            j += 1
        lines[i] = lines[i][j:]
    output_text = '\n'.join(lines)
    return output_text

def xyz_string_to_geom(xyz_string: str) -> Tuple[List[str], np.array]:
    lines = xyz_string.split('\n')
    atoms, coords = [], []
    for line in lines[2:]:
        if len(line.split()) == 4:
            a, x, y, z = line.split()
            atoms.append(a)
            coords.append([float(x), float(y), float(z)])
    return atoms, np.array(coords)

def geom_to_xyz_string(atoms: List[str], geom: np.array) -> str:
    lines = []
    lines.append(f'{len(atoms)}')
    lines.append('')
    for a, coord in zip(atoms, geom):
        lines.append(f'{a} {coord[0]} {coord[1]} {coord[2]}')
    return "\n".join(lines)

"""
autodE utils
"""
import autode as ade
from autode.conformers.conformer import Conformer
from autode.atoms import Atoms
from autode.values import Distance
from autode.atoms import Atom as AutodeAtom
from autode.exceptions import XYZfileWrongFormat


def set_autode_settings(settings):
    ade.Config.n_cores = settings['xtb_n_cores']
    ade.Config.XTB.path = os.environ["XTB_PATH"]
    ade.Config.rmsd_threshold = Distance(0.3, units="Å")
    ade.Config.num_conformers = settings["num_conformers"]
    ade.Config.num_complex_sphere_points = settings["num_complex_sphere_points"]
    ade.Config.num_complex_random_rotations = settings["num_complex_random_rotations"]


def remap_conformer(
    conformer: Conformer, 
    mapping: Dict[int, int]
) -> Conformer:
    return Conformer(
        name=conformer.name,
        atoms=[conformer.atoms[i] for i in sorted(mapping, key=mapping.get)],
        charge=conformer.charge,
        mult=conformer.mult
    )

def sort_complex_conformers_on_distance(
    conformers: List[Conformer],
    mols: List[ade.Molecule] 
) -> List[Conformer]:
    """
    Returns a list of autodE confomers sorted on ascending distance between
    parts of a complex
    """
    distances = []
    for conformer in conformers:
        if len(mols) == 1:
            continue
        elif len(mols) == 2:
            centroid_1 = np.mean(np.array([atom.coord for atom in conformer.atoms[:len(mols[0].atoms)]]), axis=0)
            centroid_2 = np.mean(np.array([atom.coord for atom in conformer.atoms[len(mols[0].atoms):]]), axis=0)
            distances.append(np.linalg.norm(centroid_2 - centroid_1))
        elif len(mols) == 3:
            centroid_1 = np.mean(np.array([atom.coord for atom in conformer.atoms[:len(mols[0].atoms)]]), axis=0)
            centroid_2 = np.mean(np.array([atom.coord for atom in conformer.atoms[len(mols[0].atoms):len(mols[0].atoms) + len(mols[1].atoms)]]), axis=0)
            centroid_3 = np.mean(np.array([atom.coord for atom in conformer.atoms[len(mols[0].atoms) + len(mols[1].atoms):]]), axis=0)
            distances.append(np.linalg.norm(centroid_2 - centroid_1) + np.linalg.norm(centroid_3 - centroid_1) + np.linalg.norm(centroid_3 - centroid_2))
        else:
            raise ValueError(f"Why does complex contain {len(mols)} mols")
    
    if len(mols) == 1:
        return conformers
    else:
        return [conformers[i] for i in np.argsort(np.array(distances))]


def autode_conf_to_xyz_string(conf) -> str:
    str = f"{len(conf.atoms)}\n \n"
    for atom in conf.atoms:
        str += f"{atom.atomic_symbol} {round(atom.coord.x, 4)} {round(atom.coord.y, 4)} {round(atom.coord.z, 4)}\n"
    return str


def xyz_string_to_autode_atoms(xyz_file: str) -> Atoms:
    """
    From a .xyz file get a list of autode atoms

    ---------------------------------------------------------------------------
    Arguments:
        filename: .xyz filename

    Returns:
        (autode.atoms.Atoms): Atoms
    """
    atoms = Atoms()

    xyz_file = xyz_file.split('\n')

    try:
        # First item in an xyz file is the number of atoms
        n_atoms = int(xyz_file[0].split()[0])

    except (IndexError, ValueError):
        raise XYZfileWrongFormat("Number of atoms not found")

    # XYZ lines should be the following 2 + n_atoms lines
    xyz_lines = xyz_file[2 : n_atoms + 2]

    for i, line in enumerate(xyz_lines):

        try:
            atom_label, x, y, z = line.split()[:4]
            atoms.append(AutodeAtom(atomic_symbol=atom_label, x=x, y=y, z=z))

        except (IndexError, TypeError, ValueError):
            raise XYZfileWrongFormat(
                f"Coordinate line {i} ({line}) " f"not the correct format"
            )

    if len(atoms) != n_atoms:
        raise XYZfileWrongFormat(
            f"Number of atoms declared ({n_atoms}) "
            f"not equal to the number of atoms found "
            f"{len(atoms)}"
        )
    
    return atoms
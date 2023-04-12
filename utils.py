import numpy as np
import re
from typing import List

from openbabel import pybel

class Atom:
    def __init__(self, atomic_symbol, x, y, z) -> None:
        self.atomic_symbol = atomic_symbol
        self.x = x
        self.y = y
        self.z = z
  
    @property
    def coordinates(self):
        return np.array([self.x, self.y, self.z])

    @coordinates.setter
    def coordinates(self, coords):
        self.x = coords[0]
        self.y = coords[1]
        self.z = coords[2]

class Molecule:
    def __init__(self, geometry: List[Atom], charge: int = 0, mult: int = 0) -> None:
        self.geometry = geometry
        self.charge = charge
        self.mult = mult
        self.n_atoms = len(geometry)

    @classmethod
    def from_autode_mol(cls, species):
        return cls(
            geometry=[
                Atom(a.atomic_symbol, a.coord.x, a.coord.y, a.coord.z) for a in species.atoms
            ],
            charge=species.charge,
            mult=species.mult
        )
    
    def to_pybel(self):
        string = ""
        string += str(len(self.geometry)) + ' \n'
        string += '\n'
        for atom in self.geometry:
            string += atom.atomic_symbol
            for cartesian in ['x', 'y', 'z']:
                if getattr(atom, cartesian) < 0:
                    string += '         '
                else:
                    string += '          '
                string += "%.5f" % getattr(atom, cartesian)
            string += '\n'
        string += '\n'
        return pybel.readstring("xyz", string.lower()).OBMol
    
    def to_xyz_string(self) -> str:
        string = ""
        string += str(len(self.geometry)) + ' \n'
        string += '\n'
        for atom in self.geometry:
            string += atom.atomic_symbol
            for cartesian in ['x', 'y', 'z']:
                if getattr(atom, cartesian) < 0:
                    string += '         '
                else:
                    string += '          '
                string += "%.5f" % getattr(atom, cartesian)
            string += '\n'
        string += '\n'
        return string

    def to_xyz(self, filename) -> None:
        with open(filename, 'w') as f:
            f.writelines(self.to_xyz_string())


def comment_line_energy(comment_line):
    m = re.search('-?[0-9]*\.[0-9]*', comment_line)
    if m:
        E = float(m.group())
    else:
        E = np.nan
    return E

def traj2str(filepath, index=None, as_list=False):
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
    return structures,energies





def read_xyz_file(filename):
  atoms = []

  with open(filename) as f:
    n_atoms = int(f.readline())
    _ = f.readline()

    for i in range(n_atoms):
      data = f.readline().replace('\n', '').split(' ')
      data = list(filter(lambda a: a != '', data))
      atoms.append(Atom(data[0], float(data[1]), float(data[2]), float(data[3])))

  return atoms


def write_xyz_file(atoms: List[Atom], filename: str):
  with open(filename, 'w') as f:
    f.write(str(len(atoms)) + ' \n')
    f.write('\n')

    for atom in atoms:
        f.write(atom.atomic_symbol)
        for cartesian in ['x', 'y', 'z']:
            if getattr(atom.coord, cartesian) < 0:
                f.write('         ')
            else:
                f.write('          ')
            f.write("%.5f" % getattr(atom.coord, cartesian))
        f.write('\n')
    
    f.write('\n')


import os

from typing import Collection, List, Optional
from autode.atoms import Atoms
from autode.atoms import Atom as AutodeAtom
from autode.exceptions import XYZfileDidNotExist
from autode.exceptions import XYZfileWrongFormat
from autode.log import logger
from autode.utils import StringDict

def _check_xyz_file_exists(filename: str) -> None:

    if not os.path.exists(filename):
        raise XYZfileDidNotExist(f"{filename} did not exist")

    if not filename.endswith(".xyz"):
        raise XYZfileWrongFormat("xyz file must have a .xyz file extension")

    return None

def conf_to_xyz_string(conf) -> str:
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
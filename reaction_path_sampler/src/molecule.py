"""
Internal Atom & Molecule data type
"""

import numpy as np
from typing import List, Tuple
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
    def __init__(
            self, 
            geometry: List[Atom], 
            charge: int = 0, 
            mult: int = 0
        ) -> None:
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

    @classmethod
    def from_xyz_string(cls, xyz_string: str, charge: int, mult: int):
        return cls(geometry=read_xyz_string(xyz_string), charge=charge, mult=mult)
    
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

    def to_geometry(self) -> Tuple[str, np.array]:
        symbols = [a.atomic_symbol for a in self.geometry]
        coords = np.array([[a.x, a.y, a.z] for a in self.geometry])
        return symbols, coords

    def to_xyz(self, filename) -> None:
        with open(filename, 'w') as f:
            f.writelines(self.to_xyz_string())



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


def read_xyz_string(xyz_string):
    atoms = []

    n_atoms = int(xyz_string[0])
    _ = xyz_string[1]

    for i in range(n_atoms):
        data = xyz_string[2 + i].replace('\n', '').split(' ')
        data = list(filter(lambda a: a != '', data))
        atoms.append(Atom(data[0], float(data[1]), float(data[2]), float(data[3])))

    return atoms

def parse_geometry_from_xyz_string(xyz_string):
    if type(xyz_string) == str:
        xyz_string = xyz_string.split('\n')

    symbols, coords = [], []
    n_atoms = int(xyz_string[0])
    _ = xyz_string[1]

    for i in range(n_atoms):
        data = xyz_string[2 + i].replace('\n', '').split(' ')
        data = list(filter(lambda a: a != '', data))
        symbols.append(data[0])
        coords.append([float(data[1]), float(data[2]), float(data[3])])

    return symbols, np.array(coords)

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
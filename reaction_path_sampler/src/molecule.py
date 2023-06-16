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
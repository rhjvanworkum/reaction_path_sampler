import numpy as np

from reaction_path_sampler.src.interfaces.lewis import compute_adjacency_matrix
from reaction_path_sampler.src.molecule import parse_geometry_from_xyz_string
from reaction_path_sampler.src.xyz2mol import xyz2AC, __ATOM_LIST__


file1, file2 = 'loose.xyz', 'bonded.xyz'

with open(file1, 'r') as f:
    conformer1 = f.readlines()

with open(file2, 'r') as f:
    conformer2 = f.readlines()

symbols, coords = parse_geometry_from_xyz_string(conformer1)
symbols = [
    __ATOM_LIST__.index(s.lower()) + 1 for s in symbols
]
# adj_matrix_1 = compute_adjacency_matrix(symbols, coords)
adj_matrix_1, _ = xyz2AC(symbols, coords, 0, use_huckel=True)

symbols, coords = parse_geometry_from_xyz_string(conformer2)
symbols = [
    __ATOM_LIST__.index(s.lower()) + 1 for s in symbols
]
# adj_matrix_2 = compute_adjacency_matrix(symbols, coords)
adj_matrix_2, _ = xyz2AC(symbols, coords, 0, use_huckel=True)

print(np.abs(adj_matrix_1 - adj_matrix_2))
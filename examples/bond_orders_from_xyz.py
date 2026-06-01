"""Infer connectivity and bond orders from a 3D geometry.

Reads a small bundled ethene geometry, builds the adjacency matrix from
interatomic distances, and runs the Lewis-structure search to assign bond
orders -- recovering the C=C double bond. This is the same machinery
``MolecularSystem`` uses internally.

Runnable as-is (only needs numpy/scipy):

    uv run examples/bond_orders_from_xyz.py
"""

from pathlib import Path

import numpy as np

from reaction_path_sampler.graphs.lewis import compute_adjacency_matrix, find_lewis
from reaction_path_sampler.molecule import parse_geometry_from_xyz_string

xyz_path = Path(__file__).parent / "data" / "ethene.xyz"
symbols, coords = parse_geometry_from_xyz_string(xyz_path.read_text())

adjacency = np.array(compute_adjacency_matrix(symbols, coords)).astype(int)
bond_matrix = np.array(find_lewis(symbols, adjacency, q_tot=0, b_mat_only=True, verbose=False))[
    0
].astype(int)

print(f"atoms: {symbols}\n")
print("adjacency matrix (who is bonded to whom):")
print(adjacency, "\n")
print("bond-order matrix (single=1, double=2, ...):")
print(bond_matrix, "\n")
print(f"C(0)=C(1) bond order: {bond_matrix[0, 1]}  (expected 2 for ethene)")

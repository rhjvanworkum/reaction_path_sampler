"""Golden-value tests for the vendored Lewis-structure / adjacency code.

``graphs/lewis`` is vendored from YARP and underpins
``MolecularSystem.compute_bond_order_matrix``. It is treated here as a black
box: these tests pin its *current* observable output (including the unusual
``(1000, N, N)`` stack that ``find_lewis`` returns) so any future cleanup of the
module cannot silently change bond-order inference. They need only numpy/scipy.
"""

import numpy as np

from reaction_path_sampler.graphs.lewis import compute_adjacency_matrix, find_lewis


def test_compute_adjacency_matrix_water():
    elements = ["O", "H", "H"]
    geo = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
    adj = np.array(compute_adjacency_matrix(elements, geo)).astype(int)
    expected = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(adj, expected)


def test_find_lewis_returns_stack_of_1000():
    # current behaviour: candidate de-duplication is a no-op, so find_lewis
    # returns a stack of 1000 (identical) bond-order matrices.
    elements = ["O", "H", "H"]
    adj = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    bm = np.array(find_lewis(elements, adj, q_tot=0, b_mat_only=True, verbose=False))
    assert bm.ndim == 3
    assert bm.shape == (1000, 3, 3)


def test_find_lewis_ethene_double_bond():
    elements = ["C", "C", "H", "H", "H", "H"]
    adj = np.array(
        [
            [0, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ]
    )
    bm = np.array(find_lewis(elements, adj, q_tot=0, b_mat_only=True, verbose=False))
    bond_mat = bm[0].astype(int)
    expected = np.array(
        [
            [0, 2, 1, 1, 0, 0],
            [2, 0, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(bond_mat, expected)
    # the C=C bond order is 2
    assert bond_mat[0, 1] == 2

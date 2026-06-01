"""Characterization tests for reaction-end checking.

``reaction_ends`` is the success/failure gate for every simulated reaction, and
it is pure SMILES / adjacency-matrix logic (no QM binaries), so it is pinned
here in detail. These tests capture the *current* observable behaviour so the
upcoming refactor cannot change the verdicts.
"""

import numpy as np
import pytest

from reaction_path_sampler.reaction_path.reaction_ends import (
    check_product_connectivity,
    check_reactant_product_graphs_identical,
    check_reactant_product_graphs_threshold,
    check_reaction_ends_by_graph_topology,
    check_reaction_ends_by_smiles,
)


# --------------------------------------------------------------------------- #
# check_reaction_ends_by_smiles
# --------------------------------------------------------------------------- #
def test_smiles_exact_match():
    assert check_reaction_ends_by_smiles(["CC"], ["C=C"], ["CC"], ["C=C"]) is True


def test_smiles_swapped_match():
    # reactants/products swapped between "true" and "pred" still counts as a match
    assert check_reaction_ends_by_smiles(["CC"], ["C=C"], ["C=C"], ["CC"]) is True


def test_smiles_match_is_order_insensitive():
    # multi-fragment lists in a different order match (the function sorts first)
    assert check_reaction_ends_by_smiles(["O", "CC"], ["C=C"], ["CC", "O"], ["C=C"]) is True


def test_smiles_no_match():
    assert check_reaction_ends_by_smiles(["CC"], ["C=C"], ["CCC"], ["N"]) is False


def test_smiles_does_not_mutate_inputs():
    # regression: the function used to sort the caller's lists in place
    true_rc = ["O", "CC"]
    true_pc = ["C=C"]
    pred_rc = ["CC", "O"]
    pred_pc = ["C=C"]
    check_reaction_ends_by_smiles(true_rc, true_pc, pred_rc, pred_pc)
    assert true_rc == ["O", "CC"]
    assert pred_rc == ["CC", "O"]


# --------------------------------------------------------------------------- #
# check_product_connectivity
# --------------------------------------------------------------------------- #
def test_connectivity_multiproduct_short_circuits_false():
    # current behaviour: >=2 true products -> always False (the in-code TODO)
    assert check_product_connectivity(["CC"], ["C=C", "O"], ["CC"], ["C=C", "O"]) is False


def test_connectivity_matching_adjacency_true():
    # same reactants, product adjacency matrices identical -> True
    assert check_product_connectivity(["CC"], ["CCO"], ["CC"], ["CCO"]) is True


def test_connectivity_mismatched_adjacency_false():
    # same reactants but product adjacency matrices differ in shape -> False
    assert check_product_connectivity(["CC"], ["CCO"], ["CC"], ["C"]) is False


def test_connectivity_no_reactant_match_false():
    assert check_product_connectivity(["CC"], ["CCO"], ["NN"], ["NN"]) is False


# --------------------------------------------------------------------------- #
# graph-topology checks
# --------------------------------------------------------------------------- #
def _ring():
    # a tiny symmetric 3x3 adjacency matrix
    return np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _chain():
    return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])


def test_graphs_identical_direct():
    rc, pc = _chain(), _ring()
    assert check_reactant_product_graphs_identical(rc, pc, rc, pc) is True


def test_graphs_identical_swapped():
    rc, pc = _chain(), _ring()
    assert check_reactant_product_graphs_identical(rc, pc, pc, rc) is True


def test_graphs_identical_mismatch():
    rc, pc = _chain(), _ring()
    assert check_reactant_product_graphs_identical(rc, pc, rc, rc) is False


def test_graphs_threshold_allows_small_difference():
    rc = _chain()
    pc = _ring()
    pred_pc = pc.copy()
    pred_pc[0, 1] = pred_pc[1, 0] = 0  # remove one (symmetric) bond -> abs-diff sum == 2
    assert check_reactant_product_graphs_threshold(rc, pc, rc, pred_pc, 2) is True
    assert check_reactant_product_graphs_threshold(rc, pc, rc, pred_pc, 1) is False


def test_graph_topology_threshold_zero_requires_identity():
    rc, pc = _chain(), _ring()
    assert check_reaction_ends_by_graph_topology(rc, pc, rc, pc, 0) is True
    assert check_reaction_ends_by_graph_topology(rc, pc, rc, rc, 0) is False


def test_graph_topology_positive_threshold():
    rc = _chain()
    pc = _ring()
    pred_pc = pc.copy()
    pred_pc[0, 1] = pred_pc[1, 0] = 0
    assert check_reaction_ends_by_graph_topology(rc, pc, rc, pred_pc, 2) is True


def test_graph_topology_negative_threshold_raises():
    rc, pc = _chain(), _ring()
    with pytest.raises(ValueError):
        check_reaction_ends_by_graph_topology(rc, pc, rc, pc, -1)

"""Tests for MolecularSystem construction from SMILES.

MolecularSystem builds an autodE complex, so these tests require autodE (and
its openbabel/rdkit stack). They are skipped automatically where autodE is not
installed (e.g. the binary-free CI job) and run on a fully provisioned machine.
"""

import pytest

pytest.importorskip("autode")

from reaction_path_sampler.molecular_system import MolecularSystem  # noqa: E402


def test_smiles_initialization():
    system = MolecularSystem.from_smiles("C=C.CC(=O)")

    assert system.smiles == "C=C.CC(=O)"
    assert system.charge == 0
    assert system.mult == 1
    # init_geometry_autode replaces the old (non-existent) `init_geometry`
    assert system.init_geometry_autode.coordinates.shape == (13, 3)

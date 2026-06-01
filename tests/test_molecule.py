"""Tests for the lightweight Atom type and xyz parsing/writing helpers.

Pure numpy + stdlib, so they run anywhere (no QM stack required).
"""

import numpy as np

from reaction_path_sampler.molecule import (
    Atom,
    parse_geometry_from_xyz_string,
    read_xyz_file,
    read_xyz_string,
    write_xyz_file,
)

XYZ = "3\ncomment\nO 0.00000 0.00000 0.00000\nH 0.75700 0.58600 0.00000\nH -0.75700 0.58600 0.00000"


def test_atom_coordinates_roundtrip():
    a = Atom("C", 1.0, 2.0, 3.0)
    np.testing.assert_array_equal(a.coordinates, np.array([1.0, 2.0, 3.0]))
    a.coordinates = [4.0, 5.0, 6.0]
    assert (a.x, a.y, a.z) == (4.0, 5.0, 6.0)


def test_read_xyz_string():
    atoms = read_xyz_string(XYZ.split("\n"))
    assert [a.atomic_symbol for a in atoms] == ["O", "H", "H"]
    np.testing.assert_allclose(atoms[1].coordinates, [0.757, 0.586, 0.0])


def test_parse_geometry_from_xyz_string_accepts_str_and_list():
    s1, c1 = parse_geometry_from_xyz_string(XYZ)
    s2, c2 = parse_geometry_from_xyz_string(XYZ.split("\n"))
    assert s1 == s2 == ["O", "H", "H"]
    np.testing.assert_array_equal(c1, c2)
    assert c1.shape == (3, 3)


def test_write_then_read_xyz_file_roundtrip(tmp_path):
    # regression: write_xyz_file used to read a non-existent `atom.coord`
    # attribute and raised AttributeError for every call.
    atoms = [Atom("O", 0.0, 0.0, 0.0), Atom("H", 0.757, 0.586, 0.0), Atom("H", -0.757, 0.586, 0.0)]
    path = tmp_path / "mol.xyz"
    write_xyz_file(atoms, str(path))

    read_back = read_xyz_file(str(path))
    assert [a.atomic_symbol for a in read_back] == ["O", "H", "H"]
    for original, parsed in zip(atoms, read_back, strict=True):
        np.testing.assert_allclose(parsed.coordinates, original.coordinates, atol=1e-5)

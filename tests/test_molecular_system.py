from reaction_path_sampler.src.molecular_system import MolecularSystem


def test_smiles_initialization():
    system = MolecularSystem.from_smiles('C=C.CC(=O)')

    assert system.smiles == 'C=C.CC(=O)'
    assert system.charge == 0
    assert system.mult == 1
    assert system.init_geometry.coordinates.shape == (13, 3)

if __name__ == "__main__":
    test_smiles_initialization()
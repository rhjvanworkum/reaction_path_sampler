from typing import List
from rdkit import Chem
import numpy as np
import autode as ade
from autode.species import Complex
from reaction_path_sampler.src.reaction_path.complexes import generate_reaction_complex

from reaction_path_sampler.src.utils import get_canonical_smiles

class MolecularSystem:

    """
    A class to represent a molecular system. Can either be initialized from a molecular graph in the form of
    1) a SMILES string
    2) a RDKit Mol object
    3) a List of atoms and their connectivity matrix


    """

    def __init__(
        self,
        smiles: str,
        rdkit_mol: Chem.Mol,
        atoms_list: List[str],
        connectivity_matrix: np.ndarray,
    ) -> None:
        self.__smiles = smiles
        self.__rdkit_mol = rdkit_mol
        self.__atoms_list = atoms_list
        self.__connectivity_matrix = connectivity_matrix

        self.__autode_complex = generate_reaction_complex(smiles.split('.'))
        self.__charge = self.__autode_complex.charge
        self.__mult = self.__autode_complex.mult
        
        # we need to generate the networkx graph here

        # we need to compute lewis structure here
        self.__bond_electron_matrix = None

    @property
    def smiles(self) -> str:
        return self.__smiles
    
    @property
    def rdkit_mol(self) -> str:
        return self.__rdkit_mol
    
    @property
    def atoms_list(self) -> str:
        return self.__atoms_list
    
    @property
    def connectivity_matrix(self) -> str:
        return self.__connectivity_matrix

    @classmethod
    def from_smiles(cls, smiles: str) -> 'MolecularSystem':
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_mol = Chem.AddHs(rdkit_mol)
        if rdkit_mol is None:
            raise ValueError("Could not parse SMILES string")

        atoms_list = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
        connectivity_matrix = Chem.GetAdjacencyMatrix(rdkit_mol)

        return cls(smiles, rdkit_mol, atoms_list, connectivity_matrix)

    @classmethod
    def from_rdkit_mol(cls, rdkit_mol: str) -> 'MolecularSystem':
        smiles = Chem.MolToSmiles(rdkit_mol)
        smiles = get_canonical_smiles(smiles)
        atoms_list = [atom.GetSymbol() for atom in rdkit_mol.GetAtoms()]
        connectivity_matrix = Chem.GetAdjacencyMatrix(rdkit_mol)

        return cls(smiles, rdkit_mol, atoms_list, connectivity_matrix)

    @classmethod
    def from_graph(cls, atoms_list: List[str], connectivity_matrix: np.ndarray) -> 'MolecularSystem':
        pass
        
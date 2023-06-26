import time
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
import numpy as np
import autode as ade
from autode.conformers.conformer import Conformer
import networkx as nx
from autode.species import Complex
from reaction_path_sampler.src.graphs.lewis import find_lewis
from reaction_path_sampler.src.reaction_path.complexes import generate_reaction_complex
from reaction_path_sampler.src.reaction_path.reaction_graph import get_reaction_graph_isomorphism

from reaction_path_sampler.src.utils import autode_conf_to_xyz_string, get_canonical_smiles, remap_conformer, xyz_string_to_autode_atoms
from reaction_path_sampler.src.visualization.plotly import plot_networkx_mol_graph

class Reaction:

    def __init__(
        self,
        reactants: 'MolecularSystem',
        products: 'MolecularSystem',
        solvent: str
    ) -> None:
        self.reactants = reactants
        self.products = products
        self.solvent = solvent

        self._bond_rearr = None
        self._isomorphism = None
        self._isomorphism_idx = None

    def map_reaction(self, n_workers: int) -> None:
        bond_rearr, isomorphism, isomorphism_idx = get_reaction_graph_isomorphism(
            rc_complex=self.reactants.autode_complex, 
            pc_complex=self.products.autode_complex, 
            n_workers=n_workers,
            node_label="atom_label"
        )

        [self.reactants, self.products][isomorphism_idx].reorder_atoms(isomorphism)

        self._bond_rearr = bond_rearr
        self._isomorphism = isomorphism
        self._isomorphism_idx = isomorphism_idx

class FakeComplex:

    def __init__(self, geometry_string) -> None:
        pass


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
        mult: Optional[int] = None,
        charge: Optional[int] = None,
        geometry: Optional[str] = None
    ) -> None:
        self.__smiles = smiles
        self.__rdkit_mol = rdkit_mol
        self.__atoms_list = None
        self.__connectivity_matrix = None
        self.__graph = None
        self.__bond_order_matrix = None
        self.__bond_electron_matrix = None

        if geometry is not None:
            self.autode_complex = ade.Species(
                name=str(time.time()),
                atoms=xyz_string_to_autode_atoms(geometry),
                charge=charge,
                mult=mult
            )
            self.autode_complex.conformers = [
                Conformer(
                    name=str(time.time()),
                    atoms=xyz_string_to_autode_atoms(geometry),
                    charge=charge,
                    mult=mult
                )
            ]
        else:
            self.autode_complex = generate_reaction_complex(smiles.split('.'))
        
        self.charge = self.autode_complex.charge
        self.mult = self.autode_complex.mult
        if mult is not None:
            self.mult = mult
        if charge is not None:
            self.charge = charge

        self.connectivity_matrix = nx.to_numpy_array(self.autode_complex.graph)
        self.atoms_list = np.array([atom.atomic_symbol for atom in self.autode_complex.atoms])

        self.compute_bond_order_matrix()
        self.build_graph()
        # self.visualize_graph()

    @property
    def init_geometry_autode(self) -> ade.Species:
        return self.autode_complex.conformers[0]
    
    @property
    def init_geometry_xyz_string(self) -> str:
        return autode_conf_to_xyz_string(self.autode_complex.conformers[0])

    @property
    def n_atoms(self) -> int:
        return len(self.__atoms_list)

    @property
    def smiles(self) -> str:
        return self.__smiles
    
    @property
    def rdkit_mol(self) -> Chem.Mol:
        return self.__rdkit_mol
    
    @property
    def atoms_list(self) -> np.array:
        return self.__atoms_list

    @atoms_list.setter
    def atoms_list(self, atoms_list: List[str]) -> None:
        self.__atoms_list = atoms_list
    
    @property
    def connectivity_matrix(self) -> np.ndarray:
        return self.__connectivity_matrix

    @connectivity_matrix.setter
    def connectivity_matrix(self, connectivity_matrix: np.ndarray) -> None:
        self.__connectivity_matrix = connectivity_matrix

    @property
    def bond_order_matrix(self) -> np.ndarray:
        return self.__bond_order_matrix
    
    @bond_order_matrix.setter
    def bond_order_matrix(self, bond_order_matrix: np.ndarray) -> None:
        self.__bond_order_matrix = bond_order_matrix

    def compute_bond_order_matrix(self) -> None:
        self.bond_order_matrix = np.array(find_lewis(
            self.atoms_list,
            self.connectivity_matrix, 
            q_tot=self.charge,
            b_mat_only=True,
            verbose=False
        ))

    @property
    def graph(self) -> nx.Graph:
        return self.__graph

    @graph.setter
    def graph(self, graph: nx.Graph) -> None:
        self.__graph = graph

    def build_graph(self) -> None:
        graph = nx.from_numpy_array(self.connectivity_matrix)
        nx.set_node_attributes(graph, dict(enumerate(self.atoms_list)), name="atom_label")
        nx.set_node_attributes(graph, dict(enumerate(self.init_geometry_autode.coordinates)), name="cartesian")
        self.graph = graph

    def visualize_graph(self) -> None:
        plot_networkx_mol_graph(self.graph)

    @property
    def bond_electron_matrix(self) -> str:
        return self.__bond_electron_matrix
    
    @bond_electron_matrix.setter
    def bond_electron_matrix(self, bond_electron_matrix: np.ndarray) -> None:
        self.__bond_electron_matrix = bond_electron_matrix

    def reorder_atoms(self, mapping: Dict[int, int]) -> None:
        self.autode_complex.conformers = [remap_conformer(conf, mapping) for conf in self.autode_complex.conformers]
        ordered_idxs = [i for i in sorted(mapping, key=mapping.get)]
        self.atoms_list = self.atoms_list[ordered_idxs]
        self.connectivity_matrix = self.connectivity_matrix[ordered_idxs, :][:, ordered_idxs]
        self.compute_bond_order_matrix()
        self.build_graph()

    @classmethod
    def from_smiles(
        cls, 
        smiles: str,
        mult: Optional[int] = None,
    ) -> 'MolecularSystem':
        rdkit_mol = Chem.MolFromSmiles(smiles)
        rdkit_mol = Chem.AddHs(rdkit_mol)
        if rdkit_mol is None:
            raise ValueError("Could not parse SMILES string")
        return cls(smiles, rdkit_mol, mult)

    @classmethod
    def from_rdkit_mol(cls, rdkit_mol: str) -> 'MolecularSystem':
        smiles = Chem.MolToSmiles(rdkit_mol)
        smiles = get_canonical_smiles(smiles)
        return cls(smiles, rdkit_mol)

    @classmethod
    def from_graph(cls, atoms_list: List[str], connectivity_matrix: np.ndarray) -> 'MolecularSystem':
        pass
        
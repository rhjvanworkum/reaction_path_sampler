from typing import List
from rdkit import Chem
from rdkit.Chem import AllChem
from rxnmapper import RXNMapper
import numpy as np
import time
import networkx as nx

from autode.atoms import Atom
import autode as ade
from autode.species.complex import Complex
from autode.conformers import Conformer

from reaction_path_sampler.src.reaction_path.complexes import generate_reaction_complex


def map_rxn_smiles(rxn_smiles: str) -> str:
    """
    Map a reaction smiles using RXN Mapper
    """
    rxn_mapper = RXNMapper()
    return rxn_mapper.get_attention_guided_atom_maps([rxn_smiles])[0]['mapped_rxn']

def correct_atom_map_numbers(
    reactants: List[Chem.Mol], 
    products: List[Chem.Mol]
) -> None:
    """
    Corrects atom map number by ensuring same mapping numbers are 
    present in both reactants & products
    """
    reactants_map_numbers = []
    for r in reactants:
        reactants_map_numbers += [a.GetAtomMapNum() for a in r.GetAtoms()]
    reactants_map_numbers = set(reactants_map_numbers)
    
    products_map_numbers = []
    for p in products:
        products_map_numbers += [a.GetAtomMapNum() for a in p.GetAtoms()]
    products_map_numbers = set(products_map_numbers)

    if reactants_map_numbers.issubset(products_map_numbers):
        for p in products:
            for atom in p.GetAtoms():
                if atom.GetAtomMapNum() not in reactants_map_numbers:
                    atom.SetAtomMapNum(0)

    elif products_map_numbers.issubset(reactants_map_numbers):
        for r in reactants:
            for atom in r.GetAtoms():
                if atom.GetAtomMapNum() not in products_map_numbers:
                    atom.SetAtomMapNum(0)


def get_autode_geometry_from_mapped_rdkit_conformer(mols: List[Chem.Mol]) -> List[Atom]:
    """
    Returns a list of autodE atoms for an embedded RDKit conformer using the mapped
    SMILES string.
    """
    min_dist = 7.0

    mols_atoms = []
    for idx, mol in enumerate(mols):
        mols_atoms.append([])
        for atom in mol.GetAtoms():
            position = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            mols_atoms[idx].append([
                atom.GetSymbol(),
                np.array([position.x, position.y, position.z]),
                atom.GetAtomMapNum(),
                idx,
            ])

    if len(mols) == 1:
        atoms = mols_atoms[0]
    elif len(mols) == 2:
        centroid_1 = np.mean([atom[1] for atom in mols_atoms[0]], axis=0)
        centroid_2 = np.mean([atom[1] for atom in mols_atoms[1]], axis=0)

        mols_atoms[0] = [[atom[0], atom[1] + min_dist * (
            (centroid_2 - centroid_1) / np.linalg.norm(centroid_2 - centroid_1)
        ), atom[2], atom[3]] for atom in mols_atoms[0]]

        atoms = mols_atoms[0] + mols_atoms[1]
    elif len(mols) == 3:
        centroid_1 = np.mean([atom[1] for atom in mols_atoms[0]], axis=0)
        centroid_2 = np.mean([atom[1] for atom in mols_atoms[1]], axis=0)
        centroid_3 = np.mean([atom[1] for atom in mols_atoms[2]], axis=0)

        mols_atoms[1] = [[atom[0], atom[1] + min_dist * (
            (centroid_2 - centroid_1) / np.linalg.norm(centroid_2 - centroid_1)
        ), atom[2], atom[3]] for atom in mols_atoms[0]]
        mols_atoms[2] = [[atom[0], atom[1] + min_dist * (
            (centroid_3 - centroid_1) / np.linalg.norm(centroid_3 - centroid_1)
        ), atom[2], atom[3]] for atom in mols_atoms[0]]

        atoms = mols_atoms[0] + mols_atoms[1] + mols_atoms[2]

    atoms = sorted(atoms, key=lambda x: x[2])

    species_complex_mapping = {}
    for idx, atom in enumerate(atoms):
        if atom[3] not in species_complex_mapping:
            species_complex_mapping[atom[3]] = [idx]
        else:
            species_complex_mapping[atom[3]].append(idx)

    indices = [atom[2] if atom[2] > 0 else f'{atom[2]}{atom[0]}' for atom in atoms]
    atoms = [Atom(atomic_symbol=atom[0], x=atom[1][0], y=atom[1][1], z=atom[1][2]) for atom in atoms]
    return atoms, indices, species_complex_mapping


def generate_mapped_reaction_complexes(
    reactant_smiles: List[str],
    product_smiles: List[str],
    solvent: str
) -> List[Complex]:
    """
    Generate complexes for a reaction using mapped smiles strings
    """
    # create mapped rxn smiles
    rxn_smiles = f"{'.'.join(reactant_smiles)}>>{'.'.join(product_smiles)}"
    mapped_rxn_smiles = map_rxn_smiles(rxn_smiles)

    # create RDKit molecules & correct atom map numbers
    reactants, products = mapped_rxn_smiles.split('>>')
    reactants, products = reactants.split('.'), products.split('.')
    reactants, products = [Chem.MolFromSmiles(r) for r in reactants], [Chem.MolFromSmiles(p) for p in products]
    reactants, products = [Chem.AddHs(r) for r in reactants], [Chem.AddHs(p) for p in products]
    correct_atom_map_numbers(reactants, products)

    # embed conformers
    method = AllChem.ETKDGv2()
    method.randomSeed = 0xF00D
    for r in reactants:
        AllChem.EmbedMultipleConfs(r, numConfs=1, params=method)
    for p in products:
        AllChem.EmbedMultipleConfs(p, numConfs=1, params=method)

    # generate reactant complex
    reactant_atoms, reactant_indices, species_complex_mapping = get_autode_geometry_from_mapped_rdkit_conformer(reactants)
    _rc_complex = generate_reaction_complex(reactant_smiles)
    rc_complex = ade.Molecule(
        name=str(time.time()),
        atoms=reactant_atoms,
        charge=_rc_complex.charge,
        mult=_rc_complex.mult,
        solvent_name=solvent,
    )
    rc_complex.species_complex_mapping = species_complex_mapping
    nx.set_node_attributes(rc_complex.graph, {i: v for i, v in enumerate(reactant_indices)}, 'atom_index')
    rc_complex.conformers = [Conformer(
        name=str(time.time()),
        atoms=reactant_atoms,
        charge=_rc_complex.charge,
        mult=_rc_complex.mult,
        solvent_name=solvent,
    )]

    # generate product complex
    product_atoms, product_indices, species_complex_mapping = get_autode_geometry_from_mapped_rdkit_conformer(products)
    _pc_complex = generate_reaction_complex(product_smiles)
    pc_complex = ade.Molecule(
        name=str(time.time()),
        atoms=product_atoms,
        charge=_pc_complex.charge,
        mult=_pc_complex.mult,  
        solvent_name=solvent,
    )
    pc_complex.species_complex_mapping = species_complex_mapping
    nx.set_node_attributes(pc_complex.graph, {i: v for i, v in enumerate(product_indices)}, 'atom_index')
    pc_complex.conformers = [Conformer(
        name=str(time.time()),
        atoms=product_atoms,
        charge=_pc_complex.charge,
        mult=_pc_complex.mult,  
        solvent_name=solvent,
    )]

    return rc_complex, pc_complex
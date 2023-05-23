# from rxnmapper import RXNMapper

# reaction_smiles = "[Pd]([P](C)(C)(C))[P](C)(C)(C).c1cccc(Br)c1>>[Pd](Br)(c1ccccc1)([P](C)(C)(C))[P](C)(C)(C)"

# rxn_mapper = RXNMapper()
# am_reaction_smiles = rxn_mapper.get_attention_guided_atom_maps([reaction_smiles])[0]['mapped_rxn']
# print(am_reaction_smiles)

# # am reaction smiles
# # [Br:6][c:7]1[cH:8][cH:9][cH:10][cH:11][cH:12]1.[CH3:1][P:2]([CH3:3])([CH3:4])[Pd:5][P:13]([CH3:14])([CH3:15])[CH3:16]>>[CH3:1][P:2]([CH3:3])([CH3:4])[Pd:5]([Br:6])([c:7]1[cH:8][cH:9][cH:10][cH:11][cH:12]1)[P:13]([CH3:14])([CH3:15])[CH3:16]

from typing import List
import time

from reaction_path_sampler.src.reaction_path.complexes import generate_reaction_complex
from reaction_path_sampler.src.visualization.plotly import plot_networkx_mol_graph

from autode.bond_rearrangement import get_bond_rearrangs, BondRearrangement
from autode.mol_graphs import reac_graph_to_prod_graph
from autode.atoms import Atom
import autode as ade
from autode.species.complex import Complex
from autode.conformers import Conformer

import networkx as nx
import numpy as np

# smiles = ['[Br:6][c:7]1[cH:8][cH:9][cH:10][cH:11][cH:12]1', '[CH3:1][P:2]([CH3:3])([CH3:4])[Pd:5][P:13]([CH3:14])([CH3:15])[CH3:16]']

from rxnmapper import RXNMapper
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

def map_rxn_smiles(rxn_smiles):
    rxn_mapper = RXNMapper()
    return rxn_mapper.get_attention_guided_atom_maps([rxn_smiles])[0]['mapped_rxn']

def correct_atom_map_numbers(reactants, products):
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

def get_mapped_rdkit_geometry(mols: List[Chem.Mol]) -> List[Atom]:
    mols_atoms = []
    for idx, mol in enumerate(mols):
        mols_atoms.append([])
        for atom in mol.GetAtoms():
            position = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            mols_atoms[idx].append([
                atom.GetSymbol(),
                np.array([position.x, position.y, position.z]),
                atom.GetAtomMapNum()
            ])

    if len(mols) == 1:
        atoms = mols_atoms[0]
    elif len(mols) == 2:
        min_dist = 3.0
        centroid_1 = np.mean([atom[1] for atom in mols_atoms[0]], axis=0)
        centroid_2 = np.mean([atom[1] for atom in mols_atoms[1]], axis=0)

        mols_atoms[0] = [[atom[0], atom[1] + min_dist * (
            (centroid_2 - centroid_1) / np.linalg.norm(centroid_2 - centroid_1)
        ), atom[2]] for atom in mols_atoms[0]]

        atoms = mols_atoms[0] + mols_atoms[1]
    elif len(mols) == 3:
        min_dist = 3.0
        centroid_1 = np.mean([atom[1] for atom in mols_atoms[0]], axis=0)
        centroid_2 = np.mean([atom[1] for atom in mols_atoms[1]], axis=0)
        centroid_3 = np.mean([atom[1] for atom in mols_atoms[2]], axis=0)

        mols_atoms[1] = [[atom[0], atom[1] + min_dist * (
            (centroid_2 - centroid_1) / np.linalg.norm(centroid_2 - centroid_1)
        ), atom[2]] for atom in mols_atoms[0]]
        mols_atoms[2] = [[atom[0], atom[1] + min_dist * (
            (centroid_3 - centroid_1) / np.linalg.norm(centroid_3 - centroid_1)
        ), atom[2]] for atom in mols_atoms[0]]

        atoms = mols_atoms[0] + mols_atoms[1] + mols_atoms[2]

    atoms = sorted(atoms, key=lambda x: x[2])
    indices = [atom[2] if atom[2] > 0 else f'{atom[2]}{atom[0]}' for atom in atoms]
    atoms = [Atom(atomic_symbol=atom[0], x=atom[1][0], y=atom[1][1], z=atom[1][2]) for atom in atoms]
    return atoms, indices

def initiate_from_mapped_smiles(
    reactant_smiles: List[str],
    product_smiles: List[str]
):
    rxn_smiles = f"{'.'.join(reactant_smiles)}>>{'.'.join(product_smiles)}"
    mapped_rxn_smiles = map_rxn_smiles(rxn_smiles)

    reactants, products = mapped_rxn_smiles.split('>>')
    reactants, products = reactants.split('.'), products.split('.')
    reactants, products = [Chem.MolFromSmiles(r) for r in reactants], [Chem.MolFromSmiles(p) for p in products]
    reactants, products = [Chem.AddHs(r) for r in reactants], [Chem.AddHs(p) for p in products]

    correct_atom_map_numbers(reactants, products)

    method = AllChem.ETKDGv2()
    method.randomSeed = 0xF00D
    for r in reactants:
        AllChem.EmbedMultipleConfs(r, numConfs=1, params=method)
    for p in products:
        AllChem.EmbedMultipleConfs(p, numConfs=1, params=method)

    reactant_atoms, reactant_indices = get_mapped_rdkit_geometry(reactants)
    product_atoms, product_indices = get_mapped_rdkit_geometry(products)
    
    _rc_complex = generate_reaction_complex(reactant_smiles)
    _pc_complex = generate_reaction_complex(product_smiles)

    rc_complex = ade.Molecule(
        name=str(time.time()),
        atoms=reactant_atoms,
        charge=_rc_complex.charge,
        mult=_rc_complex.mult,
        # solvent_name=_rc_complex.solvent_name,
    )
    nx.set_node_attributes(rc_complex.graph, {i: v for i, v in enumerate(reactant_indices)}, 'atom_index')
    rc_complex.conformers = [Conformer(
        name=str(time.time()),
        atoms=reactant_atoms,
        charge=_rc_complex.charge,
        mult=_rc_complex.mult,
        # solvent_name=_rc_complex.solvent_name,
    )]


    pc_complex = ade.Molecule(
        name=str(time.time()),
        atoms=product_atoms,
        charge=_pc_complex.charge,
        mult=_pc_complex.mult,  
        # solvent_name=_pc_complex.solvent_name,
    )
    nx.set_node_attributes(pc_complex.graph, {i: v for i, v in enumerate(product_indices)}, 'atom_index')
    pc_complex.conformers = [Conformer(
        name=str(time.time()),
        atoms=product_atoms,
        charge=_pc_complex.charge,
        mult=_pc_complex.mult,  
        # solvent_name=_pc_complex.solvent_name,
    )]

    return rc_complex, pc_complex

smiles1 = ["C1=C(N)C=CO1", "C=CF"]
smiles2 = ["C1=C(N)C(O2)CC(F)C12"]

rc_complex, pc_complex = initiate_from_mapped_smiles(smiles1, smiles2)

# plot_networkx_mol_graph(rc_complex.graph, rc_complex.coordinates)
# plot_networkx_mol_graph(pc_complex.graph, pc_complex.coordinates)

for idx, reaction_complexes in enumerate([
    [rc_complex, pc_complex],
    [pc_complex, rc_complex],
]):
    bond_rearrs = get_bond_rearrangs(reaction_complexes[1], reaction_complexes[0], name='test')
    if bond_rearrs is not None:
        for bond_rearr in bond_rearrs:
            graph1 = reaction_complexes[0].graph
            graph2 = reac_graph_to_prod_graph(reaction_complexes[1].graph, bond_rearr)
            
            mappings = []
            for isomorphism in nx.vf2pp_all_isomorphisms(
                graph1, 
                graph2, 
                node_label="atom_index"
            ):
                mappings.append(isomorphism)

            mappings = [dict(s) for s in set(frozenset(d.items()) for d in mappings)]
            print(mappings)
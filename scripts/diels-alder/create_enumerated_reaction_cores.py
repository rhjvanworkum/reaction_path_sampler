"""
The goal of this script is to generate a dataset of Diels Alder Reaction cores
by enumerating over possible substituents next to the 'active' bonds
"""
import random
from rdkit import Chem
from rdkit.Chem import AllChem

# rdkit utils
bondtypes = {
    1: Chem.BondType.SINGLE,
    2: Chem.BondType.DOUBLE,
    3: Chem.BondType.AROMATIC
}
elements = {
    'H':  1,
    'O':  8,
    'N':  7,
    'C':  6,
    'S':  16,
    'Cl': 17,
    'Br': 35,
    'F':  9,
    'I':  53
}
da_rxn_smarts = AllChem.ReactionFromSmarts(
    '[#6:1]=[#6:2].[#6:3]=[#6:4][#6:5]=[#6:6]>>[#6:1]1[#6:2][#6:3][#6:4]=[#6:5][#6:6]1'
)
def simulate_da_reaction(substrates):
    products = []
    products += da_rxn_smarts.RunReactants(substrates)
    substrates = [substrates[1], substrates[0]]
    products += da_rxn_smarts.RunReactants(substrates)
    
    products = [Chem.MolToSmiles(product[0]) for product in products]
    products = list(set(products))
    return [Chem.MolFromSmiles(product) for product in products]


# generation utils
substituents_appearance_dict = {
    'H':  8,
    'O':  4,
    'N':  4,
    'C':  4,
    'S':  4,
    'Cl': 1,
    'Br': 1,
    'F':  1,
    'I':  1
}
substituents_likelihood_dict = {
    'H':  32,
    'O':  8,
    'N':  4,
    'C':  16,
    'S':  4,
    'Cl': 1,
    'Br': 1,
    'F':  1,
    'I':  1
}
likelihood_list = []
for k, v in substituents_likelihood_dict.items():
    for _ in range(v):
        likelihood_list.append(k)


if __name__ == "__main__":
    file_name = "./data/diels_alder_reaction_cores.txt"
    n_rcs = 1000
    ring_prob = 0.1
    rcs = []

    while len(rcs) < n_rcs:
        # 1. generate a random sequence
        substituents = []
        while len(substituents) < 8:
            sub = likelihood_list[int(random.random() * len(likelihood_list))]
            if substituents.count(sub) < substituents_appearance_dict[sub]:
                substituents.append(sub)
        diene_subs = substituents[:4]
        dienophile_subs = substituents[4:]

        # make butadiene rings as well with some probability
        if diene_subs.count('C') >= 2:
            if random.random() < 0.1:
                diene_subs.remove('C')

        # 2. generate SMILES
        # - dienophile
        rdedmol = Chem.EditableMol(Chem.Mol())
        rdedmol.AddAtom(Chem.Atom(6))
        rdedmol.AddAtom(Chem.Atom(6))
        rdedmol.AddBond(0, 1, bondtypes[2])
        for sub in dienophile_subs[:2]:
            rdedmol.AddAtom(Chem.Atom(elements[sub]))
            rdedmol.AddBond(0, len(rdedmol.GetMol().GetAtoms()) - 1, bondtypes[1])
        for sub in dienophile_subs[2:]:
            rdedmol.AddAtom(Chem.Atom(elements[sub]))
            rdedmol.AddBond(1, len(rdedmol.GetMol().GetAtoms()) - 1, bondtypes[1])
        
        dienophile_mol = rdedmol.GetMol()
        try:
            Chem.SanitizeMol(dienophile_mol)
            dienophile_smiles = Chem.MolToSmiles(dienophile_mol, isomericSmiles=True)
        except:
            continue

        # - diene
        rdedmol = Chem.EditableMol(Chem.Mol())
        rdedmol.AddAtom(Chem.Atom(6))
        rdedmol.AddAtom(Chem.Atom(6))
        rdedmol.AddAtom(Chem.Atom(6))
        rdedmol.AddAtom(Chem.Atom(6))
        rdedmol.AddBond(0, 1, bondtypes[2])
        rdedmol.AddBond(1, 2, bondtypes[1])
        rdedmol.AddBond(2, 3, bondtypes[2])

        # ring
        if len(diene_subs) == 3:
            rdedmol.AddAtom(Chem.Atom(6))
            rdedmol.AddBond(0, len(rdedmol.GetMol().GetAtoms()) - 1, bondtypes[1])
            rdedmol.AddBond(3, len(rdedmol.GetMol().GetAtoms()) - 1, bondtypes[1])
            diene_subs.remove('C')

            for sub in diene_subs[:1]:
                rdedmol.AddAtom(Chem.Atom(elements[sub]))
                rdedmol.AddBond(0, len(rdedmol.GetMol().GetAtoms()) - 1, bondtypes[1])
            for sub in diene_subs[1:]:
                rdedmol.AddAtom(Chem.Atom(elements[sub]))
                rdedmol.AddBond(3, len(rdedmol.GetMol().GetAtoms()) - 1, bondtypes[1])            
        # linear
        else:
            for sub in diene_subs[:2]:
                rdedmol.AddAtom(Chem.Atom(elements[sub]))
                rdedmol.AddBond(0, len(rdedmol.GetMol().GetAtoms()) - 1, bondtypes[1])
            for sub in diene_subs[2:]:
                rdedmol.AddAtom(Chem.Atom(elements[sub]))
                rdedmol.AddBond(3, len(rdedmol.GetMol().GetAtoms()) - 1, bondtypes[1])
        
        diene_mol = rdedmol.GetMol()
        try:
            Chem.SanitizeMol(diene_mol)
            diene_smiles = Chem.MolToSmiles(diene_mol, isomericSmiles=True)
        except:
            continue

        products = simulate_da_reaction([dienophile_mol, diene_mol])
        if len(products) == 0:
            continue

        # check if combination not already in current reaction cores
        for (diene_smi, dienophile_smi) in rcs:
            if diene_smi == diene_smiles and dienophile_smi == dienophile_smiles:
                continue

        rcs.append((diene_smiles, dienophile_smiles))


    # now make complete reaction SMILES
    reaction_smiles_list = []
    for (diene_smi, dienophile_smi) in rcs:
        dienophile_mol = Chem.MolFromSmiles(diene_smi)
        diene_mol = Chem.MolFromSmiles(dienophile_smi)
        products = simulate_da_reaction([dienophile_mol, diene_mol])
        reaction_smiles = f'{diene_smi}.{dienophile_smi}>>{Chem.MolToSmiles(products[0], isomericSmiles=True)}'
        reaction_smiles_list.append(reaction_smiles)

    with open(file_name, 'w') as f:
        f.writelines("\n".join(reaction_smiles_list))
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd

"""
write here the code for generating the intermediates
-> how do we make sure that we turn the right acyl group in oxonium ion?

"""

acyl_smiles_pattern = Chem.MolFromSmarts('[C:1](=[O:2])[Cl]')

rxn_smarts = AllChem.ReactionFromSmarts(
    "[C:1](=[O:2])[Cl:3]>>[C:1]#[O+:3]"
)
def create_oxonium_species(species):
    mol = rxn_smarts.RunReactants([species])[0][0]
    return mol

rxn_smarts_2 = AllChem.ReactionFromSmarts(
    "[C:1]([C:3](=[O:4]))=[C:2]>>[C:1]([C:3](=[O:4]))[CH+:2]"
)
def create_intermediate(species):
    Chem.Kekulize(species, clearAromaticFlags=True)
    mol = rxn_smarts_2.RunReactants([species])[0][0]
    return mol

if __name__ == "__main__":
    df = pd.read_csv('./data/fca/fca_dataset.csv')

    reaction_smiles_list = []
    for substrates, products in zip(df['substrates'].values, df['products'].values):
        substrate1, substrate2 = substrates.split('.')
        substrate1, substrate2 = Chem.MolFromSmiles(substrate1), Chem.MolFromSmiles(substrate2)
        if len(substrate1.GetSubstructMatches(acyl_smiles_pattern)) >= 1:
            substrate1 = create_oxonium_species(substrate1)
        elif len(substrate2.GetSubstructMatches(acyl_smiles_pattern)) >= 1:
            substrate2 = create_oxonium_species(substrate2)
        reactants_list = [Chem.MolToSmiles(substrate1), Chem.MolToSmiles(substrate2)]

        try:
            products_list = [Chem.MolToSmiles(create_intermediate(Chem.MolFromSmiles(products)))]
        except:
            products_list = ['CC']
       
        reaction_smiles = f"{'.'.join(reactants_list)}>>{'.'.join(products_list)}"
        reaction_smiles_list.append(reaction_smiles)
    
    with open('./data/fca/fca_dataset.txt', 'w') as f:
        f.writelines("\n".join(reaction_smiles_list))
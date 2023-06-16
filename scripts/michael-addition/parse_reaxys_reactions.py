from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


PRODUCT_RXN = AllChem.ReactionFromSmarts(
    "[S,s:5][#6:1][#6:2][#6:3]=[O:4]>>[SH+,sH+:5][#6:1][#6:2]=[#6:3]-[O-:4]"
)

def simulate_reaction(substrates, reaction_smarts, fix_oxygen: bool = False):
    products = []
    products += reaction_smarts.RunReactants(substrates)
    if len(substrates) == 2:
        substrates = [substrates[1], substrates[0]]
        products += reaction_smarts.RunReactants(substrates)
    products = [Chem.MolToSmiles(product[0]) for product in products]
    products = list(set(products))

    if fix_oxygen:
        products = [product.replace('=[O-]', '=[O]') for product in products]

    products = [Chem.MolFromSmiles(product) for product in products]
    return list(filter(lambda x: x is not None, products))


if __name__ == "__main__":
    df = pd.read_csv('./data/michael_addition/ma_dataset_thio.csv')

    reaction_smiles_list = []
    for substrates, products in zip(df['substrates'].values, df['products'].values):
        reactants_list = substrates.split('.')

        product = [Chem.MolFromSmiles(products)]
        intermediate = simulate_reaction(product, PRODUCT_RXN, fix_oxygen=True)[0]
        products_list = Chem.MolToSmiles(intermediate).split('.')
       
        reaction_smiles = f"{'.'.join(reactants_list)}>>{'.'.join(products_list)}"
        reaction_smiles_list.append(reaction_smiles)
    
    with open('./data/michael_addition/ma_dataset_thio.txt', 'w') as f:
        f.writelines("\n".join(reaction_smiles_list))
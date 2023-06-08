from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


intermediate_rxn_smarts = AllChem.ReactionFromSmarts(
    '[C:1](=[O:2])[OH].[N,n:3]>>n1nn(O[C:1](-[O-:2])[N+,n+:3])c2ncccc12'
)

product_rxn_smarts = AllChem.ReactionFromSmarts(
    'n1nn(O[C:1](-[O-:2])[N,n:3])c2ncccc12>>[C:1](=[O:2])[N,n:3]'
)

def get_intermediate_smiles(
    substrate_smiles: str,
    product_smiles: str
) -> str:
    can_product_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(product_smiles), isomericSmiles=False)

    reactants = [Chem.MolFromSmiles(smi) for smi in substrate_smiles.split('.')]
    
    products = []
    products += intermediate_rxn_smarts.RunReactants(reactants)
    reactants = [reactants[1], reactants[0]]
    products += intermediate_rxn_smarts.RunReactants(reactants)
    potential_intermediates = [p[0] for p in products]
    
    for intermediate in potential_intermediates:
        int_product = product_rxn_smarts.RunReactants((intermediate,))[0][0]
        # TODO: change this, this hack makes it now work for nitro or diazoniuim groups for example
        for atom in int_product.GetAtoms():
            if atom.GetAtomicNum() == 8:
                atom.SetFormalCharge(0)
                
            if atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1:
                atom.SetNumExplicitHs(0)
                atom.SetFormalCharge(0)
                
        try:
            Chem.SanitizeMol(int_product)
        except:
            continue
            
        int_product_smiles = Chem.MolToSmiles(int_product)  
        can_int_product_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(int_product_smiles), isomericSmiles=False)

        if can_int_product_smiles == can_product_smiles:
            return Chem.MolToSmiles(intermediate)


if __name__ == "__main__":
    df = pd.read_csv('./data/ac_dataset_dcm_small.csv')

    reaction_smiles_list = []
    for substrates, products in zip(df['substrates'].values, df['products'].values):
        # 1. create HATU tertiary intermediate
        reactants_list = [get_intermediate_smiles(substrates, products)]
        # 2. create product + HATU alcohol
        products_list = [products, "n1nn(O)c2ncccc12"]
        # 3. write reaction
        reaction_smiles = f"{'.'.join(reactants_list)}>>{'.'.join(products_list)}"
        reaction_smiles_list.append(reaction_smiles)
    
    with open('./data/ac_dataset_dcm_small.txt', 'w') as f:
        f.writelines("\n".join(reaction_smiles_list))
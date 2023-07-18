from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


tert_intermediate_rxn_smarts = AllChem.ReactionFromSmarts(
    '[C:1](=[O:2])[OH].[N,n:3]>>n1nn(O[C:1](-[O-:2])[N+,n+:3])c2ncccc12'
)

activated_acid_rxn_smarts = AllChem.ReactionFromSmarts(
    '[C:1](=[O:2])[OH]>>'
)

def get_tert_intermediate_smiles(
    substrate_smiles: str
) -> str:
    reactants = [Chem.MolFromSmiles(smi) for smi in substrate_smiles.split('.')]
    
    products = []
    products += tert_intermediate_rxn_smarts.RunReactants(reactants)
    reactants = [reactants[1], reactants[0]]
    products += tert_intermediate_rxn_smarts.RunReactants(reactants)
    products = [p[0] for p in products]
    
    for product in products:
        try:
            Chem.SanitizeMol(product)
            return Chem.MolToSmiles(product)
        except:
            continue

def get_activated_acid_smiles(
    acid_smiles: str
) -> str:
    CARBOXYLIC_ACID_SMILES_PATTERN = "C(=O)O"

    activated_acid = Chem.ReplaceSubstructs(
        Chem.MolFromSmiles(acid_smiles),
        Chem.MolFromSmiles(CARBOXYLIC_ACID_SMILES_PATTERN),
        Chem.MolFromSmiles("C(=O)On1nnc2cccnc21"),
        replaceAll=False,
    )[0]
    
    Chem.SanitizeMol(activated_acid)
    return Chem.MolToSmiles(activated_acid)

if __name__ == "__main__":
    df = pd.read_csv('./data/HATU_ac/ac_hatu_test.csv')

    reaction_smiles_list = []
    for substrates, products in zip(df['substrates'].values, df['products'].values):
        # 1. amine + HATU-activated acid
        reactants_list = substrates.split('.')
        reactants_list = [reactants_list[0], get_activated_acid_smiles(reactants_list[1])]
        
        # 2. tertiary intermediate
        products_list = [get_tert_intermediate_smiles(substrates)]
        
        # 3. write reaction
        reaction_smiles = f"{'.'.join(reactants_list)}>>{'.'.join(products_list)}"
        reaction_smiles_list.append(reaction_smiles)
    
    with open('./data/HATU_ac/ac_hatu_test.txt', 'w') as f:
        f.writelines("\n".join(reaction_smiles_list))
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


rxn_smarts = AllChem.ReactionFromSmarts(
    "[NH,nH,NH2,SH,S-:1].[C,c:2]([F,Cl,Br,I:3])=[C,c:4]>>[C,c:2]([N+,n+,S:1])=[C,c:4].[F-,Cl-,Br-,I-:3]"
)
rxn_smarts_2 = AllChem.ReactionFromSmarts(
    "[NH,nH,NH2,SH,S-:1].[C,c:2]([F,Cl,Br,I:3])=[N,n:4]>>[C,c:2]([N+,n+,S:1])=[N,n:4].[F-,Cl-,Br-,I-:3]"
)

def simulate_reaction(substrates, smarts):
    products = []
    products += smarts.RunReactants(substrates)
    substrates = [substrates[1], substrates[0]]
    products += smarts.RunReactants(substrates)
    
    products = [Chem.CombineMols(*p) for p in products]
    products = [Chem.MolToSmiles(product) for product in products]
    products = list(set(products))
    products = [Chem.MolFromSmiles(product) for product in products]
    return list(filter(lambda x: x is not None, products))


if __name__ == '__main__':
    df = pd.read_csv('./data/snar/snar.csv')

    sim_reaction_smiles = []

    for idx, row in df.iterrows():
        substrates = row['substrates'].split('.')
        substrates = [Chem.MolFromSmiles(sub) for sub in substrates]
        substrates = [Chem.AddHs(sub) for sub in substrates]
        for sub in substrates:
            Chem.Kekulize(sub, clearAromaticFlags=True)
        
        products = []
        products += simulate_reaction(substrates, rxn_smarts)
        products += simulate_reaction(substrates, rxn_smarts_2)

        og_product = Chem.MolFromSmiles(row['products'].split('.')[0])
        matches = [len(p.GetSubstructMatch(og_product)) for p in products]

        i = 0
        for match in matches:
            if match != 0:
                i += 1

        if i == 1:
            for product in products:
                if len(product.GetSubstructMatch(og_product)) != 0:
                    product_smiles = Chem.MolToSmiles(product)
                    sim_reaction_smiles.append(f"{row['substrates']}>>{product_smiles}")
        else:
            sim_reaction_smiles.append(None)

    df['sim_reaction_smiles'] = sim_reaction_smiles
    
    df = df.dropna()
    ridxs = [r for r in df['reaction_idx'].unique() if len(df[df['reaction_idx'] == r]) == 2]
    new_df = df[df['reaction_idx'].isin(ridxs)]
    new_df.to_csv('./data/snar/snar_simulated.csv', index=False)

    with open('./data/snar/snar_simulated.txt', 'w') as f:
        f.writelines("\n".join(new_df['sim_reaction_smiles'].values))
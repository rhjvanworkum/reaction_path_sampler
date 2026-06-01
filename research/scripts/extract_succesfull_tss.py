import os
import shutil
import pandas as pd

if __name__ == "__main__":
    path = "./scratch/da_tss_test/"
    name = "da_tss_test"

    df = pd.read_csv('data/test_da_reactions_reaction_core.csv')
    reaction_smiles_list = df['reaction_smiles'].values    

    list = []
    for root, dirs, files in os.walk(path):
        if len(root.split('/')) > 3 and root.split('/')[-2] == name:
            if os.path.exists(os.path.join(root, 'reaction.xyz')):
                list.append(int(root.split('/')[-1]))
    list = sorted(list)

    new_df = df.iloc[list]
    uids_to_keep = []
    for reaction_idx in new_df['reaction_idx'].unique():
        selection = new_df[new_df['reaction_idx'] == reaction_idx]

        if len(selection) >= 2 and (1 in selection['labels'].values):
            for uid in new_df[new_df['reaction_idx'] == reaction_idx]['uid'].values:
                uids_to_keep.append(uid)
    
    
    new_df = new_df[new_df['uid'].isin(uids_to_keep)]
    new_df.to_csv('./data/test_da_reactions_tss.csv')
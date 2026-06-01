import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    barriers = []
    indices = []

    output_folder = './scratch/snar_simulated_1/'
    name = "snar_simulated_1"
    for root, dirs, files in os.walk(output_folder):
        if len(root.split('/')) > 3 and root.split('/')[-2] == name:
            try:
                idx = int(root.split('/')[-1])

                if os.path.exists(os.path.join(root, 'barrier.txt')):
                    with open(os.path.join(root, 'barrier.txt'), 'r') as f:
                        barrier = float(f.readlines()[0])
                        barriers.append(barrier)
                else:
                    barriers.append(None)

                indices.append(idx)
            except:
                continue

    df = pd.read_csv('./data/snar/snar_simulated.csv')
    df['barrier'] = None
    for barrier, index in zip(barriers, indices):
        df.loc[index, 'barrier'] = barrier

    new_df = df[df['barrier'].notna()]
    new_df['label'] = new_df['barrier']

    print(len(new_df), len(new_df['reaction_idx'].unique()))
    new_df.to_csv('./data/snar/snar_rps.csv')
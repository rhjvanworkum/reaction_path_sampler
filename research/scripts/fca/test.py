import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    barriers = []
    indices = []

    output_folder = './scratch/fca_test_small_mult/'
    name = "fca_test_small_mult"
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

    df = pd.read_csv('./data/fca/fca_dataset_small.csv')
    df['barrier'] = None
    for barrier, index in zip(barriers, indices):
        df.loc[index, 'barrier'] = barrier

    keep_r_idxs = []
    for r_idx in df['reaction_idx'].unique():
        if not df[df['reaction_idx'] == r_idx]['barrier'].isna().any():
            keep_r_idxs.append(r_idx)

    new_df = df[df['reaction_idx'].isin(keep_r_idxs)]
    
    print(len(new_df), len(new_df['reaction_idx'].unique()))

    # compute labels
    labels = []
    for _, row in new_df.iterrows():
        barrier = row['barrier']
        other_barriers = new_df[new_df['substrates'] == row['substrates']]['barrier']

        if np.isnan(barrier) or True in [np.isnan(val) for val in other_barriers.values]:
            labels.append(np.nan)
        else:
            label = int((barrier <= other_barriers).all())
            labels.append(label)
    new_df['test_label'] = labels

    # compute scores
    targets, preds = [], []
    right = 0
    for idx in new_df['reaction_idx'].unique():
        target = new_df[new_df['reaction_idx'] == idx]['label']
        pred = new_df[new_df['reaction_idx'] == idx]['test_label']

        if len(pred) > 0 and len(target) > 0 and 1 in pred.values:

            if (target.values == pred.values).all():
                right += 1
                
            for val in target:
                targets.append(val)
            for val in pred:
                preds.append(val)

    print(roc_auc_score(targets, preds) * 100, 'AUROC')
    print(right/ len(new_df['reaction_idx'].unique()) * 100, "%", "accuracy")


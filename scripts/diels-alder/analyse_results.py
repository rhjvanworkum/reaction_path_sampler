from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import os
import tempfile
import yaml
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    dataset_path = "./data/DA_regio_no_solvent_success.csv"
    dataset = pd.read_csv(dataset_path)
    base_dir = "./scratch/DA_test_no_solvent/"

    activation_energies = []
    for i in dataset['uid'].values:
        results_dir = os.path.join(base_dir, f'{i}')
        try:
            with open(os.path.join(results_dir, 'dft_barrier.txt'), 'r') as f:
                barrier = float(f.readlines()[0])
        except:
            barrier = np.nan
        activation_energies.append(barrier)

    activation_energies = np.array(activation_energies)
    dataset['barrier'] = activation_energies

    # compute labels
    labels = []
    for _, row in dataset.iterrows():
        barrier = row['barrier']
        other_barriers = dataset[dataset['substrates'] == row['substrates']]['barrier']

        if np.isnan(barrier) or True in [np.isnan(val) for val in other_barriers.values]:
            labels.append(np.nan)
        else:
            label = int((barrier <= other_barriers).all())
            labels.append(label)
    dataset['dft_ensemble_labels'] = labels

    # save dataset
    dataset.to_csv(dataset_path)
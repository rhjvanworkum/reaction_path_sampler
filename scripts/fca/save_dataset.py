import os
import numpy as np
import pandas as pd


if __name__ == "__main__":
    output_folder = './scratch/fca_test_methanol/'
    name = "fca_test_methanol"
    base_settings_file = 'systems/rps.yaml'
    file_path = 'data/fca/fca_dataset.txt'

    reaction_smiles_list = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            reaction_smiles_list.append(line.strip())

    idx_list = []
    for root, dirs, files in os.walk(output_folder):
        if len(root.split('/')) > 3 and root.split('/')[-2] == name:
            if os.path.exists(os.path.join(root, 'barrier.txt')):
                idx_list.append(int(root.split('/')[-1]))
    idx_list = sorted(idx_list)
    successfull_reaction_smiles = [reaction_smiles_list[i] for i in idx_list]


    barriers = []
    for i in idx_list:
        results_dir = os.path.join(output_folder, f'{i}')
        with open(os.path.join(results_dir, 'barrier.txt'), 'r') as f:
            barrier = float(f.readlines()[0])
            barriers.append(barrier)

    
    substrates = [
        smi.split('>>')[0] for smi in successfull_reaction_smiles
    ]
    products = [
        smi.split('>>')[1] for smi in successfull_reaction_smiles
    ]

    df = pd.DataFrame({
        'reaction_idx': np.arange(len(substrates)),
        'uid': np.arange(len(substrates)),
        'substrates': substrates,
        'products': products,
        'reaction_smiles': successfull_reaction_smiles,
        'label': barriers,
        'simulation_idx': np.zeros(len(substrates))
    })

    df.to_csv('./data/fca/fca_test_result.csv')
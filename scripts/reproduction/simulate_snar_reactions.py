"""
Script to compute a set of DA cycloadditions
"""
import pandas as pd
import yaml
import os


if __name__ == "__main__":
    output_folder = './scratch/snar_reproduction_test/'
    base_settings_file = 'systems/rps.yaml'
    file_path = 'data/snar_reproduction_test.csv'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(base_settings_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    reaction_smiles_list = pd.read_csv(file_path)['sim_reaction_smiles'].values
    for idx, reaction_smiles in enumerate(reaction_smiles_list):
        reaction_smiles = reaction_smiles_list[idx]

        output_dir = os.path.join(output_folder, f'{idx}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        settings['output_dir'] = output_dir
        reactants, products = reaction_smiles.split('>>')
        settings['reactant_smiles'] = reactants.split('.')
        settings['product_smiles'] = products.split('.')
        settings['solvent'] = "Methanol"
        settings['use_cregen_pruning'] = False
        settings['n_processes'] = 8

        settings['graph_pruning_threshold'] = 2
        settings['irc_end_graph_threshold'] = 4

        # yaml file
        yaml_file_name = f'{idx}.yaml'
        with open(os.path.join(output_folder, yaml_file_name), 'w') as f:
            yaml.dump(settings, f)

        # bash file
        bash_file_name = f'{idx}.sh'
        with open(os.path.join(output_folder, bash_file_name), 'w') as f:
            f.writelines([
                '#!/bin/bash \n',
                'source env.sh \n',
                f'python -u search_rxn_path.py {os.path.join(output_folder, yaml_file_name)}'
            ])

        # execute
        os.system(f'sbatch --cpus-per-task=16 --time=2:00:00 --qos=cpus100 --output={output_folder}{idx}/job_%A.out {os.path.join(output_folder, bash_file_name)}')


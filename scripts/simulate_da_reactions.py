"""
Script to compute a set of DA cycloadditions
"""

import yaml
import os


if __name__ == "__main__":
    output_folder = './scratch/diels_alder_reaction_cores/'
    base_settings_file = 'systems/rps.yaml'
    file_path = 'data/diels_alder_reaction_cores.txt'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(base_settings_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    with open(file_path, 'r') as f:
        reaction_smiles_list = [line.replace('\n', '') for line in f.readlines()]

    for idx, reaction_smiles in enumerate(reaction_smiles_list):
        output_dir = os.path.join(output_folder, f'{idx}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        settings['output_dir'] = output_dir
        reactants, products = reaction_smiles.split('>>')
        settings['reactant_smiles'] = reactants.split('.')
        settings['product_smiles'] = products.split('.')
        settings['n_processes'] = 10

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
        os.system(f'sbatch --cpus-per-task=20 --time=01:00:00 --qos=cpus100 --output={output_folder}{idx}/job_%A.out {os.path.join(output_folder, bash_file_name)}')


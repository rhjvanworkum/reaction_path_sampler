import pandas as pd
import os
import yaml

if __name__ == "__main__":
    output_folder = './scratch/da_tss_test2/'
    base_settings_file = 'systems/ts_opt.yaml'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(base_settings_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    df = pd.read_csv('data/test_da_reactions.csv')
    reaction_smiles_list = df['reaction_smiles'].values

    for idx, reaction_smiles in enumerate(reaction_smiles_list):
        output_dir = os.path.join(output_folder, f'{idx}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        settings['output_dir'] = output_dir
        settings['reaction_smiles'] = reaction_smiles

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
                f'python -u search_rxn_path_from_template.py {os.path.join(output_folder, yaml_file_name)}'
            ])

        # execute
        os.system(f'sbatch --cpus-per-task=2 --time=01:00:00 --qos=cpus100 --output={output_folder}{idx}/job_%A.out {os.path.join(output_folder, bash_file_name)}')

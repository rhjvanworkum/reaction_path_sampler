"""
Script to compute a set of DA cycloadditions
"""

import yaml
import os

# TODO: do more in parallel
# TODO: reduce amount of conformers that are optimized

if __name__ == "__main__":
    output_folder = './scratch/DA_test_solvent/'
    base_settings_file = 'systems/rps.yaml'
    file_path = 'data/DA_test_solvent.txt'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(base_settings_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    reaction_smiles_list, xtb_solvent_list = [], []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            args = line.split(' ')
            if len(args) == 3:
                reaction_smiles, xtb_solvent, orca_solvent = args
            elif len(args) == 4:
                reaction_smiles, xtb_solvent, orca_solvent, orca_solvent_2 = args
                orca_solvent += f' {orca_solvent_2}'
            reaction_smiles_list.append(reaction_smiles)
            xtb_solvent_list.append(xtb_solvent)

    for idx, (reaction_smiles, solvent) in enumerate(zip(reaction_smiles_list, xtb_solvent_list)):
        output_dir = os.path.join(output_folder, f'{idx}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        settings['output_dir'] = output_dir
        reactants, products = reaction_smiles.split('>>')
        settings['reactant_smiles'] = reactants.split('.')
        settings['product_smiles'] = products.split('.')
        settings['solvent'] = solvent
        settings['n_processes'] = 4

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
                f'python -u search_rxn_path_2.py {os.path.join(output_folder, yaml_file_name)}'
            ])

        # execute
        os.system(f'sbatch --cpus-per-task=8 --time=01:00:00 --qos=cpus100 --output={output_folder}{idx}/job_%A.out {os.path.join(output_folder, bash_file_name)}')


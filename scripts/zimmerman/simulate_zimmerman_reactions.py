import yaml
import os
from rdkit import Chem


if __name__ == "__main__":
    output_folder = './scratch/zimmerman_dataset/'
    base_settings_file = 'systems/ac1.yaml'
    file_path = 'data/zimmerman_reactions_am.txt'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(base_settings_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    reaction_smiles_list = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            line = line.split(' ')[1]
            reaction_smiles_list.append(line.strip())

    for idx, reaction_smiles in enumerate([reaction_smiles_list[38]]):
        output_dir = os.path.join(output_folder, f'{idx}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # reactants, products = reaction_smiles.split('>>')
        # reactants, products = reactants.split('.'), products.split('.')
        # reactants, products = [Chem.MolFromSmiles(smi) for smi in reactants], [Chem.MolFromSmiles(smi) for smi in products]
        # reactants, products = [Chem.MolToSmiles(mol) for mol in reactants], [Chem.MolToSmiles(mol) for mol in products]
        # reaction_smiles = '.'.join(reactants) + '>>' + '.'.join(products)
        # print(reaction_smiles)

        settings['output_dir'] = output_dir
        reactants, products = reaction_smiles.split('>>')
        settings['reactant_smiles'] = reactants.split('.')
        settings['product_smiles'] = products.split('.')
        settings['solvent'] = ""
        settings['n_processes'] = 8

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
        os.system(f'sbatch --cpus-per-task=16 --time=01:00:00 --qos=cpus150 --output={output_folder}{idx}/job_%A.out {os.path.join(output_folder, bash_file_name)}')
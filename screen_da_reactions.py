"""
Script to compute a set of DA cycloadditions
"""

import yaml
import os

butadiene_reactions = [
    "C=CC=C.C=C>>C1=CCCCC1",
    "C=CC([N+](=O)[O-])=C.C=C>>C1=C([N+](=O)[O-])CCCC1",
    "C=CC(O)=C.C=CCl>>C1=C(O)CCC(Cl)C1",
    "C=CC(CCO)=C.C=C(CCN)>>C1=C(CCO)CCC(CCN)C1"
]
    
oxodiene_reactions = [
    "O=CC=O.C=C>>C1=COCCO1",
    "O=CC([N+](=O)[O-])=O.C=C>>C1=C([N+](=O)[O-])OCCO1",
    "O=CC(O)=O.C=CCl>>C1=C(O)OCC(Cl)O1",
    "O=CC(CCO)=O.C=C(CCN)>>C1=C(CCO)OCC(CCN)O1"
]
    
furan_reactions = [
    "C1=CC=CO1.C=C>>C12CCC(O1)C=C2",
    "C1=C([N+](=O)[O-])C=CO1.C=C>>C12CCC(O1)C([N+](=O)[O-])=C2",
    "C1=C(O)C=CO1.C=CCl>>C12C(Cl)CC(O1)C(O)=C2",
    "C1=CC(CCO)=CO1.C=C(CCN)>>C12CC(CCN)C(O1)C(CCO)=C2"
]

reactions = {
    'butadiene': butadiene_reactions,
    'oxodiene': oxodiene_reactions,
    'furan': furan_reactions
}
names = ["original", "EWG", "EDG", "large"]

if __name__ == "__main__":
    output_folder = './scratch/da_screening_2/'
    base_settings_file = 'systems/da_large.yaml'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(base_settings_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    for key, reaction_smiles_list in reactions.items():
        for name, reaction_smiles in zip(names, reaction_smiles_list):
            
            if not os.path.exists(f'./scratch/da_screening/{key}_{name}'):
                os.makedirs(f'./scratch/da_screening/{key}_{name}')

            settings['output_dir'] = f'./scratch/da_screening/{key}_{name}'
            reactants, products = reaction_smiles.split('>>')
            settings['reactant_smiles'] = reactants.split('.')
            settings['product_smiles'] = products.split('.')
            settings['n_processes'] = 16

            # yaml file
            yaml_file_name = f'{key}_{name}.yaml'
            with open(os.path.join(output_folder, yaml_file_name), 'w') as f:
                yaml.dump(settings, f)

            # bash file
            bash_file_name = f'{key}_{name}.sh'
            with open(os.path.join(output_folder, bash_file_name), 'w') as f:
                f.writelines([
                    '#!/bin/bash \n',
                    'source env.sh \n',
                    f'python main.py {yaml_file_name}'
                ])

            # execute
            os.system(f'sbatch --cpus-per-task=20 --qos=cpus100 --output={output_folder}{key}_{name}/job_%A.out {os.path.join(output_folder, bash_file_name)}')


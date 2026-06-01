"""
Script to compute a set of DA cycloadditions
"""

import yaml
import os

success_list = [
    0, 1, 12, 13, 14, 15, 16, 17, 48, 49, 54, 55, 60, 61, 62, 63, 66, 67,
    76, 77, 80, 81, 84, 85, 92, 93, 94, 95, 104, 105, 106, 107, 118, 119, 132, 133,
    136, 137, 140, 141, 146, 147, 156, 157, 158, 159, 164, 165, 168, 169, 170, 171, 180, 181,
    196, 197, 198, 199, 200, 201, 208, 209, 226, 227, 230, 231, 232, 233, 238, 239, 242, 243,
    244, 245, 248, 249, 254, 255, 258, 259, 260, 261, 268, 269, 280, 281, 284, 285, 286, 287,
    294, 295, 308, 309, 310, 311, 318, 319, 324, 325, 328, 329, 338, 339, 340, 341, 342, 343,
    366, 367, 368, 369, 372, 373, 376, 377, 388, 389, 390, 391, 392, 393, 400, 401, 410, 411,
    420, 421, 424, 425, 426, 427, 434, 435, 440, 441, 448, 449, 450, 451, 452, 453, 454, 455,
    456, 457, 474, 475, 476, 477, 484, 485, 486, 487, 488, 489, 496, 497, 500, 501, 502, 503,
    508, 509, 510, 511, 516, 517, 518, 519, 520, 521, 526, 527, 528, 529, 530, 531, 534, 535,
    548, 549, 562, 563, 570, 571, 572, 573, 578, 579, 586, 587, 588, 589, 590, 591, 592, 593,
    598, 599, 600, 601, 602, 603, 604, 605, 610, 611, 620, 621, 622, 623, 626, 627, 628, 629,
    634, 635, 640, 641, 672, 673, 682, 683, 684, 685, 696, 697, 698, 699, 700, 701, 702, 703,
    786, 787, 788, 789, 792, 793
]

if __name__ == "__main__":
    output_folder = './scratch/snar_simulated_2/'
    base_settings_file = 'systems/rps.yaml'
    file_path = 'data/snar/snar_simulated.txt'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(base_settings_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    reaction_smiles_list = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            reaction_smiles_list.append(line.strip())

    # for idx, reaction_smiles in enumerate(reaction_smiles_list):
    for idx in success_list:
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
        settings['barrier_method'] = "orca_B3LYP"
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
        os.system(f'sbatch --cpus-per-task=16 --time=12:00:00 --qos=cpus150 --output={output_folder}{idx}/job_%A.out {os.path.join(output_folder, bash_file_name)}')


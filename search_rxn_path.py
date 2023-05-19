import argparse
import yaml
import os

from src.reaction_path_sampler import ReactionPathSampler

def search_rxn_path():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "settings_file_path",
        help="Path to file containing the settings",
        type=str
    )
    args = parser.parse_args()

    # open yaml settings
    with open(args.settings_file_path, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    reaction_path_sampler = ReactionPathSampler(settings)
    reaction_path_sampler.generate_reaction_complexes()
    reaction_path_sampler.map_reaction_complexes()
    rc_conformers, pc_conformers = reaction_path_sampler.sample_reaction_complex_conformers()
    conformer_pairs = reaction_path_sampler.select_promising_reactant_product_pairs(rc_conformers, pc_conformers)

    for idx, conformer_pair in enumerate(conformer_pairs):
        print(f'Working on Reactant-Product Complex pair {idx}')
        
        output_dir = os.path.join(settings['output_dir'], f'{idx}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        success = reaction_path_sampler.find_reaction_path(
            rc_conformer=conformer_pair[0],
            pc_conformer=conformer_pair[1],
            output_dir=output_dir,
            final_dir=settings['output_dir'],
        )

        if success:
            return

if __name__ == "__main__":    
    search_rxn_path()
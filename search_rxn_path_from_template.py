import argparse
import logging
import yaml
import os
import sys

from autode.conformers.conformers import atoms_from_rdkit_mol
from autode.conformers.conformer import Conformer

from reaction_path_sampler.src.template_sampler import TemplateSampler
from reaction_path_sampler.src.ts_template import get_ts_templates

# TODO: doesnt work
def set_log_level():
    log_level_string = os.environ["RPS_LOG_LEVEL"]
    if log_level_string == "DEBUG":
        log_level = logging.DEBUG
    if log_level_string == "WARNING":
        log_level = logging.WARNING
    if log_level_string == "INFO":
        log_level = logging.INFO
    if log_level_string == "ERROR":
        log_level = logging.ERROR
    if log_level_string == "CRITICAL":
        log_level = logging.CRITICAL

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        # handlers=[
        #     logging.FileHandler("debug.log"),
        #     logging.StreamHandler()
        # ]
    )

def search_rxn_path_from_template():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "settings_file_path",
        help="Path to file containing the settings",
        type=str
    )
    args = parser.parse_args()

    set_log_level()

    # open yaml settings
    with open(args.settings_file_path, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    template_reaction_sampler = TemplateSampler(settings)
    template_reaction_sampler.generate_reaction_complexes()

    ts_templates = get_ts_templates(folder_path=settings['ts_template_dir'])
    template_reaction_sampler.select_and_load_ts_template(ts_templates=ts_templates)
    n_guesses = template_reaction_sampler.embed_ts_guesses()    

    for idx in range(n_guesses):
        print(f'Working on TS guess geometry {idx}')
        
        output_dir = os.path.join(settings['output_dir'], f'{idx}')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        complex = [
            template_reaction_sampler.pc_complex,
            template_reaction_sampler.rc_complex,
        ][template_reaction_sampler.isomorphism_idx]
        atoms = atoms_from_rdkit_mol(complex.rdkit_mol_obj, idx)
        ts_guess = Conformer(atoms=atoms)

        success = template_reaction_sampler.optimize_ts_guess(
            ts_guess=ts_guess,
            output_dir=output_dir,
            final_dir=settings['output_dir'],
        )

        if success:
            return

if __name__ == "__main__":    
    search_rxn_path_from_template()
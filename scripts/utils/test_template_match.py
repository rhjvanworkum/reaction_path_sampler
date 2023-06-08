import yaml
from reaction_path_sampler.src.template_sampler import TemplateSampler

from reaction_path_sampler.src.ts_template import TStemplate, get_ts_templates

from autode.bond_rearrangement import get_bond_rearrangs
from autode.mol_graphs import (
    get_mapping_ts_template,
    get_truncated_active_mol_graph,
)
from autode.transition_states.templates import template_matches
import autode as ade
from autode.species import Complex

if __name__ == "__main__":
    text_file = './data/ac_dataset_dcm_small.txt'
    base_settings_file = 'systems/ts_opt.yaml'

    with open(base_settings_file, "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    reaction_smiles_list = []
    with open(text_file, 'r') as f:
        for line in f.readlines():
            reaction_smiles_list.append(line.strip())

    ts_templates = get_ts_templates(folder_path='./scratch/templates/ac2/')

    for reaction_smiles in reaction_smiles_list:
        reactants, products = reaction_smiles.split('>>')
        settings['reactant_smiles'] = reactants.split('.')
        settings['product_smiles'] = products.split('.')
        settings['solvent'] = "DMF"

        template_sampler = TemplateSampler(settings)
        template_sampler.select_and_load_ts_template(ts_templates)
        print(template_sampler._cartesian_coord_constraints)
        break
        
    
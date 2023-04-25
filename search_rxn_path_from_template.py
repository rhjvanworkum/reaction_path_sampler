import argparse
import yaml
from typing import Any
import os
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import shutil

import autode as ade
from autode.species import Complex
from autode.bond_rearrangement import get_bond_rearrangs
from autode.transition_states.templates import template_matches
from autode.mol_graphs import (
    get_mapping_ts_template,
    get_truncated_active_mol_graph,
)
from autode.conformers.conformers import atoms_from_rdkit_mol
from autode.conformers.conformer import Conformer

from search_rxn_path import write_output_file
from src.interfaces.PYSISYPHUS import pysisyphus_driver
from src.interfaces.XTB import xtb_driver
from src.interfaces.xtb_utils import get_fixing_constraints
from src.reaction_path.reaction_ends import check_reaction_ends

from src.ts_template import get_ts_templates
from src.utils import autode_conf_to_xyz_string, get_canonical_smiles
from src.xyz2mol import get_canonical_smiles_from_xyz_string_ob


def search_reaction_path_from_template(settings: Any) -> None:
    # create output dir
    output_dir = settings["output_dir"]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # parse reaction SMILES
    if settings['reaction_smiles'] is not None:
        reactants, products = settings['reaction_smiles'].split('>>')
        reactant_smiles = reactants.split('.')
        product_smiles = products.split('.')
    else:
        reactant_smiles = settings['reactant_smiles']
        product_smiles = settings['product_smiles']

    # create autodE objects
    if len(reactant_smiles) == 1:
        reactants = ade.Molecule(smiles=reactant_smiles[0])
    else:
        reactants = Complex(*[ade.Molecule(smiles=smi) for smi in reactant_smiles])
    if len(product_smiles) == 1:
        products = ade.Molecule(smiles=product_smiles[0])
    else:
        products = Complex(*[ade.Molecule(smiles=smi) for smi in product_smiles])

    # find bond rearrangement & search for templates
    # TODO: write this for both reactnat -> products & product -> reacrtants
    bond_rearr = get_bond_rearrangs(products, reactants, name='test')[0]
    truncated_graph = get_truncated_active_mol_graph(graph=products.graph, active_bonds=bond_rearr.all)

    print([data["atom_label"] for _, data in truncated_graph.nodes(data=True)])

    for idx, ts_template in enumerate(get_ts_templates(folder_path=settings['template_folder_path'])):
        match, ignore_active_bonds = template_matches(products, truncated_graph, ts_template)
        if not match:
            print('Not MATCHED')
            continue
        else:
            print(f'MATCHED!!  {idx}')
            mapping = get_mapping_ts_template(
                larger_graph=truncated_graph, smaller_graph=ts_template.graph, ignore_active_bonds=ignore_active_bonds
            )
            # bond_constraints = {}
            # for active_bond in bond_rearr.all:
            #     i, j = active_bond
            #     try:
            #         dist = ts_template.graph.edges[mapping[i], mapping[j]]["distance"]
            #         bond_constraints[active_bond] = dist
            #     except KeyError:
            #         print(f"Couldn't find a mapping for bond {i}-{j}")
            # if len(bond_constraints) != len(bond_rearr.all):
            #     continue

            cartesian_constraints = {}
            for node in truncated_graph.nodes:
                try:
                    coords = ts_template.graph.nodes[mapping[node]]["cartesian"]
                    cartesian_constraints[node] = coords
                except KeyError:
                    print(f"Couldn't find a mapping for atom {node}")
            if len(cartesian_constraints) != len(truncated_graph.nodes):
                continue
            
            break

    return 

    # generate TS guesses using matched template
    cids = AllChem.EmbedMultipleConfs(
        mol=products.rdkit_mol_obj,
        numConfs=settings['n_confs'],
        coordMap={k: Point3D(*v) for k,v in cartesian_constraints.items()},
        randomSeed=420,
        numThreads=1,
        maxAttempts=1000,
        pruneRmsThresh=1,
        ignoreSmoothingFailures=True,
        useRandomCoords=True,
    )
    i = 0
    for id in cids:
        print(id)
        i += 1
    print(i)

    for conformer_idx in range(settings['n_confs']):
        if not os.path.exists(f'{output_dir}/{conformer_idx}'):
            os.makedirs(f'{output_dir}/{conformer_idx}/')

        """ 1. Do constrained optimization of the TS guess """
        atoms = atoms_from_rdkit_mol(products.rdkit_mol_obj, cids[conformer_idx])
        conf = Conformer(atoms=atoms)
        cc = autode_conf_to_xyz_string(conf)
        write_output_file(cc, f'{output_dir}/{conformer_idx}/pre_opt_conf.xyz')

        opt_structure = xtb_driver(
            xyz_string=autode_conf_to_xyz_string(conf),
            charge=products.charge,
            mult=products.mult,
            job="opt",
            method="2",
            solvent="Methanol",
            xcontrol_settings=get_fixing_constraints([str(k + 1) for k in sorted(cartesian_constraints.keys())]),
            n_cores=2
        )
        write_output_file(opt_structure, f'{output_dir}/{conformer_idx}/opt_conf.xyz')

        """ 2. TS optimization + IRC """
        output, tsopt, imaginary_freq = pysisyphus_driver(
            geometry_files=[f'{output_dir}/{conformer_idx}/opt_conf.xyz'],
            charge=products.charge,
            mult=products.mult,
            job="ts_opt",
            solvent=settings['solvent'],
            n_mins_timeout=10 # settings['n_mins_timeout']
        )

        write_output_file(output, f'{output_dir}/{conformer_idx}/ts_search.out')
        if tsopt is not None and imaginary_freq is not None:
            tsopt[1] = f"{imaginary_freq} \n"
            write_output_file(tsopt, f'{output_dir}/{conformer_idx}/ts_opt.xyz') 

            if imaginary_freq < settings['min_ts_imaginary_freq'] and imaginary_freq > settings['max_ts_imaginary_freq']:
                output, forward_irc, backward_irc, forward_end, backward_end = pysisyphus_driver(
                    geometry_files=[f'{output_dir}/{conformer_idx}/ts_opt.xyz'],
                    charge=products.charge,
                    mult=products.mult,
                    job="irc",
                    solvent=settings['solvent'],
                    n_mins_timeout=10 # settings['n_mins_timeout']
                )
                write_output_file(output, f'{output_dir}/{conformer_idx}/irc.out')

                if None not in [backward_irc, forward_irc]:
                    backward_irc.reverse()
                    write_output_file(backward_irc + 10 * (tsopt + ["\n"]) + forward_irc, f'{output_dir}/{conformer_idx}/irc_path.xyz')
                    
                    if None not in [backward_end, forward_end]:
                        write_output_file(backward_end + ["\n"] + tsopt + ["\n"] + forward_end, f'{output_dir}/{conformer_idx}/reaction.xyz')
                        
                        try:
                            true_rc_smi_list = [get_canonical_smiles(smi) for smi in reactant_smiles]
                            true_pc_smi_list = [get_canonical_smiles(smi) for smi in product_smiles]
                            pred_rc_smi_list = get_canonical_smiles_from_xyz_string_ob("".join(backward_end))
                            pred_pc_smi_list = get_canonical_smiles_from_xyz_string_ob("".join(forward_end))

                            print(true_rc_smi_list, pred_rc_smi_list)
                            print(true_pc_smi_list, pred_pc_smi_list)
                            print('\n\n')

                            if check_reaction_ends(
                                true_rc_smi_list,
                                true_pc_smi_list,
                                pred_rc_smi_list,
                                pred_pc_smi_list,
                            ):
                                # also save tsopt + reaction + irc path
                                shutil.copy2(f'{output_dir}/{conformer_idx}/ts_opt.xyz', f'{output_dir}/ts_opt.xyz')
                                shutil.copy2(f'{output_dir}/{conformer_idx}/reaction.xyz', f'{output_dir}/reaction.xyz')
                                shutil.copy2(f'{output_dir}/{conformer_idx}/irc_path.xyz', f'{output_dir}/irc_path.xyz')
                                
                                print('finshed reaction \n\n')
                                break  

                        except Exception as e:
                            print('Failed to retrieve SMILES from IRC ends \n\n')

                    else:
                        print("IRC end opt failed\n\n")

                else:
                    print("IRC failed\n\n")

            else:
                print(f"TS curvature, ({imaginary_freq} cm-1), is not within allowed interval \n\n")

        else:
            print("TS optimization failed\n\n")


if __name__ == "__main__":    
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

    search_reaction_path_from_template(settings)








    # """ getting TS with autodE method """
    # import os
    # from autode.wrappers.XTB import XTB
    # from autode.transition_states.ts_guess import TSguess

    # ade.Config.XTB.path = os.environ["XTB_PATH"]

    # name = "guess"
    # method = XTB()
    # ts_guess = TSguess(
    #     name=f"ts_guess_{name}",
    #     atoms=product.atoms,
    #     reactant=product,
    #     product=reactants,
    #     bond_rearr=bond_rearr,
    # )
    # ts_guess.run_constrained_opt(
    #     name=name,
    #     distance_consts=bond_constraints,
    #     method=method,
    #     keywords=method.keywords.opt,
    # )
    # print(ts_guess)
    # with open('test.xyz', 'w') as f:
    #     f.writelines(autode_conf_to_xyz_string(ts_guess))
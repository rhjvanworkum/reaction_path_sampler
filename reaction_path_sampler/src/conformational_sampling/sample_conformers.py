from typing import Any, Union, List, Tuple
import os
from autode.conformers import Conformer
from autode.species import Complex

from reaction_path_sampler.src.conformational_sampling.topolgy_conformer_sampler import TopologyConformerSampler
from reaction_path_sampler.src.conformational_sampling.metadyn_conformer_sampler import MetadynConformerSampler
from reaction_path_sampler.src.molecular_system import MolecularSystem
from reaction_path_sampler.src.utils import read_trajectory_file, remove_whitespaces_from_xyz_strings, xyz_string_to_autode_atoms


# TODO: this is currently orderded in pc -> rc direction, why not rc -> pc?
def sample_reactant_and_product_conformers(
    reactants: MolecularSystem,
    products: MolecularSystem,
    settings: Any
) -> Tuple[List[Conformer]]:
    # pc conformers
    pc_conformers_save_path = os.path.join(settings["output_dir"], 'pcs.xyz')
    if settings["force_conformer_sampling"] or not os.path.exists(pc_conformers_save_path):
        if settings["sampling_method"] == "metadynamics":
            pc_conf_sampler = MetadynConformerSampler(
                smiles_strings=settings["product_smiles"], 
                settings=settings,
                solvent=settings["solvent"],
            )
            pc_conformers = pc_conf_sampler.sample_conformers(mol=products)
        elif settings["sampling_method"] == "autode":
            raise NotImplementedError
        elif settings["sampling_method"] == "metadynamics+topology":
            pc_conf_sampler =  MetadynConformerSampler(
                smiles_strings=settings["product_smiles"], 
                settings=settings,
                solvent=settings["solvent"],
            )
            pc_conformers = pc_conf_sampler.sample_conformers(mol=products)
        elif settings["sampling_method"] == "autode+topology":
            raise NotImplementedError
        
        with open(pc_conformers_save_path, 'w') as f:
            f.writelines(remove_whitespaces_from_xyz_strings(pc_conformers))
    else:
        pc_conformers, _ = read_trajectory_file(pc_conformers_save_path)

    # rc conformers
    rc_conformers_save_path = os.path.join(settings["output_dir"], 'rcs.xyz')
    if settings["force_conformer_sampling"] or not os.path.exists(rc_conformers_save_path):
        if settings["sampling_method"] == "metadynamics":
            rc_conf_sampler = MetadynConformerSampler(
                smiles_strings=settings["reactant_smiles"], 
                settings=settings,
                solvent=settings["solvent"],
            )
            rc_conformers = rc_conf_sampler.sample_conformers(mol=reactants)
        elif settings["sampling_method"] == "autode":
            raise NotImplementedError
        elif settings["sampling_method"] == "metadynamics+topology":
            rc_conf_sampler = TopologyConformerSampler(
                smiles_strings=settings["reactant_smiles"], 
                settings=settings,
                solvent=settings["solvent"],
                mol=reactants
            )
            rc_conformers = rc_conf_sampler.sample_conformers(pc_conformers)
        elif settings["sampling_method"] == "autode+topology":
            raise NotImplementedError

        with open(rc_conformers_save_path, 'w') as f:
            f.writelines(remove_whitespaces_from_xyz_strings(rc_conformers))
    else:
        rc_conformers, _ = read_trajectory_file(rc_conformers_save_path)


    rc_conformer_list, pc_conformer_list = [], []
    for conformer in rc_conformers:
        rc_conformer_list.append(Conformer(
            atoms=xyz_string_to_autode_atoms(conformer), 
            charge=reactants.charge, 
            mult=reactants.mult
        ))
    for conformer in pc_conformers:
        pc_conformer_list.append(Conformer(
            atoms=xyz_string_to_autode_atoms(conformer), 
            charge=reactants.charge, 
            mult=reactants.mult
        ))


    return rc_conformer_list, pc_conformer_list
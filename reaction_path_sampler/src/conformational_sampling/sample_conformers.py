from typing import Any, Union, List, Tuple
import os
from autode.conformers import Conformer
from autode.species import Complex

from reaction_path_sampler.src.conformational_sampling.topolgy_conformer_sampler import TopologyConformerSampler
from reaction_path_sampler.src.conformational_sampling.metadyn_conformer_sampler import MetadynConformerSampler
from reaction_path_sampler.src.utils import read_trajectory_file, remove_whitespaces_from_xyz_strings, xyz_string_to_autode_atoms


# this is currently orderded in pc -> rc direction
def sample_reactant_and_product_conformers(
    rc_complex: Complex,
    pc_complex: Complex,
    settings: Any
) -> Tuple[List[Conformer]]:
    output_dir = settings["output_dir"]
    reactant_smiles = settings["reactant_smiles"]
    product_smiles = settings["product_smiles"]
    solvent = settings["solvent"]
    sampling_method = settings["sampling_method"]

    pc_conformers_save_path = os.path.join(output_dir, 'pcs.xyz')
    if settings["force_conformer_sampling"] or not os.path.exists(pc_conformers_save_path):
        if sampling_method == "metadynamics":
            pc_conf_sampler = MetadynConformerSampler(
                smiles_strings=product_smiles, 
                settings=settings,
                solvent=solvent
            )
            pc_conformers = pc_conf_sampler.sample_conformers(pc_complex.conformers[0])
        elif sampling_method == "autode":
            raise NotImplementedError
        elif sampling_method == "metadynamics+topology":
            pc_conf_sampler =  MetadynConformerSampler(
                smiles_strings=product_smiles, 
                settings=settings,
                solvent=solvent
            )
            pc_conformers = pc_conf_sampler.sample_conformers(pc_complex.conformers[0])
        elif sampling_method == "autode+topology":
            raise NotImplementedError
        
        with open(pc_conformers_save_path, 'w') as f:
            f.writelines(remove_whitespaces_from_xyz_strings(pc_conformers))
    else:
        pc_conformers, _ = read_trajectory_file(pc_conformers_save_path)

        
    rc_conformers_save_path = os.path.join(output_dir, 'rcs.xyz')
    if settings["force_conformer_sampling"] or not os.path.exists(rc_conformers_save_path):
        if sampling_method == "metadynamics":
            rc_conf_sampler = MetadynConformerSampler(
                smiles_strings=reactant_smiles, 
                settings=settings,
                solvent=solvent
            )
            rc_conformers = rc_conf_sampler.sample_conformers(rc_complex.conformers[0])
        elif sampling_method == "autode":
            raise NotImplementedError
        elif sampling_method == "metadynamics+topology":
            rc_conf_sampler = TopologyConformerSampler(
                smiles_strings=reactant_smiles, 
                settings=settings,
                solvent=solvent,
                mol=rc_complex.conformers[0]
            )
            rc_conformers = rc_conf_sampler.sample_conformers(pc_conformers)
        elif sampling_method == "autode+topology":
            raise NotImplementedError

        with open(rc_conformers_save_path, 'w') as f:
            f.writelines(remove_whitespaces_from_xyz_strings(rc_conformers))
    else:
        rc_conformers, _ = read_trajectory_file(rc_conformers_save_path)


    rc_conformer_list, pc_conformer_list = [], []
    for conformer in rc_conformers:
        rc_conformer_list.append(Conformer(
            atoms=xyz_string_to_autode_atoms(conformer), 
            charge=rc_complex.charge, 
            mult=rc_complex.mult
        ))
    for conformer in pc_conformers:
        pc_conformer_list.append(Conformer(
            atoms=xyz_string_to_autode_atoms(conformer), 
            charge=rc_complex.charge, 
            mult=rc_complex.mult
        ))


    return rc_conformer_list, pc_conformer_list
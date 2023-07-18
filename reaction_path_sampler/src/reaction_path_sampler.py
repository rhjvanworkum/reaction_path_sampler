import logging
import numpy as np
from typing import Dict, List, Tuple, Any
import os
import time
import shutil

from rdkit import Chem

import autode as ade
from autode.species import Complex
from autode.conformers.conformer import Conformer
from autode.input_output import atoms_to_xyz_file

from geodesic_interpolate.fileio import write_xyz as write_geodesic_xyz
from reaction_path_sampler.src import ReactionSampler

from reaction_path_sampler.src.conformational_sampling.sample_conformers import sample_reactant_and_product_conformers
from reaction_path_sampler.src.interfaces.PYSISYPHUS import pysisyphus_driver
from reaction_path_sampler.src.interfaces.XTB import xtb_driver
from reaction_path_sampler.src.interfaces.xtb_utils import comp_ad_mat_xtb
from reaction_path_sampler.src.molecular_system import Reaction
from reaction_path_sampler.src.molecule import parse_geometry_from_xyz_string
from reaction_path_sampler.src.reaction_path.complexes import compute_optimal_coordinates, generate_reaction_complex, select_promising_reactant_product_pairs
from reaction_path_sampler.src.reaction_path.barrier import compute_barrier
from reaction_path_sampler.src.reaction_path.mapped_complex import generate_mapped_reaction_complexes
from reaction_path_sampler.src.reaction_path.path_interpolation import interpolate_geodesic
from reaction_path_sampler.src.reaction_path.reaction_ends import check_reaction_ends_by_smiles, check_reaction_ends_by_graph_topology
from reaction_path_sampler.src.reaction_path.reaction_graph import get_reaction_graph_isomorphism
from reaction_path_sampler.src.ts_template import get_constraints_from_template, save_ts_template
from reaction_path_sampler.src.utils import autode_conf_to_xyz_string, comp_adj_mat, get_adj_mat_from_mol_block_string, get_canonical_smiles, remap_conformer, set_autode_settings, visualize_graph, write_output_file
from reaction_path_sampler.src.graphs.xyz2mol import get_canonical_smiles_from_xyz_string
from reaction_path_sampler.src.interfaces.methods import barrier_calculation_methods_dict

class ReactionPathSampler(ReactionSampler):

    def __init__(
        self,
        settings: Dict[str, Any],
        reaction: Reaction
    ) -> None:
        super().__init__(settings=settings)
        self.rc_complex = reaction.reactants.autode_complex
        self.pc_complex = reaction.products.autode_complex

        self.reaction = reaction

        self.isomorphism_idx = reaction._isomorphism_idx
        self.bond_rearr = reaction._bond_rearr

        self.solvent = reaction.solvent
        self.charge = reaction.reactants.charge
        self.mult = reaction.reactants.mult
    
    def sample_reaction_complex_conformers(
        self,
    ) -> Tuple[List[Conformer]]:
        """
        Sample conformers of both the reactant and product complexes.
        """
        rc_conformers, pc_conformers = sample_reactant_and_product_conformers(
            self.reaction.reactants,
            self.reaction.products,
            settings=self.settings
        )
        return rc_conformers, pc_conformers

    def select_promising_reactant_product_pairs(
        self,
        rc_conformers,
        pc_conformers,
        charge
    ) -> List[List[Conformer]]:
        """
        Select the most promising reactant-product pairs to try and find a reaction path for.
        """
        t = time.time()
        print('Selecting most promising Reactant-Product complexes now...')
        closest_pairs = select_promising_reactant_product_pairs(
            rc_conformers=rc_conformers,
            pc_conformers=pc_conformers,
            species_complex_mapping=None,       # currently unused
            bonds=None,                         # currently unused
            charge=charge,
            settings=self.settings
        )
        print(f'Selecting most promising Reactant-Product Complex pairs took: {time.time() - t}\n')
        print(f'Selected Reactant-Product Complex pairs: {closest_pairs}\n\n')

        return [[rc_conformers[idx[0]], pc_conformers[idx[1]]] for idx in closest_pairs]

    def _align_conformers(
        self,
        rc_conformer: Conformer,
        pc_conformer: Conformer,
        output_dir: str
    ) -> None:
        """
        Align the reactant and product conformers to each other.
        """
        rc_conformer._coordinates = compute_optimal_coordinates(rc_conformer.coordinates, pc_conformer.coordinates)
        atoms_to_xyz_file(rc_conformer.atoms, os.path.join(output_dir, 'selected_rc.xyz'))
        atoms_to_xyz_file(pc_conformer.atoms,  os.path.join(output_dir, 'selected_pc.xyz'))

    def _create_interpolated_path(
        self,
        rc_conformer: Conformer,
        pc_conformer: Conformer,
        output_dir: str
    ):
        """
        Create an interpolated path between the reactant and product conformers using geodescic interpolation.
        """
        t = time.time()
        curve = interpolate_geodesic(
            rc_conformer.atomic_symbols, 
            rc_conformer.coordinates, 
            pc_conformer.coordinates,
            self.settings
        )
        write_geodesic_xyz(os.path.join(output_dir, 'geodesic_path.trj'), rc_conformer.atomic_symbols, curve.path)
        write_geodesic_xyz(os.path.join(output_dir, 'geodesic_path.xyz'), rc_conformer.atomic_symbols, curve.path)
        print(f'geodesic interpolation: {time.time() - t}')
        
    def _perform_cos_and_tsopt(
        self,
        output_dir: str,
    ) -> Tuple[bool, List[str], float]:
        t = time.time()
        output, cos_final_traj, tsopt, imaginary_freq = pysisyphus_driver(
            geometry_files=[os.path.join(output_dir, 'geodesic_path.trj')],
            charge=self.charge,
            mult=self.mult,
            job="ts_search",
            solvent=self.solvent
        )
        print(f'TS search time: {time.time() - t}, imaginary freq: {imaginary_freq}')

        write_output_file(output, os.path.join(output_dir, 'ts_search.out'))
        write_output_file(cos_final_traj, os.path.join(output_dir, 'cos_final_traj.xyz'))

        if tsopt is not None and imaginary_freq is not None:
            tsopt[1] = f"{imaginary_freq} \n"   # put imaginary freq in second line of tsopt
            write_output_file(tsopt, os.path.join(output_dir, 'tsopt.xyz'))

            if imaginary_freq < self.settings['min_ts_imaginary_freq'] and imaginary_freq > self.settings['max_ts_imaginary_freq']:
                    return True, tsopt, imaginary_freq
            else:
                print(f"TS curvature, ({imaginary_freq} cm-1), is not within allowed interval \n\n")
                return False, tsopt, imaginary_freq
        else:
            print("TS optimization failed\n\n")
            return False, None, None

    def _perform_irc_calculation(
        self,
        tsopt: List[str],
        output_dir: str,
    ) -> Tuple[bool, Tuple[List[str]]]:
        t = time.time()
        output, forward_irc, backward_irc, forward_end, backward_end = pysisyphus_driver(
            geometry_files=[os.path.join(output_dir, 'tsopt.xyz')],
            charge=self.charge,
            mult=self.mult,
            job="irc",
            solvent=self.solvent
        )
        print(f'IRC time: {time.time() - t} \n\n')
        write_output_file(output, os.path.join(output_dir, 'irc.out'))

        if None not in [backward_irc, forward_irc]:
            backward_irc.reverse()
            write_output_file(backward_irc + 10 * (tsopt + ["\n"]) + forward_irc, os.path.join(output_dir, 'irc_path.xyz'))
            
            if None not in [backward_end, forward_end]:
                write_output_file(backward_end + ["\n"] + tsopt + ["\n"] + forward_end, os.path.join(output_dir, 'reaction.xyz'))
                
                return True, (backward_end, forward_end)

            else:
                print("IRC end opt failed\n\n")
                return False, (None, None)
        else:
            print("IRC failed\n\n")
            return False, (None, None)

    def _finalize_reaction(
        self,
        backward_end: List[str],
        tsopt: List[str],
        forward_end: List[str],
        output_dir: str,
        final_dir: str,
    ) -> bool:
        true_rc_smi_list = [get_canonical_smiles(smi) for smi in self.settings['reactant_smiles']]
        true_pc_smi_list = [get_canonical_smiles(smi) for smi in self.settings['product_smiles']]        
        try:
            pred_rc_smi_list = get_canonical_smiles_from_xyz_string("".join(backward_end), self.charge)
            pred_pc_smi_list = get_canonical_smiles_from_xyz_string("".join(forward_end), self.charge)

            print(f'True RC: {".".join(true_rc_smi_list)}, pred RC: {".".join(pred_rc_smi_list)}')
            print(f'True PC: {".".join(true_pc_smi_list)}, pred PC: {".".join(pred_pc_smi_list)}')
            print('\n\n')

            if self.settings['check_reaction_smiles_end']:
                match = check_reaction_ends_by_smiles(
                    true_rc_smi_list,
                    true_pc_smi_list,
                    pred_rc_smi_list,
                    pred_pc_smi_list,
                )
            else:
                true_rc_adj_mat = self.reaction.reactants.connectivity_matrix
                true_pc_adj_mat = self.reaction.products.connectivity_matrix
                pred_rc_adj_mat = comp_ad_mat_xtb(
                    xyz_string="".join(backward_end),
                    charge=self.charge,
                    mult=self.mult,
                    solvent=self.solvent
                )
                pred_pc_adj_mat = comp_ad_mat_xtb(
                    xyz_string="".join(forward_end),
                    charge=self.charge,
                    mult=self.mult,
                    solvent=self.solvent
                )

                if (len(true_rc_smi_list) == len(pred_rc_smi_list) and len(true_pc_smi_list) == len(pred_pc_smi_list)) or \
                    (len(true_rc_smi_list) == len(pred_pc_smi_list) and len(true_pc_smi_list) == len(pred_rc_smi_list)):
                    match = check_reaction_ends_by_graph_topology(
                        true_rc_adj_mat,
                        true_pc_adj_mat,
                        pred_rc_adj_mat,
                        pred_pc_adj_mat,
                        self.settings['irc_end_graph_threshold']
                    )
                else:
                    match = False
        except:
            print('Could not get canonical SMILES from xyz string\n\n')
            print('\n\n')

            true_rc_adj_mat = self.reaction.reactants.connectivity_matrix
            true_pc_adj_mat = self.reaction.products.connectivity_matrix
            pred_rc_adj_mat = comp_ad_mat_xtb(
                xyz_string="".join(backward_end),
                charge=self.charge,
                mult=self.mult,
                solvent=self.solvent
            )
            pred_pc_adj_mat = comp_ad_mat_xtb(
                xyz_string="".join(forward_end),
                charge=self.charge,
                mult=self.mult,
                solvent=self.solvent
            )

            match = check_reaction_ends_by_graph_topology(
                true_rc_adj_mat,
                true_pc_adj_mat,
                pred_rc_adj_mat,
                pred_pc_adj_mat,
                self.settings['irc_end_graph_threshold']
            )

        if match:
            complex = [self.rc_complex, self.pc_complex][1 - self.isomorphism_idx].copy()

            # save TS template
            ts_template = save_ts_template(
                tsopt="\n".join(tsopt), 
                complex=complex,
                bond_rearr=self.bond_rearr,
                output_dir=final_dir
            )

            # also save tsopt + reaction + irc path
            for file in ['tsopt.xyz', 'reaction.xyz', 'irc_path.xyz']:
                shutil.copy2(
                    os.path.join(output_dir, file),
                    os.path.join(final_dir, file)
                )
            
            # compute barrier & save barrier
            constraints = get_constraints_from_template(complex, self.bond_rearr, ts_template)
            barrier = compute_barrier(
                reactant_conformers_file_path=os.path.join(final_dir, 'rcs.xyz'),
                ts_geometry=tsopt,
                charge=self.charge,
                mult=self.mult,
                solvent=self.solvent,
                settings=self.settings,
                constraints=constraints,
                method=barrier_calculation_methods_dict[self.settings['barrier_method']]
            )

            with open(os.path.join(final_dir, 'barrier.txt'), 'w') as f:
                f.write(str(barrier))

            print('finshed reaction \n\n')
            
            return True
        
        return False
        
    def _cleanup(self, output_dir: str):
        if os.path.exists(f'{output_dir}/geodesic_path.trj'):
            os.remove(f'{output_dir}/geodesic_path.trj')
        if os.path.exists('test_BR.txt'):
            os.remove('test_BR.txt')
        if os.path.exists('nul'):
            os.remove('nul')
        if os.path.exists('run.out'):
            os.remove('run.out')

    def find_reaction_path(
        self,
        rc_conformer: Conformer,
        pc_conformer: Conformer,
        output_dir: str,
        final_dir: str
    ) -> bool:
        self._align_conformers(rc_conformer, pc_conformer, output_dir)
        self._create_interpolated_path(rc_conformer, pc_conformer, output_dir)
        ts_found, tsopt, imaginary_freq = self._perform_cos_and_tsopt(output_dir)

        success = False
        if ts_found:
            irc_path_found, (backward_end, forward_end) = self._perform_irc_calculation(tsopt, output_dir)

            if irc_path_found:
                reaction_path_found = self._finalize_reaction(backward_end, tsopt, forward_end, output_dir, final_dir)

                if reaction_path_found:
                    success = True
                
        self._cleanup(output_dir)

        return success
import logging
import numpy as np
from typing import Dict, List, Tuple, Any
import os
import time
import shutil

import autode as ade
from autode.species import Complex
from autode.conformers.conformer import Conformer
from autode.input_output import atoms_to_xyz_file

from geodesic_interpolate.fileio import write_xyz as write_geodesic_xyz

from reaction_path_sampler.src.conformational_sampling.sample_conformers import sample_reactant_and_product_conformers
from reaction_path_sampler.src.interfaces.PYSISYPHUS import pysisyphus_driver
from reaction_path_sampler.src.reaction_path.complexes import compute_optimal_coordinates, generate_reaction_complex, select_promising_reactant_product_pairs
from reaction_path_sampler.src.reaction_path.barrier import compute_barrier
from reaction_path_sampler.src.reaction_path.mapped_complex import generate_mapped_reaction_complexes
from reaction_path_sampler.src.reaction_path.path_interpolation import interpolate_geodesic
from reaction_path_sampler.src.reaction_path.reaction_ends import check_reaction_ends
from reaction_path_sampler.src.reaction_path.reaction_graph import get_reaction_graph_isomorphism
from reaction_path_sampler.src.ts_template import get_constraints_from_template, save_ts_template
from reaction_path_sampler.src.utils import autode_conf_to_xyz_string, get_canonical_smiles, remap_conformer, set_autode_settings, write_output_file
from reaction_path_sampler.src.xyz2mol import get_canonical_smiles_from_xyz_string
from reaction_path_sampler.src.interfaces.methods import barrier_calculation_methods_dict

class ReactionPathSampler:

    def __init__(
        self,
        settings: Dict[str, Any]
    ) -> None:
        self.settings = settings

        # create output dir
        if not os.path.exists(self.settings['output_dir']):
            os.makedirs(self.settings['output_dir'])

        # set autode settings
        set_autode_settings(settings)

        self._rc_complex = None
        self._pc_complex = None

        self.solvent = self.settings['solvent']
        self._charge = None
        self._mult = None

        self._bond_rearr = None
        self._isomorphism_idx = None

    @property
    def rc_complex(self) -> Complex:
        if self._rc_complex is None:
            raise ValueError('Reactant complex not set, call generate_reaction_complexes() first')
        return self._rc_complex

    @property
    def pc_complex(self) -> Complex:
        if self._pc_complex is None:
            raise ValueError('Reactant complex not set, call generate_reaction_complexes() first')
        return self._pc_complex

    @property
    def charge(self) -> int:
        if self._charge is None:
            raise ValueError('Charge not set, call generate_reaction_complexes() first')
        return self._charge

    @property
    def mult(self) -> int:
        if self._mult is None:
            raise ValueError('Mult not set, call generate_reaction_complexes() first')
        return self._mult
    
    @property
    def bond_rearr(self) -> ade.bond_rearrangement.BondRearrangement:
        if self._bond_rearr is None:
            raise ValueError('Bond rearrangement not set, call map_reaction_complexes() first')
        return self._bond_rearr

    @property
    def isomorphism_idx(self) -> int:
        if self._isomorphism_idx is None:
            raise ValueError('Isomorphism index not set, call map_reaction_complexes() first')
        return self._isomorphism_idx


    def generate_reaction_complexes(self) -> None:
        """
        Generate reactant and product complexes using autodE.
        """
        if self.settings['use_rxn_mapper']:
            rc_complex, pc_complex = generate_mapped_reaction_complexes(
                self.settings['reactant_smiles'],
                self.settings['product_smiles'],
                solvent=self.solvent
            )
        else:
            rc_complex = generate_reaction_complex(self.settings['reactant_smiles'])
            pc_complex = generate_reaction_complex(self.settings['product_smiles'])

        assert rc_complex.charge == pc_complex.charge
        assert pc_complex.mult == pc_complex.mult

        self._rc_complex = rc_complex
        self._pc_complex = pc_complex
        self._charge = rc_complex.charge
        self._mult = rc_complex.mult

    def map_reaction_complexes(
        self,
    ) -> None:
        """
        Map the reactant and product complexes to each other, such that atom ordering
        is the same in both geometries/complexes.
        """
        if self.settings["use_rxn_mapper"]:
            try:
                bond_rearr, isomorphism, isomorphism_idx = get_reaction_graph_isomorphism(
                    rc_complex=self.rc_complex, 
                    pc_complex=self.pc_complex, 
                    settings=self.settings,
                    node_label="atom_index"
                )
            except:
                bond_rearr, isomorphism, isomorphism_idx = get_reaction_graph_isomorphism(
                    rc_complex=self.rc_complex, 
                    pc_complex=self.pc_complex, 
                    settings=self.settings,
                    node_label="atom_label"
                )
        else:
            bond_rearr, isomorphism, isomorphism_idx = get_reaction_graph_isomorphism(
                rc_complex=self.rc_complex, 
                pc_complex=self.pc_complex, 
                settings=self.settings,
                node_label="atom_label"
            )

        if isomorphism_idx == 0:
            self.rc_complex.conformers = [remap_conformer(conf, isomorphism) for conf in self.rc_complex.conformers]
        elif isomorphism_idx == 1:
            self.pc_complex.conformers = [remap_conformer(conf, isomorphism) for conf in self.pc_complex.conformers]

        self._bond_rearr = bond_rearr
        self._isomorphism_idx = isomorphism_idx
    
    def sample_reaction_complex_conformers(
        self,
    ) -> Tuple[List[Conformer]]:
        """
        Sample conformers of both the reactant and product complexes.
        """
        rc_conformers, pc_conformers = sample_reactant_and_product_conformers(
            rc_complex=self.rc_complex,
            pc_complex=self.pc_complex,
            settings=self.settings
        )
        return rc_conformers, pc_conformers

    def select_promising_reactant_product_pairs(
        self,
        rc_conformers,
        pc_conformers
    ) -> List[List[Conformer]]:
        """
        Select the most promising reactant-product pairs to try and find a reaction path for.
        """
        t = time.time()
        logging.info('Selecting most promising Reactant-Product complexes now...')
        closest_pairs = select_promising_reactant_product_pairs(
            rc_conformers=rc_conformers,
            pc_conformers=pc_conformers,
            species_complex_mapping=None,       # currently unused
            bonds=None,                         # currently unused
            settings=self.settings
        )
        logging.info(f'Selecting most promising Reactant-Product Complex pairs took: {time.time() - t}\n')
        logging.info(f'Selected Reactant-Product Complex pairs: {closest_pairs}\n\n')

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
        logging.info(f'geodesic interpolation: {time.time() - t}')
        
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
        logging.info(f'TS search time: {time.time() - t}, imaginary freq: {imaginary_freq}')

        write_output_file(output, os.path.join(output_dir, 'ts_search.out'))
        write_output_file(cos_final_traj, os.path.join(output_dir, 'cos_final_traj.xyz'))

        if tsopt is not None and imaginary_freq is not None:
            tsopt[1] = f"{imaginary_freq} \n"   # put imaginary freq in second line of tsopt
            write_output_file(tsopt, os.path.join(output_dir, 'tsopt.xyz'))

            if imaginary_freq < self.settings['min_ts_imaginary_freq'] and imaginary_freq > self.settings['max_ts_imaginary_freq']:
                    return True, tsopt, imaginary_freq
            else:
                logging.info(f"TS curvature, ({imaginary_freq} cm-1), is not within allowed interval \n\n")
                return False, tsopt, imaginary_freq
        else:
            logging.info("TS optimization failed\n\n")
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
        logging.info(f'IRC time: {time.time() - t} \n\n')
        write_output_file(output, os.path.join(output_dir, 'irc.out'))

        if None not in [backward_irc, forward_irc]:
            backward_irc.reverse()
            write_output_file(backward_irc + 10 * (tsopt + ["\n"]) + forward_irc, os.path.join(output_dir, 'irc_path.xyz'))
            
            if None not in [backward_end, forward_end]:
                write_output_file(backward_end + ["\n"] + tsopt + ["\n"] + forward_end, os.path.join(output_dir, 'reaction.xyz'))
                
                return True, (backward_end, forward_end)

            else:
                logging.info("IRC end opt failed\n\n")
                return False, None
        else:
            logging.info("IRC failed\n\n")
            return False, None

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
        pred_rc_smi_list = get_canonical_smiles_from_xyz_string("".join(backward_end), self.charge)
        pred_pc_smi_list = get_canonical_smiles_from_xyz_string("".join(forward_end), self.charge)

        print(f'True RC: {".".join(true_rc_smi_list)}, pred RC: {".".join(pred_rc_smi_list)}')
        print(f'True PC: {".".join(true_pc_smi_list)}, pred PC: {".".join(pred_pc_smi_list)}')
        print('\n\n')

        if check_reaction_ends(
            true_rc_smi_list,
            true_pc_smi_list,
            pred_rc_smi_list,
            pred_pc_smi_list,
        ):
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

            logging.info('finshed reaction \n\n')
            
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
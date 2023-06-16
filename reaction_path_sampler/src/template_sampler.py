import logging
import os
import shutil
import time
from typing import Dict, Any, List, Tuple
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
import autode as ade
from autode.bond_rearrangement import get_bond_rearrangs
from autode.transition_states.templates import template_matches
from autode.mol_graphs import (
    get_mapping_ts_template,
    get_truncated_active_mol_graph,
)
from autode.conformers.conformer import Conformer

from reaction_path_sampler.src import ReactionSampler
from reaction_path_sampler.src.interfaces.PYSISYPHUS import pysisyphus_driver
from reaction_path_sampler.src.interfaces.XTB import xtb_driver
from reaction_path_sampler.src.interfaces.xtb_utils import get_fixing_constraints
from reaction_path_sampler.src.reaction_path.reaction_ends import check_reaction_ends
from reaction_path_sampler.src.ts_template import TStemplate
from reaction_path_sampler.src.utils import autode_conf_to_xyz_string, get_canonical_smiles, write_output_file
from reaction_path_sampler.src.visualization.plotly import plot_networkx_mol_graph
from reaction_path_sampler.src.graphs.xyz2mol import get_canonical_smiles_from_xyz_string


class TemplateSampler(ReactionSampler):

    def __init__(
        self,
        settings: Dict[str, Any]
    ) -> None:
        super().__init__(settings)

        self._isomorphism_idx = None
        self._bond_rearr = None
        self._cartesian_coord_constraints = None

    @property
    def isomorphism_idx(self) -> int:
        if self._isomorphism_idx is None:
            raise ValueError('Bond rearrangement not set, call select_and_load_ts_template() first')
        return self._isomorphism_idx

    @property
    def bond_rearr(self) -> ade.bond_rearrangement.BondRearrangement:
        if self._bond_rearr is None:
            raise ValueError('Bond rearrangement not set, call select_and_load_ts_template() first')
        return self._bond_rearr

    @property
    def cartesian_coord_constraints(self) -> Dict[int, np.ndarray]:
        if self._cartesian_coord_constraints is None:
            raise ValueError('Cartesian Coord Constraints not set, call select_and_load_ts_template() first')
        return self._cartesian_coord_constraints

    def select_and_load_ts_template(
        self,
        ts_templates: List[TStemplate]
    ) -> None:
        for isomorphism_idx, (reactants, products) in enumerate([
            (self.rc_complex, self.pc_complex),
            (self.pc_complex, self.rc_complex)
        ]):
            bond_rearrs = get_bond_rearrangs(products, reactants, name='test')
            if bond_rearrs is not None:
                for bond_rearr in bond_rearrs:
                    truncated_product_graph = get_truncated_active_mol_graph(graph=products.graph, active_bonds=bond_rearr.all)

                    import networkx as nx
                    nx.set_node_attributes(
                        truncated_product_graph, 
                        {node: products.coordinates[node] for _, node in enumerate(truncated_product_graph.nodes)},
                        'cartesian'
                    )
                    plot_networkx_mol_graph(
                        truncated_product_graph
                    )

                    for ts_template in ts_templates:
                        match, ignore_active_bonds = template_matches(products, truncated_product_graph, ts_template)
                        
                        if not match:
                            continue
                        else:
                            mapping = get_mapping_ts_template(
                                larger_graph=truncated_product_graph, 
                                smaller_graph=ts_template.graph, 
                                ignore_active_bonds=ignore_active_bonds
                            )

                            cartesian_constraints = {}
                            for node in truncated_product_graph.nodes:
                                try:
                                    coords = ts_template.graph.nodes[mapping[node]]["cartesian"]
                                    cartesian_constraints[node] = coords
                                except KeyError:
                                    print(f"Couldn't find a mapping for atom {node}")
                            if len(cartesian_constraints) != len(truncated_product_graph.nodes):
                                continue

                            self._isomorphism_idx = isomorphism_idx
                            self._bond_rearr = bond_rearr
                            self._cartesian_coord_constraints = cartesian_constraints
                            
                            return

        raise ValueError('Could not find a matching TS template')
                        
    def embed_ts_guesses(
        self,
    ) -> int:
        complex = [self.pc_complex, self.rc_complex][self.isomorphism_idx]
        cids = AllChem.EmbedMultipleConfs(
            mol=complex.rdkit_mol_obj,
            numConfs=self.settings['n_confs'],
            coordMap={k: Point3D(*v) for k,v in self.cartesian_coord_constraints.items()},
            randomSeed=420,
            numThreads=1,
            maxAttempts=1000,
            pruneRmsThresh=1,
            ignoreSmoothingFailures=True,
            useRandomCoords=False,
        )
        print(f'Embedded {len(cids)} TS guess geometries \n\n')
        return len(cids)

    def _constrained_optimization(
        self,
        geometry: Conformer,
        output_dir: str,
    ):
        complex = [self.pc_complex, self.rc_complex][self.isomorphism_idx]

        cc = autode_conf_to_xyz_string(geometry)
        write_output_file(cc, os.path.join(output_dir, 'pre_opt_conf.xyz'))

        opt_structure = xtb_driver(
            xyz_string=autode_conf_to_xyz_string(geometry),
            charge=complex.charge,
            mult=complex.mult,
            job="opt",
            method="2",
            solvent=self.solvent,
            xcontrol_settings=get_fixing_constraints([str(k + 1) for k in sorted(self.cartesian_coord_constraints.keys())]),
            n_cores=2
        )
        write_output_file(opt_structure, os.path.join(output_dir, 'opt_conf.xyz'))

    def _ts_optimization(
        self,
        output_dir: str,
    ):
        t = time.time()
        output, tsopt, imaginary_freq = pysisyphus_driver(
            geometry_files=[os.path.join(output_dir, 'opt_conf.xyz')],
            charge=self.pc_complex.charge,
            mult=self.pc_complex.mult,
            job="ts_opt",
            solvent=self.settings['solvent'],
            n_mins_timeout=10 # self.settings['n_mins_timeout']
        )
        print(f'TS search time: {time.time() - t}, imaginary freq: {imaginary_freq}')

        write_output_file(output, os.path.join(output_dir, 'ts_search.out'))

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
                return False, None
        else:
            print("IRC failed\n\n")
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

            # # save TS template
            # ts_template = save_ts_template(
            #     tsopt="\n".join(tsopt), 
            #     complex=complex,
            #     bond_rearr=self.bond_rearr,
            #     output_dir=final_dir
            # )

            # also save tsopt + reaction + irc path
            for file in ['tsopt.xyz', 'reaction.xyz', 'irc_path.xyz']:
                shutil.copy2(
                    os.path.join(output_dir, file),
                    os.path.join(final_dir, file)
                )
            
            # compute barrier & save barrier
            # constraints = get_constraints_from_template(complex, self.bond_rearr, ts_template)
            # barrier = compute_barrier(
            #     reactant_conformers_file_path=os.path.join(final_dir, 'rcs.xyz'),
            #     ts_geometry=tsopt,
            #     charge=self.charge,
            #     mult=self.mult,
            #     solvent=self.solvent,
            #     settings=self.settings,
            #     constraints=constraints,
            #     method=barrier_calculation_methods_dict[self.settings['barrier_method']]
            # )
            barrier = 0.0

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

    def optimize_ts_guess(
        self,
        ts_guess: Conformer,
        output_dir: str,
        final_dir: str
    ) -> None:
        self._constrained_optimization(ts_guess, output_dir)
        ts_found, tsopt, imaginary_freq = self._ts_optimization(output_dir)

        success = False
        if ts_found:
            irc_path_found, (backward_end, forward_end) = self._perform_irc_calculation(tsopt, output_dir)

            if irc_path_found:
                reaction_path_found = self._finalize_reaction(backward_end, tsopt, forward_end, output_dir, final_dir)

                if reaction_path_found:
                    success = True
                
        self._cleanup(output_dir)

        return success
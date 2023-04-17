
from typing import List
from autode.transition_states.templates import template_matches
from autode.mol_graphs import (
    get_mapping_ts_template,
    get_truncated_active_mol_graph,
)
from src.interfaces.PYSISYPHUS import pysisyphus_driver
from main import write_output_file
from src.interfaces.XTB import xtb_driver

from src.utils import autode_conf_to_xyz_string
from src.xyz2mol import canonical_smiles_from_xyz_string


if __name__ == "__main__":
    import autode as ade
    from autode.species import Complex
    from autode.bond_rearrangement import get_bond_rearrangs
    import networkx as nx
    import matplotlib.pyplot as plt

    from src.ts_template import TStemplate, get_ts_templates
    
    """ Optimize using template example """
    reaction = "C1=C([N+](=O)[O-])C=CO1.C=C>>C12CCC(O1)C([N+](=O)[O-])=C2"
    reactants, products = reaction.split('>>')
    reactants = reactants.split('.')
    products = products.split('.')
    reactants = Complex(*[ade.Molecule(smiles=reactants[0]), ade.Molecule(smiles=reactants[1])])
    product = ade.Molecule(smiles=products[0])

    bond_rearr = get_bond_rearrangs(product, reactants, name='test')[0]
    truncated_graph = get_truncated_active_mol_graph(graph=product.graph, active_bonds=bond_rearr.all)

    for ts_template in get_ts_templates(folder_path='./templates/'):
        if not template_matches(product, truncated_graph, ts_template):
            print('Not MATCHED')
            continue
        else:
            print('MATCHED!!')
            mapping = get_mapping_ts_template(
                larger_graph=truncated_graph, smaller_graph=ts_template.graph
            )
            bond_constraints = {}
            cartesian_constraints = {}

            for active_bond in bond_rearr.all:
                i, j = active_bond
                try:
                    dist = ts_template.graph.edges[mapping[i], mapping[j]]["distance"]
                    bond_constraints[active_bond] = dist
                except KeyError:
                    print(f"Couldn't find a mapping for bond {i}-{j}")
            
            for node in truncated_graph.nodes:
                try:
                    coords = ts_template.graph.nodes[mapping[node]]["cartesian"]
                    cartesian_constraints[node] = coords
                except KeyError:
                    print(f"Couldn't find a mapping for atom {node}")
            
            if len(bond_constraints) != len(bond_rearr.all):
                continue
            if len(cartesian_constraints) != len(truncated_graph.nodes):
                continue

            break


    
    """ Getting TS guess with cartesian constraints optimization """
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
    from autode.conformers.conformers import atoms_from_rdkit_mol
    from autode.conformers.conformer import Conformer

    n_confs = 1

    cids = AllChem.EmbedMultipleConfs(
        mol=product.rdkit_mol_obj,
        numConfs=n_confs,
        coordMap={k: Point3D(*v) for k,v in cartesian_constraints.items()},
        randomSeed=420,
        numThreads=1,
        pruneRmsThresh=1,
        useRandomCoords=True,
    )

    atoms = atoms_from_rdkit_mol(product.rdkit_mol_obj, cids[0])
    conf = Conformer(atoms=atoms)
    cc = autode_conf_to_xyz_string(conf)
    with open('pre_test.xyz', 'w') as f:
        f.writelines(cc)

    def get_fixing_constraints(atom_idxs: List[int]) -> str:
        string  = "$fix\n"
        string += f"  atoms: {','.join(atom_idxs)}\n"
        string += "$end\n"
        return string

    opt_structure = xtb_driver(
        xyz_string=autode_conf_to_xyz_string(conf),
        charge=product.charge,
        mult=product.mult,
        job="opt",
        method="2",
        solvent="Methanol",
        xcontrol_settings=get_fixing_constraints([str(k + 1) for k in sorted(cartesian_constraints.keys())]),
        n_cores=2
    )

    with open('test.xyz', 'w') as f:
        f.writelines(opt_structure)

    """ getting TS with autodE method """
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




    """ TS Optimization + IRC calculation """
    output, tsopt, imaginary_freq = pysisyphus_driver(
        geometry_files=[f'test.xyz'],
        charge=product.charge,
        mult=product.mult,
        job="ts_opt"
    )

    write_output_file(output, f'ts_search.out')
    if tsopt is not None and imaginary_freq is not None:
        tsopt[1] = f"{imaginary_freq} \n"
        write_output_file(tsopt, f'ts_opt.xyz') 

        if imaginary_freq < -300:

            output, forward_irc, backward_irc, forward_end, backward_end = pysisyphus_driver(
                geometry_files=[f'ts_opt.xyz'],
                charge=product.charge,
                mult=product.mult,
                job="irc"
            )

            if None not in [backward_irc, forward_irc]:
                backward_irc.reverse()
                write_output_file(backward_irc + 10 * (tsopt + ["\n"]) + forward_irc, f'irc_path.xyz')
                
                if None not in [backward_end, forward_end]:
                    write_output_file(backward_end + ["\n"] + tsopt + ["\n"] + forward_end, f'reaction.xyz')
                    
                    try:
                        rc_smiles = canonical_smiles_from_xyz_string("\n".join(backward_end), product.charge)
                        pc_smiles = canonical_smiles_from_xyz_string("\n".join(forward_end), product.charge)
                        print(rc_smiles)
                        print(pc_smiles)
                        print('\n\n')

                    except Exception as e:
                        print('Failed to retrieve SMILES from IRC ends \n\n')

                else:
                    print("IRC end opt failed\n\n")

            else:
                print("IRC failed\n\n")

    # 1. define reaction

    # 2. get template?

    # 3. sample conformers from RDKit using constrained positions

    # 4. optimize in XTB using constrained positions

    # 5. TSOPT + IRC calculation in Pysisyphus

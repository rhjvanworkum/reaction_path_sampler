import yaml

from autode.input_output import atoms_to_xyz_file

from reaction_pathway_sampler import ReactionPathwaySampler

# sn2 - C-Cl bond
reactant_smiles = ["[O-][N+](=O)CCCl", "[F-]"]
product_smiles =  ["[O-][N+](=O)CCF", "[Cl-]"]
rc_rc = [4, 5]
pc_rc = [4, 10]

# e2 - C-H bond
reactant_smiles = ["[O-][N+](=O)CCCl", "[F-]"]
product_smiles = ["[O-][N+](=O)C=C", "F", "[Cl-]"]
rc_rc = [3, 6]
pc_rc = [3, 9]

# DA 1 - dienophile dihedral
reactant_smiles = ["C1=CC=CO1", "C=C"]
product_smiles = ["C1=CC(O2)CCC12"]
rc_rc = [9, 11, 13, 14]
pc_rc = [4, 5, 12, 13]

# DA 2 - dienophile dihedral
reactant_smiles = ["C1=C(C(=O)O)C(Cl)=CO1", "C=CCNO"]
product_smiles = ["C(Cl)1=C(C(=O)O)C(O2)CC(CNO)C12"]
rc_rc = [13, 12, 17, 18]
pc_rc = [9, 8, 16, 17]

# claissen rearrangement - O-C bond
reactant_smiles = ["c1cccc(OCC=C)c1"]
product_smiles = ["C1=CC=CC(CC=C)C1(=O)"]
rc_rc = [5, 6]
pc_rc = [7, 9]

# SMC OA - aryl-halide bond
reactant_smiles = ["[Pd]([P](c1ccccc1)(c1ccccc1)(c1ccccc1))[P](c1ccccc1)(c1ccccc1)(c1ccccc1)", "c1cccc(Br)c1"]
product_smiles = ["[Pd](Br)(c1ccccc1)([P](c1ccccc1)(c1ccccc1)(c1ccccc1))[P](c1ccccc1)(c1ccccc1)(c1ccccc1)"]
rc_rc = []
pc_rc = []

# SMC TM -
reactant_smiles = ["[Pd](O)(c1ccccc1)([P](c1ccccc1)(c1ccccc1)(c1ccccc1))[P](c1ccccc1)(c1ccccc1)(c1ccccc1)", "c1ccccc1(B(O)(O)O)"]
product_smiles = ["[Pd](c1ccccc1)(c1ccccc1)([P](c1ccccc1)(c1ccccc1)(c1ccccc1))[P](c1ccccc1)(c1ccccc1)(c1ccccc1)", "B(O)(O)(O)O"]
rc_rc = []
pc_rc = []

# SMC RE - 
reactant_smiles = ["[Pd](c1ccccc1)(c1ccccc1)([P](c1ccccc1)(c1ccccc1)(c1ccccc1))[P](c1ccccc1)(c1ccccc1)(c1ccccc1)"]
product_smiles = ["[Pd]([P](c1ccccc1)(c1ccccc1)(c1ccccc1))[P](c1ccccc1)(c1ccccc1)(c1ccccc1)", "c1ccccc1c1ccccc1"]
rc_rc = []
pc_rc = []

if __name__ == "__main__":
    from autode.bond_rearrangement import get_bond_rearrangs

    reactant_smiles = ["C1=CC=CO1", "C=C"]
    product_smiles = ["C1=CC(O2)CCC12"]

    with open('./scratch/settings.yaml', "r") as f:
        settings = yaml.load(f, Loader=yaml.Loader)

    rps = ReactionPathwaySampler(reactant_smiles, settings)
    rc_complex = rps._sample_initial_complexes()[0]
    atoms_to_xyz_file(rc_complex.atoms, 'rc.xyz')

    rps = ReactionPathwaySampler(product_smiles, settings)
    pc_complex = rps._sample_initial_complexes()[0]
    atoms_to_xyz_file(pc_complex.atoms, 'pc.xyz')

    for idx, reaction_complexes in enumerate([
        [rc_complex, pc_complex],
        [pc_complex, rc_complex]
    ]):
        bond_rearrs = get_bond_rearrangs(reaction_complexes[1], reaction_complexes[0], name='test')
        if bond_rearrs is not None:
            for bond_rearr in bond_rearrs:
                if idx == 0:
                    print('reactants -> products')
                else:
                    print('products -> reactants')
                print(bond_rearr.fbonds, bond_rearr.bbonds, '\n')
        break


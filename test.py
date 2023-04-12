# from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
# import autode as ade
# from autode.species import Complex
# from autode.bond_rearrangement import get_bond_rearrangs
# from autode.mol_graphs import reac_graph_to_prod_graph

# from tqdm import tqdm
# import time

# import networkx as nx
# import matplotlib.pyplot as plt

# from XTB import xtb_driver

# def get_opt_energy(args):
#     conf, complex, method, settings, cores = args
#     return xtb_driver(
#         conf,
#         complex.charge,
#         complex.mult,
#         "opt",
#         method=method,
#         xcontrol_settings=settings,
#         n_cores=cores
#     )

# def conf_to_xyz_string(conf) -> str:
#     str = f"{len(conf.atoms)}\n \n"
#     for atom in conf.atoms:
#         str += f"{atom.atomic_symbol} {round(atom.coord.x, 4)} {round(atom.coord.y, 4)} {round(atom.coord.z, 4)}\n"
#     return str


# if __name__ == "__main__":
#     # reac_smiles = ["C1=CC=CO1", "C=C"]
#     # prod_smiles = ["C1=CC(O2)CCC12"]
#     reactant_smiles = ["C1=C(C(=O)O)C(Cl)=CO1", "C=CCNO"]
#     product_smiles = ["C(Cl)1=C(C(=O)O)C(O2)CC(CNO)C12"]
#     start = Complex(*[ade.Molecule(smiles=smi) for smi in reactant_smiles])
#     start._generate_conformers()


#     N_PROCESSES = 8
#     n_cores = 2
#     method = "2"

#     arguments = [(conf_to_xyz_string(conf), start, method, None, n_cores) for conf in start.conformers[:8]]

#     t = time.time()
#     with ThreadPoolExecutor(max_workers=N_PROCESSES) as executor:
#         results = list(tqdm(executor.map(get_opt_energy, arguments), total=len(arguments)))
#     print(time.time() - t)


import numpy as np

a = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

b = 2 * a

out = (b[np.newaxis, :, :] - a[:, np.newaxis, :])**2
out = np.sqrt(np.mean(out, axis=-1))

print(np.argwhere(out == np.min(out)))
print(out[np.argwhere(out == np.min(out))[0]])
# Out[21]: array([[2, 1]])

print(np.argmin(out, axis=-1))
print(np.argmin(out, axis=-2))

print(out)

# print(out[0, 0])
# print(out[1, 1])
# print(out[3, 1])
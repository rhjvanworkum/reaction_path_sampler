import numpy as np
import math

all_args = np.arange(2008)
n_processes = 7
chunk_size = math.ceil(len(all_args) / n_processes)
print(chunk_size)
args = [all_args[i:i+chunk_size] for i in np.arange(0, len(all_args), chunk_size)]
print(np.arange(0, len(all_args), chunk_size))

""" Test how many reaction paths worked """
# import os
# i = 0
# list = []
# for root, dirs, files in os.walk('./scratch/da_reaction_cores_2/'):
#     if len(root.split('/')) > 3 and root.split('/')[-2] == 'da_reaction_cores_2':
#         if os.path.exists(os.path.join(root, 'reaction.xyz')):
#             list.append(int(root.split('/')[-1]))
#             i += 1
# print(i)
# print(sorted(list))


""" Plot interpolated paths stuff """
# import matplotlib.pyplot as plt
# import numpy as np
# from src.molecule import read_xyz_string
# from src.utils import read_trajectory_file

# structures, _ = read_trajectory_file('better_path.xyz')
# geometries = []

# for struct in structures:
#     geometries.append(read_xyz_string(struct.split('\n')))

# n_atoms = len(geometries[0])

# for i in range(n_atoms):
#     for j in range(3):
#         plt.plot(np.arange(len(geometries)), [geom[i].coordinates[j] for geom in geometries])

# plt.savefig('test2.png')
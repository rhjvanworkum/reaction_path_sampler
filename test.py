import numpy as np


coords = np.ones((5, 3, 3))
coords[:, 0, :] = np.array([1, 2, 3])
coords[:, 1, :] = np.array([4, 5, 6])
coords[:, 2, :] = np.array([7, 8, 9])

coords2 = np.concatenate([coords, coords], axis=0)

rmsd = (coords2[..., np.newaxis, :, :] - coords[..., :, np.newaxis, :])**2
print(rmsd.shape)



""" Test how many reaction paths worked """
# import os
# i = 0
# for root, dirs, files in os.walk('./scratch/da_reaction_cores/'):
#     if len(root.split('/')) > 3 and root.split('/')[-2] == 'da_reaction_cores':
#         if os.path.exists(os.path.join(root, 'reaction.xyz')):
#             i += 1
# print(i)


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
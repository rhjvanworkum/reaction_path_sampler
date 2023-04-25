# import numpy as np
# import math

# all_args = np.arange(2008)
# n_processes = 7
# chunk_size = math.ceil(len(all_args) / n_processes)
# print(chunk_size)
# args = [all_args[i:i+chunk_size] for i in np.arange(0, len(all_args), chunk_size)]
# print(np.arange(0, len(all_args), chunk_size))

# print([0, 1, 2, 3, 4][2:])
# import os
# for i in range(28940, 30000):
#     os.system(f'scancel {i}')

# import os
# import shutil 

# for root, dirs, files in os.walk('./scratch/da_reaction_cores_4/'):
#     if len(root.split('/')) > 3 and root.split('/')[-2] == 'da_reaction_cores_4':
#         for dir in ['0', '1', '2', '3', '4']:
#             path = os.path.join(root, dir)
#             if os.path.exists(path):
#                 shutil.rmtree(path)

#             for _, _, files in os.walk(root):
#                 for file in files:
#                     if os.path.exists(os.path.join(root, file)):
#                         if file != 'rcs.xyz' and file != 'pcs.xyz':
#                             os.remove(os.path.join(root, file))

# a = [0, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 21, 23, 24, 25, 26, 30, 31, 32, 34, 35, 37, 38, 39, 40, 42, 43, 44, 45, 47, 49, 50, 51, 52, 53, 54, 55, 56, 57, 59, 61, 64, 65, 66, 67, 70, 71, 72, 73, 74, 75, 78, 79, 80, 81, 82, 83, 85, 86, 87, 88, 90, 91, 92, 93, 94, 95, 97, 98, 99, 100, 104, 105, 107, 108, 109, 110, 111, 112, 113, 114, 115, 117, 120, 122, 123, 124, 126, 127, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145]
# b = [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 62, 64, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 83, 84, 88, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 119, 120, 122, 123, 124, 126, 128, 130, 131, 132, 133, 134, 136, 137, 138, 139, 140, 141, 142, 144, 145]
c = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 30, 31, 32, 33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 110, 111, 112, 113, 114, 115, 116, 117, 119, 120, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 135, 136, 137, 138, 139, 140, 141, 144, 145]

# # for i in a:
# #     if i not in b:
# #         print(i)

for i in range(146):
    if i not in c:
        print(i)

""" Test how many reaction paths worked """
import os
i = 0
list = []
for root, dirs, files in os.walk('./scratch/da_reaction_cores_6/'):
    if len(root.split('/')) > 3 and root.split('/')[-2] == 'da_reaction_cores_6':
        if os.path.exists(os.path.join(root, 'reaction.xyz')):
            list.append(int(root.split('/')[-1]))
            i += 1
print(i)
print(sorted(list))


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
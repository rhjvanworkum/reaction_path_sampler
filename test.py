""" actual test area """
# import itertools

# nodes = [1, 2, 3, 4]
# for comb in itertools.combinations(nodes, 2):
#     print(comb)


""" Cancel SLURM jobs """
# import os
# for i in range(31686, 32000):
#     os.system(f'scancel {i}')

""" Keep only rc's & pc's from previous job """
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

""" Test how many reaction paths worked """
# import os
# i = 0
# list = []
# for root, dirs, files in os.walk('./scratch/da_reaction_cores_new/'):
#     if len(root.split('/')) > 3 and root.split('/')[-2] == 'da_reaction_cores_new':
#         if os.path.exists(os.path.join(root, 'reaction.xyz')):
#             list.append(int(root.split('/')[-1]))
#             i += 1
# print(i)
# print(sorted(list))


""" Print which TS templates matched or not """
import os
i = 0
list = []
for root, dirs, files in os.walk('./scratch/da_tss_test_new1/'):
    if len(root.split('/')) > 3 and root.split('/')[-2] == 'da_tss_test_new1':
       
        if root != './scratch/da_tss_test_new1/':

            for _, _, files in os.walk(root):
                for file in files:
                    if file.split('.')[-1] == 'out' and file.split('_')[0] == 'job':
                        with open(os.path.join(root, file), 'r') as f:
                            lines = "\n".join(f.readlines())

                            if "MATCHED!!" in lines:
                                list.append(int(root.split('/')[-1]))
                                i += 1

        # if os.path.exists(os.path.join(root, 'reaction.xyz')):
        #     list.append(int(root.split('/')[-1]))
        #     i += 1

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
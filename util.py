""" actual test area """
# import itertools

# nodes = [1, 2, 3, 4]
# for comb in itertools.combinations(nodes, 2):
#     print(comb)


""" Compare DF results """
# from sklearn.metrics import roc_auc_score, accuracy_score
# import pandas as pd

# df = pd.read_csv('./data/DA_regio_no_solvent_success.csv')
# true_column, pred_column = 'label', 'test_label'

# filtered_reaction_idxs = []
# for reaction_idx in df['reaction_idx'].unique():
#     reaction_df = df[df['reaction_idx'] == reaction_idx]
#     if len(reaction_df[~reaction_df[pred_column].isna()]) == len(reaction_df):
#         filtered_reaction_idxs.append(reaction_idx)

# df = df[df['reaction_idx'].isin(filtered_reaction_idxs)]
# true, pred = df[true_column], df[pred_column]

# print('Converged calculations: ', len(df))
# print(roc_auc_score(true, pred) * 100, 'AUROC')
# print(accuracy_score(true, pred) * 100, "%", "accuracy")


# # df = pd.read_csv('./data/test_da_reactions_2.csv')
# # original_df = df[df['simulation_idx'] == 0]
# # virtual_df = df[df['simulation_idx'] == 1]

# # labels = []
# # for _, row in virtual_df.iterrows():
# #     barrier = row['reaction_energy']
# #     other_barriers = virtual_df[virtual_df['substrates'] == row['substrates']]['reaction_energy']
# #     label = int((barrier <= other_barriers).all())
# #     labels.append(label)
# # virtual_df['labels'] = labels
# # df = pd.concat([original_df, virtual_df])


# right = 0

# targets, preds = [], []
# for idx in df['reaction_idx'].unique():
#     target = df[(df['reaction_idx'] == idx) & (df['simulation_idx'] == 0)]['labels']
#     pred = df[(df['reaction_idx'] == idx) & (df['simulation_idx'] == 1)]['labels']

#     if len(pred) > 0 and len(target) > 0 and 1 in pred.values:

#         if (target.values == pred.values).all():
#             right += 1
            
#         for val in target:
#             targets.append(val)
#         for val in pred:
#             preds.append(val)

# print(roc_auc_score(targets, preds) * 100, 'AUROC')
# print(right/ len(df['reaction_idx'].unique()) * 100, "%", "accuracy")


""" Cancel SLURM jobs """
import os
for i in range(47614, 50000):
    os.system(f'scancel {i}')

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
# for root, dirs, files in os.walk('./scratch/diels_alder_reaction_cores/'):
#     if len(root.split('/')) > 3 and root.split('/')[-2] == 'diels_alder_reaction_cores':
#         if os.path.exists(os.path.join(root, 'reaction.xyz')):
            
#             # for _, dirs, _ in os.walk(root):
#             #     if len(dirs) > 0:
#             #         list.append(len(dirs))

#             list.append(int(root.split('/')[-1]))
#             i += 1

# print(i)
# print(sorted(list))

# import numpy as np
# l = [j if j not in list else None for j in np.arange(300)]
# print(tuple(filter(lambda x: x is not None, l)))


""" Print which TS templates matched or not """
# import os
# i = 0
# list = []
# for root, dirs, files in os.walk('./scratch/da_tss_test/'):
#     if len(root.split('/')) > 3 and root.split('/')[-2] == 'da_tss_test':
       
#         # if root != './scratch/da_tss_test_new_new1/':

#         #     for _, _, files in os.walk(root):
#         #         for file in files:
#         #             if file.split('.')[-1] == 'out' and file.split('_')[0] == 'job':
#         #                 with open(os.path.join(root, file), 'r') as f:
#         #                     lines = "\n".join(f.readlines())

#         #                     if "MATCHED!!" in lines:
#         #                         list.append(int(root.split('/')[-1]))
#         #                         i += 1

#         if os.path.exists(os.path.join(root, 'reaction.xyz')):
#             list.append(int(root.split('/')[-1]))
#             i += 1

# print(i)
# print(sorted(list))



""" Plot interpolated paths stuff """
# import matplotlib.pyplot as plt
# import numpy as np
# from reaction_path_sampler.src.molecule import read_xyz_string
# from reaction_path_sampler.src.utils import read_trajectory_file

# structures, _ = read_trajectory_file('better_path.xyz')
# geometries = []

# for struct in structures:
#     geometries.append(read_xyz_string(struct.split('\n')))

# n_atoms = len(geometries[0])

# for i in range(n_atoms):
#     for j in range(3):
#         plt.plot(np.arange(len(geometries)), [geom[i].coordinates[j] for geom in geometries])

# plt.savefig('test2.png')
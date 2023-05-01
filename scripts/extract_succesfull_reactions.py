import os
import shutil

if __name__ == "__main__":
    reaction_cores_path = "./data/da_reaction_cores_new.txt"
    path = "./scratch/da_reaction_cores_new/"
    name = "da_reaction_cores_new"
    template_dir = './scratch/templates/da_cores_new/'

    with open(reaction_cores_path, 'r') as f:
        reaction_smiles_list = [line.replace('\n', '') for line in f.readlines()]

    list = []
    for root, dirs, files in os.walk(path):
        if len(root.split('/')) > 3 and root.split('/')[-2] == name:
            if os.path.exists(os.path.join(root, 'reaction.xyz')):
                list.append(int(root.split('/')[-1]))
    list = sorted(list)

    successfull_reaction_smiles = [reaction_smiles_list[i] for i in list]

    with open('succesfull_da_reaction_cores_new.txt', 'w') as f:
        f.writelines("\n".join(successfull_reaction_smiles))


    n = 0
    for i in list:
        for root, dirs, files in os.walk(template_dir):
            n = len(files)
        shutil.copy2(
            os.path.join(path, f'{i}/template0.txt'),
            os.path.join(template_dir, f'template{n}.txt')
        )

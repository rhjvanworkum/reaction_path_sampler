import os
import shutil

if __name__ == "__main__":
    path = "./scratch/da_reaction_cores_6/"
    name = "da_reaction_cores_6"
    template_dir = './scratch/templates/da_cores/'

    with open('da_reaction_cores.txt', 'r') as f:
        reaction_smiles_list = [line.replace('\n', '') for line in f.readlines()]

    list = []
    for root, dirs, files in os.walk(path):
        if len(root.split('/')) > 3 and root.split('/')[-2] == name:
            if os.path.exists(os.path.join(root, 'reaction.xyz')):
                list.append(int(root.split('/')[-1]))
    list = sorted(list)

    successfull_reaction_smiles = [reaction_smiles_list[i] for i in list]

    with open('succesfull_da_reaction_cores.txt', 'w') as f:
        f.writelines("\n".join(successfull_reaction_smiles))



    for i in list:
        for root, dirs, files in os.walk(template_dir):
            n = len(files)
        print(n)
        shutil.copy2(
            os.path.join(path, f'{i}/template0.txt'),
            os.path.join(template_dir, f'template{n}.txt')
        )

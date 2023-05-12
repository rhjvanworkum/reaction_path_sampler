import matplotlib.pyplot as plt
import numpy as np

file = './scratch/da_reaction_cores_test/4/0/cos_final_traj.xyz'

with open(file, 'r') as f:
    lines = f.readlines()

n_atoms = int(lines[0].split()[0])

energies = []

nice = True
i = 1
while nice:
    try:
        e = float(lines[i].split()[0])
        energies.append(e)
    except:
        nice = False
    i += (n_atoms + 2)


plt.plot(np.arange(len(energies)), energies)
plt.savefig('test.png')

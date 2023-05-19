import os
import pandas as pd
import numpy as np


dataset_path = "./data/DA_regio_no_solvent_success.csv"
dataset = pd.read_csv(dataset_path)
base_dir = "./scratch/DA_test_no_solvent/"

for i in dataset['uid'].values:
    results_dir = os.path.join(base_dir, f'{i}')

    jobs = []
    for _, _, files in os.walk(results_dir):
        for file in files:
            if file.split('.')[-1] == "out" and file.split('_')[0] == "job":
                jobs.append(int(file.split('.')[0].split('_')[-1]))

    jobs = sorted(jobs)
    job_file = os.path.join(results_dir, f"job_{jobs[-4]}.out")
    
    with open(job_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'conformers after pruning' in line:
                n_confs = int(line.split()[-1])
                break

    print(n_confs)


    # try:
    #     with open(os.path.join(results_dir, 'barrier.txt'), 'r') as f:
    #         barrier = float(f.readlines()[0])
    # except:
    #     barrier = np.nan
    # activation_energies.append(barrier)
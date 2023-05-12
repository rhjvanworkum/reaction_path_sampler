from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from tqdm import tqdm
from src.interfaces.ORCA import orca_driver
import os

from src.interfaces.PYSCF import pyscf_driver
from src.interfaces.XTB import xtb_driver
from src.utils import read_trajectory_file

os.environ['OMP_NUM_THREADS'] = "1" 

SOLVENT = "Methanol"

def compute_energies(structures):
    energies = [
        orca_driver(
            xyz_string="\n".join(struct.split('\n')[2:]),
            charge=0,
            mult=1,
            job="sp",
            solvent=SOLVENT
        ) for struct in structures
    ]
    return energies


if __name__ == "__main__":
    

    n_processes = 100 // 4

    df_path = './data/test_da_reactions_tss.csv'
    df = pd.read_csv(df_path)

    args = []
    for idx, row in df.iterrows():
        path = f'./scratch/da_tss_test/{row.iloc[0]}/reaction.xyz'
        structures, _ = read_trajectory_file(path)
        args.append(structures)
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(tqdm(executor.map(compute_energies, args), total=len(args), desc="DFT SP's"))


    uid = []
    reaction_idx = []
    substrates = []
    products = []
    reaction_smiles = []
    activation_barrier = []
    reaction_energy = []
    simulation_idx = []
    for idx, row in df.iterrows():
        energies = results[idx]

        uid.append(max(df['uid']) + 1 + idx)
        reaction_idx.append(row['reaction_idx'])
        substrates.append(row['substrates'])
        products.append(row['products'])
        reaction_smiles.append(row['reaction_smiles'])

        if energies[1] is not None and energies[0] is not None:
            barrier = energies[1] - energies[0]
        else:
            barrier = None
        activation_barrier.append(barrier)

        if energies[2] is not None and energies[0] is not None:
            r_energy = energies[2] - energies[0]
        else:
            r_energy = None
        reaction_energy.append(r_energy)
        
        simulation_idx.append(1)

    virtual_df =  pd.DataFrame({
        'reaction_idx': reaction_idx,
        'uid': uid,
        'substrates': substrates,
        'products': products,
        'reaction_smiles': reaction_smiles,
        'activation_barrier': activation_barrier,
        'reaction_energy': reaction_energy,
        'simulation_idx': simulation_idx
    }) 



    labels = []
    for _, row in virtual_df.iterrows():
        barrier = row['activation_barrier']
        other_barriers = virtual_df[virtual_df['substrates'] == row['substrates']]['activation_barrier']
        label = int((barrier <= other_barriers).all())
        labels.append(label)
    virtual_df['labels'] = labels



    new_df = pd.concat([df, virtual_df])
    new_df.to_csv('./data/test_da_reactions.csv')
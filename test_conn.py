from xyz2mol import canonical_smiles_from_xyz_string

import h5py 

if __name__ == "__main__":
    with open('./scratch/da/selected_pc.xyz', 'r') as f:
        lines = f.readlines()
    smiles_list = canonical_smiles_from_xyz_string(
        xyz_file_name="\n".join(lines),
        charge=0,    
    )

    print(smiles_list)

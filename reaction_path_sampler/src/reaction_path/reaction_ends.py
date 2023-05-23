from typing import List
import numpy as np
from rdkit import Chem

def check_product_connectivity(
    true_rc_smi_list: List[str],
    true_pc_smi_list: List[str],
    pred_rc_smi_list: List[str],
    pred_pc_smi_list: List[str]
) -> bool:
    """
    Advanced method to find when SMILES are not equal, the difference
    in the found product is just a bond change (e.g. a double bond was 
    translated as a single bond + 2 radicals)
    """
    # TODO: change this here
    if len(true_pc_smi_list) >= 2:
        return False

    if true_rc_smi_list == pred_rc_smi_list:
        # compare pred_pc_smiles_list with true_pc_smiles_list
        if len(pred_pc_smi_list) != len(true_pc_smi_list):
            return False
        
        mol1 = Chem.MolFromSmiles(pred_pc_smi_list[0])
        mol2 = Chem.MolFromSmiles(true_pc_smi_list[0])
        adj_mat1 = Chem.rdmolops.GetAdjacencyMatrix(mol1)
        adj_mat2 = Chem.rdmolops.GetAdjacencyMatrix(mol2)

        try:
            np.testing.assert_equal(adj_mat1, adj_mat2)
            return True
        except Exception as e:
            return False

    elif true_rc_smi_list == pred_pc_smi_list:
        # compare pred_rc_smiles_list with true_pc_smiles_list
        if len(pred_rc_smi_list) != len(true_pc_smi_list):
            return False
        
        mol1 = Chem.MolFromSmiles(pred_rc_smi_list[0])
        mol2 = Chem.MolFromSmiles(true_pc_smi_list[0])
        adj_mat1 = Chem.rdmolops.GetAdjacencyMatrix(mol1)
        adj_mat2 = Chem.rdmolops.GetAdjacencyMatrix(mol2)

        try:
            np.testing.assert_equal(adj_mat1, adj_mat2)
            return True
        except Exception as e:
            return False
    
    else:
        return False


def check_reaction_ends(
    true_rc_smi_list: List[str],
    true_pc_smi_list: List[str],
    pred_rc_smi_list: List[str],
    pred_pc_smi_list: List[str]
) -> bool:
    """
    Function to check whether the simulated reaction actually corresponds to 
    the intended reaction
    """
    for list in [
        true_rc_smi_list,
        true_pc_smi_list,
        pred_rc_smi_list,
        pred_pc_smi_list
    ]:
        list.sort()

    if (true_rc_smi_list == pred_rc_smi_list and true_pc_smi_list == pred_pc_smi_list) or \
       (true_rc_smi_list == pred_pc_smi_list and true_pc_smi_list == pred_rc_smi_list) or \
        check_product_connectivity(
            true_rc_smi_list,
            true_pc_smi_list,
            pred_rc_smi_list,
            pred_pc_smi_list        
        ):
        return True
    else:
        return False
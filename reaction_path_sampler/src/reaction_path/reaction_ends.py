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


def check_reaction_ends_by_smiles(
    true_rc_smi_list: List[str],
    true_pc_smi_list: List[str],
    pred_rc_smi_list: List[str],
    pred_pc_smi_list: List[str]
) -> bool:
    """
    Function to check whether the simulated reaction actually corresponds to 
    the intended reaction by comparing the retrieved SMILES strings
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


def check_reactant_product_graphs_identical(
    true_rc_adj_mat: np.ndarray,
    true_pc_adj_mat: np.ndarray,   
    pred_rc_adj_mat: np.ndarray,
    pred_pc_adj_mat: np.ndarray,
) -> bool:
    if (np.sum(np.abs(true_rc_adj_mat - pred_rc_adj_mat)) == 0 and np.sum(np.abs(true_pc_adj_mat - pred_pc_adj_mat)) == 0) or \
       (np.sum(np.abs(true_rc_adj_mat - pred_pc_adj_mat)) == 0 and np.sum(np.abs(true_pc_adj_mat - pred_rc_adj_mat)) == 0):
        return True
    else:
        return False

def check_reactant_product_graphs_threshold(
    true_rc_adj_mat: np.ndarray,
    true_pc_adj_mat: np.ndarray,   
    pred_rc_adj_mat: np.ndarray,
    pred_pc_adj_mat: np.ndarray,
    irc_end_graph_threshold: int,
) -> bool:
    if (np.sum(np.abs(true_rc_adj_mat - pred_rc_adj_mat)) == 0 and np.sum(np.abs(true_pc_adj_mat - pred_pc_adj_mat)) <= irc_end_graph_threshold) or \
       (np.sum(np.abs(true_rc_adj_mat - pred_rc_adj_mat)) <= irc_end_graph_threshold and np.sum(np.abs(true_pc_adj_mat - pred_pc_adj_mat)) == 0) or \
       (np.sum(np.abs(true_rc_adj_mat - pred_pc_adj_mat)) == 0 and np.sum(np.abs(true_pc_adj_mat - pred_rc_adj_mat)) <= irc_end_graph_threshold) or \
       (np.sum(np.abs(true_rc_adj_mat - pred_pc_adj_mat)) <= irc_end_graph_threshold and np.sum(np.abs(true_pc_adj_mat - pred_rc_adj_mat)) == 0):
        return True
    else:
        return False

def check_reaction_ends_by_graph_topology(
    true_rc_adj_mat: np.ndarray,
    true_pc_adj_mat: np.ndarray,   
    pred_rc_adj_mat: np.ndarray,
    pred_pc_adj_mat: np.ndarray,
    irc_end_graph_threshold: int = 0,
) -> bool:
    """
    Function to check whether the simulated reaction actually corresponds to 
    the intended reaction by directly comparing the graph topologies of true and
    predicted reactants / products.
    """

    print(np.sum(np.abs(true_rc_adj_mat - pred_rc_adj_mat)), np.sum(np.abs(true_pc_adj_mat - pred_pc_adj_mat)))
    print(np.sum(np.abs(true_rc_adj_mat - pred_pc_adj_mat)), np.sum(np.abs(true_pc_adj_mat - pred_rc_adj_mat)))

    if irc_end_graph_threshold == 0:
        return check_reactant_product_graphs_identical(
            true_rc_adj_mat,
            true_pc_adj_mat,
            pred_rc_adj_mat,
            pred_pc_adj_mat
        )
    elif irc_end_graph_threshold > 0:
        return (
            check_reactant_product_graphs_identical(
                true_rc_adj_mat,
                true_pc_adj_mat,
                pred_rc_adj_mat,
                pred_pc_adj_mat
            )
            or
            check_reactant_product_graphs_threshold(
                true_rc_adj_mat,
                true_pc_adj_mat,
                pred_rc_adj_mat,
                pred_pc_adj_mat,
                irc_end_graph_threshold
            )
        )
    else:
        raise ValueError("Invalid value for irc_end_graph_threshold")
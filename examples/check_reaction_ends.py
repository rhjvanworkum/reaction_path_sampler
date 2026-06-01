"""Check whether a simulated reaction matches the intended one.

This demonstrates the success/failure gate used at the end of every reaction
path search: given the intended reactant/product SMILES and the SMILES recovered
from an optimized path's endpoints, decide whether they describe the same
reaction (order-insensitive, and tolerant of reactant/product swaps).

Runnable as-is (only needs rdkit):

    uv run examples/check_reaction_ends.py
"""

from reaction_path_sampler.reaction_path.reaction_ends import check_reaction_ends_by_smiles

# A Diels-Alder-like intent: two reactants combine into one product.
intended_reactants = ["C=CC=C", "C=C"]
intended_products = ["C1=CCCCC1"]

cases = {
    "exact match": (["C=CC=C", "C=C"], ["C1=CCCCC1"]),
    "fragments reordered": (["C=C", "C=CC=C"], ["C1=CCCCC1"]),
    "reactants/products swapped": (["C1=CCCCC1"], ["C=CC=C", "C=C"]),
    "wrong product": (["C=CC=C", "C=C"], ["CCCCCC"]),
}

print(f"intended: {'.'.join(intended_reactants)} >> {'.'.join(intended_products)}\n")
for label, (pred_rc, pred_pc) in cases.items():
    matches = check_reaction_ends_by_smiles(
        list(intended_reactants), list(intended_products), pred_rc, pred_pc
    )
    verdict = "MATCH " if matches else "no    "
    print(f"  [{verdict}] {label}: {'.'.join(pred_rc)} >> {'.'.join(pred_pc)}")

#!/bin/bash

source env.sh

# srun python3 -u compute_da_regioselectivity.py
# python compute_da_regioselectivity_2.py
# python scripts/diels-alder/save_dataset.py
# python3 test_orca.py
# python -u search_rxn_path_2.py systems/smc_small.yaml
# python -u scripts/diels-alder/compute_results.py


python -u search_rxn_path.py systems/old/ac_tertiary.yaml
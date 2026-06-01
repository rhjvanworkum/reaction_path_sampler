# research/

One-off research scripts and exploratory notebooks used to produce and analyse
the datasets behind the project. **These are not part of the library**, are not
linted/type-checked/tested, and are kept here for reference and reproducibility.

- `scripts/<reaction>/` — per-reaction-family drivers:
  - `simulate_*.py` — SLURM launchers that fan a reaction list out to
    `sbatch`/`srun` jobs (require an HPC scheduler).
  - `parse_reaxys_reactions.py` / `create_*` — dataset generation from SMARTS /
    Reaxys exports.
  - `test.py` / `analyse_*.py` — ROC-AUC / accuracy analysis over the produced
    `barrier.txt` result trees.
- `notebooks/` — dataset exploration.

Known duplication (candidate for a future shared `research/_common.py`): the
`simulate_*` launcher boilerplate is repeated across reaction families, as is
the `barrier.txt`-tree analysis. These were intentionally left as-is in the
modernization pass because they depend on SLURM/external data and cannot be
verified here; consolidate them when working on that infrastructure.

Most of these reference the old top-level entry scripts; the supported entry
points are now the `search-rxn-path` / `search-rxn-path-from-template` console
commands (see the top-level README).

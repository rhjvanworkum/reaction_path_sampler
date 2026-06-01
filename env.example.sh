# Example environment for Reaction Path Sampler.
#
# Copy to `env.sh` (git-ignored) and edit the paths for your machine, then
# `source env.sh` before running the CLI. With the uv-managed install the
# package itself is already importable (no PYTHONPATH hack needed); these
# variables only point the code at the external QM binaries.

# Required: path to the xtb and crest executables.
export XTB_PATH="/path/to/xtb"
export CREST_PATH="/path/to/crest"

# Optional: ORCA (only needed for DFT barriers/refinement) and pysis must be on
# PATH. `pysis` ships with the pysisyphus pip dependency.
# export PATH="/path/to/orca:$PATH"

# Optional: logging verbosity (DEBUG/INFO/WARNING/ERROR/CRITICAL; default INFO).
export RPS_LOG_LEVEL="INFO"

# Reaction Path Sampler

Sample reaction paths and locate transition states directly from reactant and
product SMILES, using semi-empirical (xtb / CREST) and DFT (ORCA / PySCF)
quantum-chemistry backends.

Given the SMILES of the reactants and products, the tool maps the atoms between
them, samples reactant- and product-complex conformers, interpolates a reaction
path, optimizes a transition state and runs an IRC, and reports the barrier.

---

## Installation

The Python package and its dev tooling are managed with
[uv](https://docs.astral.sh/uv/). Three scientific libraries are **not on PyPI**
and a few **external binaries** are needed at runtime; these are installed
separately.

### 1. Python package (uv)

```bash
uv sync                 # core runtime + dev dependencies, into .venv
uv sync --extra qm      # also pyscf + rxnmapper (DFT / atom-mapping backends)
```

This installs everything that *is* on PyPI (rdkit, numpy/scipy, networkx,
pysisyphus — which provides the `pysis` CLI — etc.) and the package itself in
editable mode.

### 2. Non-PyPI Python dependencies

```bash
conda install -c conda-forge autode openbabel
# geodesic_interpolate: clone and install from source, then
#   uv pip install /path/to/geodesic_interpolate
```

`autode` can alternatively be installed from source with
`uv pip install "autode @ git+https://github.com/duartegroup/autodE.git"`.

### 3. External binaries

Make sure these are installed and on `PATH` / pointed at by env vars:

| Tool   | How it's found            | Needed for                       |
| ------ | ------------------------- | -------------------------------- |
| xtb    | `$XTB_PATH`               | semi-empirical energies/opt      |
| crest  | `$CREST_PATH`             | conformer ensemble pruning       |
| pysis  | on `PATH` (pip dep)       | NEB / TS-opt / IRC               |
| orca   | on `PATH` (optional)      | DFT barriers / refinement        |

Copy [`env.example.sh`](env.example.sh) to `env.sh`, edit the paths, and
`source env.sh` before running.

## Quickstart

Two examples run with no QM binaries at all:

```bash
uv run examples/check_reaction_ends.py      # reaction success/failure gate
uv run examples/bond_orders_from_xyz.py     # connectivity + bond orders
```

Run the test suite and the linters:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src
```

## Usage

The full reaction-path search is driven by a YAML settings file and exposed as a
console script:

```bash
source env.sh
uv run search-rxn-path systems/rps.yaml
```

`systems/rps.yaml` is a complete, commented settings file (reactant/product
SMILES, solvent, conformer-sampling thresholds, geodesic-interpolation and
TS-optimization parameters). A template-based variant is also available:

```bash
uv run search-rxn-path-from-template systems/ts_opt.yaml
```

See [`examples/`](examples/) for more, including the runnable building-block
demos above.

## Project structure

```
src/reaction_path_sampler/
  base.py                  ReactionSampler base class (settings + complex generation)
  cli.py                   console-script entry points
  molecular_system.py      MolecularSystem / Reaction (graph + atom mapping)
  molecule.py              lightweight Atom type + xyz parsing
  reaction_path_sampler.py ReactionPathSampler driver (path search, TS opt, IRC)
  template_sampler.py      TemplateSampler driver (TS templates)
  ts_template.py           TS template persistence
  utils.py                 geometry / trajectory / autodE helpers
  reaction_path/           complexes, atom-mapping, scoring, barrier, end checks
  conformational_sampling/ metadynamics + topology conformer samplers
  interfaces/              thin wrappers over xtb / crest / orca / pysis / pyscf
  graphs/                  vendored xyz2mol + Lewis-structure code
  visualization/           plotly graph rendering
tests/                     binary-free unit tests (QM-dependent ones auto-skip)
examples/                  runnable examples + bundled sample data
systems/                   example YAML run configurations
research/                  one-off research scripts and notebooks (unmaintained)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the dev setup, tests, and lint/format
workflow.

## License

MIT — see [LICENSE](LICENSE).

# Examples

## Runnable anywhere (no QM binaries needed)

These use only the pip-installable stack and bundled data:

```bash
uv run examples/check_reaction_ends.py     # the reaction success/failure gate
uv run examples/bond_orders_from_xyz.py    # connectivity + bond orders from a geometry
```

- [check_reaction_ends.py](check_reaction_ends.py) — decide whether a recovered
  reaction matches the intended one (order-insensitive, swap-tolerant).
- [bond_orders_from_xyz.py](bond_orders_from_xyz.py) — infer the adjacency and
  Lewis bond-order matrices from the bundled [data/ethene.xyz](data/ethene.xyz),
  recovering the C=C double bond.

## Full reaction-path search (requires the QM stack)

The end-to-end workflow needs autodE + openbabel and the external binaries
(xtb, crest, pysis); see the top-level README "Installation". Once those are
set up and `env.sh` is sourced:

```bash
uv run search-rxn-path systems/rps.yaml
```

`systems/rps.yaml` is a complete, commented settings file (reactant/product
SMILES, solvent, conformer-sampling and TS-search parameters). Copy it and edit
the SMILES / `output_dir` for your own reaction.

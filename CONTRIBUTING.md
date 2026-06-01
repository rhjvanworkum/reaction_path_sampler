# Contributing

## Development setup

```bash
uv sync                 # creates .venv with runtime + dev dependencies
```

The non-PyPI scientific dependencies (autode, openbabel, geodesic_interpolate)
and the QM binaries (xtb, crest, orca, pysis) are only needed to run the full
pipeline and its integration tests — see the README "Installation". The
binary-free unit tests, linters and type checker run without them.

## Tests

```bash
uv run pytest                         # unit tests (QM-dependent tests auto-skip)
RPS_RUN_INTEGRATION=1 uv run pytest    # also run tests marked `integration`
```

Tests that need autodE use `pytest.importorskip("autode")`, so they skip
cleanly where it is not installed. New tests should prefer the binary-free
modules (`reaction_path.reaction_ends`, `graphs.lewis`, `molecule`, scoring
helpers) and use `tmp_path`/fixtures rather than real files or network.

## Lint, format, types

```bash
uv run ruff check .            # lint
uv run ruff format .           # auto-format
uv run mypy src                # type check
```

CI runs all three plus the test suite on every push and PR. Keep `ruff check`,
`ruff format --check` and `mypy src` clean.

### Notes

- `src/reaction_path_sampler/graphs/{lewis,xyz2mol}.py` are vendored from
  upstream projects and are excluded from lint/type checks — don't reformat
  them.
- Several QM-pipeline modules are listed under a mypy `ignore_errors` override
  while their type debt is paid down; remove a module from that list once it
  type-checks cleanly.
- The package `__init__` is intentionally import-light: don't add heavy
  (autodE/openbabel) imports at module top level if they can be deferred into
  the functions that use them, so leaf modules stay importable for testing.

## Commits & PRs

Group related changes into focused commits with clear messages, branch off
`main`, and open a PR. Call out any change that could affect observable
behaviour of the QM pipeline.

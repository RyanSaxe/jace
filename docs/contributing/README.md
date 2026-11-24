# Contributing

## Setup and checks

```sh
uv sync
prek install               # mirrors .github/workflows/check.yml locally
uv run --locked pytest -q  # unit tests
prek run --all-files       # format, lint, typing, lockfile, boundary checks
```

`.pre-commit-config.yaml` runs Ruff, `ty`, `rumdl`, `uv lock --check`, the
1,000-line source cap, and `ast-grep` against `.byor/rules/project/`. Add new
boundary rules as YAML files in `.byor/rules/project/`.

## Design guidelines

- Follow the package boundaries in [Architecture](../concepts/architecture.md).
- Prefer functions and frozen dataclasses; add a class only for grouped fields
  or explicit state.
- Inject dependencies (`Callable` or `Protocol`) instead of module globals.
- Keep hot-path per-decision work batched in tensors or NumPy arrays.
- Record design-shaping papers, datasets, or repos in
  [Sources](../reference/sources.md) in the same commit.

## Tests

- Test through each module's public `__all__` exports.
- Use real tensors, small models, and synthetic `Trajectory` / `PolicyBatch`
  inputs instead of mocks.
- Keep unit tests fast and independent of 17Lands data.

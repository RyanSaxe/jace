# Contributing

## Setup

```sh
uv sync
prek install
```

The pre-commit hook runs every static check that `.github/workflows/check.yml`
runs in CI.

## Checks

```sh
uv run --locked pytest -q  # unit tests
prek run --all-files       # format, lint, typing, lockfile, and boundary checks
```

`.pre-commit-config.yaml` defines the repository checks, including Ruff (lint
and format), `ty`, `rumdl`, `uv lock --check`, the 1,000-line source-file cap,
and `ast-grep` against `.byor/rules/project/`. Add repository-specific AST
boundary rules as YAML files under `.byor/rules/project/`.

## Design guidelines

- Follow the package boundaries in [Architecture](../concepts/architecture.md).
- Prefer plain functions and frozen dataclasses over classes; add a class only
  when it holds multiple fields or owns explicit state.
- Pass dependencies (models, generators, collectors, loss functions) as
  arguments instead of module globals. Use `Callable` for single-method
  dependencies and `Protocol` for multi-method interfaces.
- Keep hot-path per-decision operations batched in tensors or NumPy arrays.
- Record every paper, dataset, or repository that changes the design in
  [Sources](../reference/sources.md) in the same commit.

## Tests

- Test behavior through each module's public exports (`__all__`).
- Use real tensors, small real models, and synthetic `Trajectory` /
  `PolicyBatch` inputs rather than mocks.
- Keep unit tests fast and independent of downloaded 17Lands data.

# Contributing

## Setup

```sh
uv sync
prek install
```

The pre-commit hook runs every static check that CI runs.

## Checks

```sh
uv run pytest -q       # unit tests
prek run --all-files   # lint, typing, and AST boundary checks
```

| Check                | Enforces                                                                               |
| -------------------- | -------------------------------------------------------------------------------------- |
| `ast-grep`           | Architecture boundaries and Python rules in `.byor/rules/project/`                     |
| `ruff`               | Full lint rule set (`select = ["ALL"]` minus explicit ignores in `pyproject.toml`), line length 120, formatting |
| `ty`                 | Strict type checking (`--error all`, ignoring only `unsound-return-statement`)         |
| Complexity           | At most 6 independent paths (`max-complexity = 6`) per function                        |
| File length          | At most 500 lines per Python file                                                      |
| Module contract      | Module docstring (summary line + paragraph) and static `__all__` in every module       |
| Thin docstrings      | No class or function docstring of three lines or fewer                                 |
| Suppression comments | No `noqa`, `type: ignore`, or similar check-silencing comments                         |

[byor](https://github.com/RyanSaxe/byor) manages these hooks. Add
repository-specific AST checks as YAML rules under `.byor/rules/project/`.

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

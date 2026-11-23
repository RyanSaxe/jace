# Contributing

## Setup

```sh
uv sync
prek install
```

The pre-commit hook runs every check that CI runs, except the test suite.

## Checks

```sh
uv run pytest -q       # unit tests
prek run --all-files   # all static, type, and AST boundary checks
```

| Check                | Enforces                                                                               |
| -------------------- | -------------------------------------------------------------------------------------- |
| `ast-grep`           | Project architecture boundaries and Python style rules in `.byor/rules/project/`       |
| `ruff`               | Full lint rule set (`select = ["ALL"]` minus explicit ignores in `pyproject.toml`), line length 120, formatting |
| `ty`                 | Strict type checking (`--error all`, ignoring only `unsound-return-statement`)         |
| Complexity           | At most 6 independent paths (`max-complexity = 6`) per function                        |
| File length          | At most 500 lines per Python file                                                      |
| Module contract      | Module docstring (summary line + paragraph) and static `__all__` in every module       |
| Thin docstrings      | No class or function docstring of three lines or fewer                                 |
| Suppression comments | No `noqa`, `type: ignore`, or similar check-silencing comments                         |

[byor](https://github.com/RyanSaxe/byor) manages these hooks. To add a repository-specific AST check, add a YAML rule under `.byor/rules/project/`.

## Architecture boundaries

Six `ast-grep` rules protect the package graph (see [Architecture](../concepts/architecture.md) for the diagram):

- `jace.no-manabot-learner-recipes`: never import `manabot.training.updates`, `manabot.training.loops`, or `manabot.recipes`.
- `jace.manabot-import-boundaries`: only `jace.data.catalog`, `jace.rl.loop`, `jace.eval`, `jace.runs`, and `jace.loops` may import `manabot` or `managym`.
- `jace.pure-rl-math-boundaries`: `jace.rl.credit`, `jace.rl.losses`, and `jace.rl.select` stay pure PyTorch/NumPy math.
- `jace.leaf-packages-stay-leaf`: `jace.cards`, `jace.data`, `jace.modules`, and `jace.tracking` never import higher-level packages.
- `jace.only-loops-import-draft`: only `jace.loops` imports `jace.draft`.
- `jace.nothing-imports-loops`: nothing outside `jace.loops` imports `jace.loops`.

## Design guidelines

- Prefer plain functions and frozen dataclasses over classes. Add a class only when it holds multiple fields or owns explicit state.
- Pass dependencies (models, random generators, collectors, loss functions) as arguments instead of creating hidden globals. Use `Callable` for single-method dependencies and `Protocol` when callers invoke multiple methods.
- Keep hot-path per-decision operations batched in tensors or NumPy arrays.
- Record every paper, dataset, or repository that influences a design change in [Sources](../reference/sources.md) in the same commit.

## Tests

- Test behavior that matters through each module's public exports (`__all__`).
- Use real tensors, small real models, and synthetic `Trajectory`/`PolicyBatch` inputs rather than mocks.
- Keep unit tests fast and independent of downloaded 17Lands data.

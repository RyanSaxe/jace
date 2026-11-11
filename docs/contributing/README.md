# Contributing

## Set up

```sh
uv sync
uvx pre-commit install
```

The pre-commit hook runs the same checks as CI.

## Checks

```sh
uv run pytest -q                   # the tests
uvx pre-commit run --all-files     # every check
```

| Check             | Enforces                                                       |
| ----------------- | -------------------------------------------------------------- |
| ruff              | Every rule, at line length 120, and formatting                 |
| ty                | Types, with every finding an error                             |
| ast-grep rules    | The style rules in `.byor/rules/project/`                      |
| Complexity        | At most six independent paths through a function               |
| File length       | At most 500 lines in a Python file                             |
| Module contract   | A docstring and a static `__all__` in every module             |

Two of the ast-grep rules protect the module boundaries:

- Only `jace/engine/manabot.py` imports manabot.
- Only `jace.loops` imports `jace.draft`.

The checks are managed with [byor](https://github.com/RyanSaxe/byor). To add
a rule for this repository, add a file under `.byor/rules/project/`.

## Design rules

- Start from functions and plain values. Add a type only when it has
  several fields or states a rule the type checker should enforce.
- Never wrap a single value in a class.
- A dependency that is one behaviour is a callable type. A protocol is for
  an injected object whose user needs several methods.
- Hand a function what it needs. It does not reach for a model, a random
  generator or the engine.
- Write a state machine only when one exists: one immutable type per state
  and one function per transition.
- A long function that reads top to bottom is fine. Many one-line functions
  are worse.
- Anything that runs once per game decision is a batched tensor operation.

## Tests

Tests are few and deliberate.

- Test behaviour that matters, through a module's public names.
- Use real objects and small real models. No mocks.
- A refactor that keeps behaviour should leave every test unchanged.

`jace.engine.toy` exists so that training code can be tested without an
engine.

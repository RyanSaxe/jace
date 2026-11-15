# Contributing

## Set up

```sh
uv sync
uvx pre-commit install
```

The pre-commit hook runs every check that CI runs, except the tests.

## Checks

```sh
uv run pytest -q                   # the tests
uvx pre-commit run --all-files     # every other check
```

| Check                | Enforces                                                                  |
| -------------------- | ------------------------------------------------------------------------- |
| ruff                 | Every rule except those under `ignore` in `pyproject.toml`, line length 120, and formatting |
| ty                   | Types. Every finding is an error, and only `unsound-return-statement` is turned off |
| ast-grep rules       | The style rules in `.byor/rules/project/`                                 |
| Complexity           | At most six independent paths through a function                         |
| File length          | At most 500 lines in a Python file                                        |
| Module contract      | A docstring with a summary line and a paragraph, and a static `__all__`, in every module |
| Thin docstrings      | No class or function docstring of three lines or fewer                   |
| Suppression comments | No comment that silences a check, such as `noqa` or `type: ignore`        |

Two of the ast-grep rules enforce the module boundaries:

- Only `jace/engine/manabot.py` and `jace/engine/scoring.py` import manabot
  or managym, so a change in the engine affects only those two files.
- Outside `jace.draft` itself, only `jace.loops` imports `jace.draft`, so
  constructed play runs without any drafting code.

[byor](https://github.com/RyanSaxe/byor) manages the checks. To add an
ast-grep rule for this repository, add a rule file under
`.byor/rules/project/`.

## Design rules

- Start from functions and plain values. Add a type only when it has
  several fields or lets the type checker enforce a rule.
- Never wrap a single value in a class.
- Give a dependency with one behaviour a callable type. Use a protocol for
  an injected object whose user calls several of its methods.
- Pass a function the model, random generator or engine it uses. It does
  not fetch or create its own.
- Write a state machine only when the behaviour has distinct states. Give
  it one immutable type per state and one function per transition.
- A long function that reads top to bottom is fine. Many one-line functions
  are worse.
- Anything that runs once per game decision is a batched tensor operation.

## Tests

Tests are few and deliberate.

- Test behaviour that matters, through a module's public names.
- Use real objects and small real models. No mocks.
- A refactor that does not change behaviour should leave every test unchanged.

`jace.engine.toy` exists so that training code can be tested without an
engine.

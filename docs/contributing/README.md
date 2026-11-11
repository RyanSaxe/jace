# Contributing

This page explains how to set up a checkout, which checks a change must
pass and how to run them, where things live, the design rules, the testing
rules, and how to add a lint rule.

## Set up

jace needs Python 3.12, [uv](https://docs.astral.sh/uv/) and a Rust
toolchain. [Install](../getting-started/install.md) explains why.

```sh
uv sync
uv run pytest -q
uvx pre-commit install
```

`uvx pre-commit install` adds a git hook that runs the checks below before
each commit.

## Checks

A change must pass the tests and the gate. The gate is the set of lint and
type checks that the pre-commit hook and the `byor gate` workflow both run.
The `tests` workflow runs pytest. Both workflows run on every pull request
and on every push to `main`.

```sh
uv run pytest -q                 # the tests
uvx pre-commit run --all-files   # the gate, over every file
byor agent-check                 # the same rules and checks, with how to fix each finding
```

`byor agent-check` needs byor installed. The other two commands need only
uv.

| Check                     | What it enforces                                                                                                   |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| ast-grep                  | The rules under `.byor/rules/project/`: byor's style rules for Python, and the two import rules of this repository. |
| ruff                      | Every ruff rule, less the few that `pyproject.toml` ignores with a reason, at a line length of 120. It also formats. |
| ty                        | The type checker, with every finding an error.                                                                     |
| `no-suppression-comments` | No comment that silences a linter or the type checker, such as `noqa`.                                             |
| `module-contract`         | Every module has a docstring with a summary line and a paragraph specific to it, and a static `__all__`.           |
| `no-thin-docstrings`      | No thin docstring on a function or class. A docstring stays only where it documents real arguments and behaviour.  |
| `max-file-lines`          | No Python file is longer than 500 lines.                                                                           |

Ruff also limits a function's cyclomatic complexity to 6. A function with
more independent paths than that is split into named helpers.

The ruff check fixes what it can and reformats the file, then reports what
is left. Fix a finding in the code. The gate rejects a suppression comment.

## Layout

| Path                    | Contents                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `src/jace/`             | The package. [Architecture](../concepts/architecture.md#modules) lists its modules.        |
| `tests/`                | One test file per module or seam, and `tests/data/` with three real engine boards.         |
| `docs/`                 | This documentation.                                                                        |
| `.byor/config.yml`      | The checks the gate runs.                                                                  |
| `.byor/rules/project/`  | The ast-grep rules, one YAML file each.                                                    |
| `.byor/scripts/`        | The scripts behind the checks. byor manages all of them except `max-file-lines.py`.        |
| `.github/workflows/`    | The `byor gate` and `tests` workflows.                                                     |
| `.pre-commit-config.yaml` | The hook's configuration, which byor writes.                                             |
| `pyproject.toml`        | Dependencies, the pinned manabot commit, and the ruff, ty and pytest settings.             |

## Design rules

jace is designed as pieces, and a caller writes a short function from them.
These rules keep the pieces small and separate.

- **Start from functions and plain values.** A tensor, a tuple of card ids
  or a mapping is the default.
- **Add a type only when it earns one.** A type has several fields, or it
  states a rule the type checker should enforce. `Running` in
  `jace.tracking` is the second kind: only `writing` makes one, so a write
  outside a `writing` block fails the type check.
- **Never wrap one value in a class.** `CardId` is an `int` and `Pool` is a
  tuple of them.
- **A dependency that is one behaviour is a callable type.** `Policy`,
  `Guesser`, `Drafter`, `Proposer`, `Scorer` and `Schedule` are all function
  types.
- **A protocol is for an injected object whose user needs several
  methods.** `GameBatch` is the one protocol.
- **A real state machine is explicit.** It has one immutable type per state
  and one function per transition, as the draft pod does with `Picking`,
  `Complete` and `advance`. Do not invent one for a plain sequence of calls.
  A sequence of calls is a function.
- **Hand dependencies in as arguments.** `collect` takes its policies and
  its random generator, and `jace.tracking` takes a clock and never reads
  the system clock.
- **Long linear functions are fine.** A pile of one-line functions is
  worse. Split a function when it holds two ideas or passes the complexity
  limit.
- **A module's public interface is its `__all__`.** Names with a leading
  underscore are private to their file.

Two import rules guard the module boundaries: only
`src/jace/engine/manabot.py` imports manabot, and only `jace.loops` imports
`jace.draft`. [Architecture](../concepts/architecture.md#two-import-rules)
explains why.

## Testing rules

- **Tests are few.** Each one protects a behaviour that matters, and its
  name states that behaviour as a sentence.
- **Tests go through public names only.** A test imports from a module's
  `__all__`, so the code behind it can change without touching the test.
- **Tests use no mocks.** A function that needs a policy gets a small real
  function, and a function that needs games gets `ToyBatch` or the real
  engine.
- **Models in tests are tiny and real.** One or two layers at a small width
  run the same code as a large model.

Run one file while you work, then the whole suite:

```sh
uv run pytest -q tests/test_pod.py
uv run pytest -q
```

## Add a lint rule

A repository rule is one YAML file under `.byor/rules/project/`, written for
[ast-grep](https://ast-grep.github.io/). The hook and CI pick it up with no
other change, because `sgconfig.yml` names that folder.

Start from one of the two rules this repository wrote:

- [`jace.only-engine-adapter-imports-manabot.yml`](../../.byor/rules/project/jace.only-engine-adapter-imports-manabot.yml)
  rejects an import of manabot outside the adapter.
- [`jace.only-loops-import-draft.yml`](../../.byor/rules/project/jace.only-loops-import-draft.yml)
  rejects an import of `jace.draft` outside `jace.loops`.

Each file has the same parts:

| Key                          | Holds                                                                      |
| ---------------------------- | -------------------------------------------------------------------------- |
| `id`                         | The rule's name, which is also the file's name. Both start with `jace.`.   |
| `severity`, `message`        | The severity, and the line a person sees.                                  |
| `files`, `ignores`           | The paths the rule covers, and the paths it exempts.                       |
| `rule`                       | The ast-grep pattern that matches the code to reject.                      |
| `metadata.byor.rationale`    | Why the rule exists.                                                       |
| `metadata.byor.agent_prompt` | What to do in place of the rejected code.                                  |

Then run the rule over the repository:

```sh
uvx --from ast-grep-cli==0.42.3 ast-grep scan --error
```

A new rule should pass on the code as it is, and fail on a line you add to
break it.

# Install

This page lists what jace needs, and explains how to install it, run the
tests, run the example and install the pre-commit hook.

## Requirements

| You need                         | Why                                                                         |
| -------------------------------- | --------------------------------------------------------------------------- |
| Python 3.12                      | manabot supports only 3.12, so `pyproject.toml` accepts no other version.   |
| [uv](https://docs.astral.sh/uv/) | It creates the environment and runs every command on this page.             |
| A Rust toolchain                 | Installing jace builds manabot's game engine from source.                   |

## Install

```sh
git clone https://github.com/RyanSaxe/jace.git
cd jace
uv sync
```

`uv sync` creates `.venv/` and installs jace with PyTorch, NumPy, manabot
and the development tools: pytest, ruff and ty. manabot comes from GitHub at
one pinned commit, which `pyproject.toml` names under `[tool.uv.sources]`.

## Run the tests

```sh
uv run pytest -q
```

```text
.............................................                            [100%]
45 passed in 2.95s
```

The tests play real Magic games on manabot's engine, so a pass also shows
that the engine was built and loads.

## Run the example

```sh
uv run python -m jace.loops.toy
```

`jace.loops.toy` is the smallest of the example loops in `jace.loops`. It
trains a small player on High Card, the toy game in `jace.engine.toy`. Each
seat holds three cards with distinct values and picks
one, and the higher pick wins. The best play is always the highest card, so
the share of decisions where the player picks it shows whether training
works.

```text
iteration  1  best card 0.36  win rate 0.56
iteration  5  best card 0.41  win rate 0.53
iteration 10  best card 0.44  win rate 0.58
iteration 15  best card 0.51  win rate 0.59
iteration 20  best card 0.59  win rate 0.60
iteration 25  best card 0.65  win rate 0.69
iteration 30  best card 0.69  win rate 0.74
iteration 35  best card 0.69  win rate 0.72
iteration 40  best card 0.78  win rate 0.74
```

A player that picks at random picks its best card a third of the time. The
opponent in the example picks at random throughout. The seeds are fixed, so
a second run on the same machine prints the same numbers.

The loop calls `collect` and `update_player` from `jace.train`, the same
functions a loop on real games calls. [Training](../concepts/training.md)
explains both.

## Install the pre-commit hook

To change jace, install the hook once in your clone:

```sh
uvx pre-commit install
```

The hook then runs the lint and type checks before each commit.
[Contributing](../contributing/README.md#checks) lists the checks and
explains how to run them yourself.

# Install

## Requirements

| Need             | Why                                                       |
| ---------------- | --------------------------------------------------------- |
| Python 3.13      | The one version `pyproject.toml` allows                   |
| [uv](https://docs.astral.sh/uv/) | It installs jace and runs every command   |
| A Rust toolchain | Only where the engine, managym 0.2.3, has no built package for your platform |

## Install and check

```sh
git clone https://github.com/RyanSaxe/jace.git
cd jace
uv sync
uv run pytest -q
```

`uv sync` installs managym and manabot 0.2.3. Where it has to compile the
engine, that takes a few minutes. The tests then run in under a minute.

## Run the example

```sh
uv run python -m jace.loops.toy
```

It trains a player on High Card, a game where each seat picks one of three
cards and the higher card wins. It needs no game engine. The share of
decisions where the player picks its best card rises as it trains:

```text
iteration  1  best card 0.36  win rate 0.56
iteration 20  best card 0.59  win rate 0.60
iteration 40  best card 0.78  win rate 0.74
```

## Next

Read [The approach](../concepts/approach.md), then
[Write a loop](../guides/write-a-loop.md).

# Install

## Requirements

| Need             | Why                                                       |
| ---------------- | --------------------------------------------------------- |
| Python 3.12      | manabot supports only 3.12                                |
| [uv](https://docs.astral.sh/uv/) | It installs jace and runs every command   |
| A Rust toolchain | Installing builds manabot's engine from source            |

## Install and check

```sh
git clone https://github.com/RyanSaxe/jace.git
cd jace
uv sync
uv run pytest -q
```

The first `uv sync` compiles manabot's engine, which takes a few minutes.
The tests then run in about three seconds, and 45 should pass.

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

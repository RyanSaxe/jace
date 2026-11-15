# Install

## Requirements

| Need             | Why                                                       |
| ---------------- | --------------------------------------------------------- |
| Python 3.13      | The one version `pyproject.toml` allows                   |
| [uv](https://docs.astral.sh/uv/) | It installs jace and runs every command   |
| A Rust toolchain | Only to compile the engine, managym 0.2.3, on a platform with no prebuilt managym package. Prebuilt packages exist for macOS on Apple silicon and Linux on x86-64 |

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

It trains a player on High Card, a toy game that runs without the engine.
Each seat holds three cards and plays one, and the higher card wins. In
each line the example prints, `best card` is the share of decisions in
which the player played its highest card. It rises as the player trains:

```text
iteration  1  best card 0.34  win rate 0.49
iteration 20  best card 0.61  win rate 0.67
iteration 40  best card 0.81  win rate 0.76
```

## Next

Read [The approach](../concepts/approach.md), then
[Write a loop](../guides/write-a-loop.md).

# Install

## Requirements

| Tool                             | Purpose                                                |
| -------------------------------- | ------------------------------------------------------ |
| Python 3.13                      | `>=3.13,<3.14` (`pyproject.toml`)                      |
| [uv](https://docs.astral.sh/uv/) | Dependency install and command runner                  |
| Rust toolchain                   | Compiles pinned `managym[academy]` from source on sync |

## Setup

```sh
git clone https://github.com/RyanSaxe/jace.git
cd jace
uv sync           # compiles managym[academy] (a few minutes first time)
uv run pytest -q  # runs offline, no external data needed
```

## 10-deck smoke run

Train on the 10 built-in `elemental-world-v2` decks:

```sh
uv run python -m jace.train.ten_decks \
  --run-id smoke \
  --updates 2 \
  --num-envs 8 \
  --steps-per-env 32 \
  --eval-every 2 \
  --eval-deals 4 \
  --probe-count 8 \
  --learn-device cpu
```

Writes to `../jace-runs/v2-ten-deck-lab/smoke/` (`--runs-dir` overrides).

## 17Lands data

`jace.train.full_data`, `jace.pretrain.draft`, and `jace.pretrain.decks` read
17Lands exports and MTGJSON attributes via
[spells](https://github.com/oelarnes/spells):

```sh
uv run spells add TLA
```

Stored in `~/.local/share/spells` (or `$SPELLS_DATA_HOME`); jace never downloads
automatically. Next: [Data](../reference/data.md),
[Architecture](../concepts/architecture.md), or
[Running experiments](../guides/running-experiments.md).

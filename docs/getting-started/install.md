# Install

## Requirements

| Requirement                      | Purpose                                                                      |
| -------------------------------- | ---------------------------------------------------------------------------- |
| Python 3.13                      | Required by `pyproject.toml` (`>=3.13,<3.14`)                                |
| [uv](https://docs.astral.sh/uv/) | Installs dependencies and runs every command                                 |
| Rust toolchain                   | Required by `uv sync` to compile the pinned `managym` engine from source     |

## Install and verify

```sh
git clone https://github.com/RyanSaxe/jace.git
cd jace
uv sync
uv run pytest -q
```

`uv sync` compiles `managym[academy]` from the pinned git revision in
`pyproject.toml` (a few minutes on first build). All unit tests run without
external datasets.

## Quick 10-deck lab run

`jace.train.ten_decks` runs closed-decklist self-play on the 10 built-in
`elemental-world-v2` decks bundled with `managym`:

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

Artifacts are written to `../jace-runs/v2-ten-deck-lab/smoke/` (override with
`--runs-dir`).

## Download 17Lands data

Full-data self-play (`jace.train.full_data`) and supervised pretraining
(`jace.pretrain.draft`, `jace.pretrain.decks`) read public 17Lands Premier Draft
exports and MTGJSON attributes via [spells](https://github.com/oelarnes/spells):

```sh
uv run spells add TLA
```

Files are stored in `~/.local/share/spells` (or `$SPELLS_DATA_HOME`). jace never
downloads datasets automatically. See [Data](../reference/data.md) for caching
and deterministic splits.

## Next

Read [Architecture](../concepts/architecture.md) for the system design, or
[Running experiments](../guides/running-experiments.md) for training,
pretraining, screening, and Academy replays.

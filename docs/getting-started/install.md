# Install

## Requirements

| Requirement                      | Purpose                                                                                             |
| -------------------------------- | --------------------------------------------------------------------------------------------------- |
| Python 3.13                      | Required by `pyproject.toml` (`>=3.13,<3.14`)                                                       |
| [uv](https://docs.astral.sh/uv/) | Installs dependencies and runs every command                                                        |
| Rust toolchain                   | Required by `uv sync` to compile the git-pinned `managym` engine from source                        |

## Install and verify

```sh
git clone https://github.com/RyanSaxe/jace.git
cd jace
uv sync
uv run pytest -q
```

`uv sync` installs `manabot` and compiles `managym[academy]` from the pinned git revision in `pyproject.toml` (a few minutes on first build). The unit test suite runs without external datasets.

## Run the 10-deck lab without external data

`jace.loops.train_ten_decks` trains on the 10 closed-decklist `elemental-world-v2` decks bundled with `managym`:

```sh
uv run python -m jace.loops.train_ten_decks \
  --run-id smoke \
  --updates 2 \
  --num-envs 8 \
  --steps-per-env 32 \
  --eval-every 2 \
  --eval-deals 4 \
  --probe-count 8 \
  --learn-device cpu
```

The run writes checkpoints, metrics, forensics, and Academy replay files to `../jace-runs/v2-ten-deck-lab/smoke/`. Pass `--runs-dir` to choose another root folder.

## Download 17Lands data (for full-data training and pretraining)

Full-data gameplay training (`jace.loops.train_full_data`) and Stage B pretraining (`jace.loops.pretrain`, `jace.loops.pretrain_decks`) read public 17Lands Premier Draft exports and MTGJSON card attributes downloaded through [spells](https://github.com/oelarnes/spells):

```sh
uv run spells add TLA
```

Avatar: The Last Airbender (`TLA`) takes about 300 MB in `~/.local/share/spells` (or `$SPELLS_DATA_HOME`). `jace` never downloads datasets automatically. See [Data](../reference/data.md) for how `jace.data` caches `decks.pt` and `drafts.pt` from those tables.

## Next

Read [Architecture](../concepts/architecture.md) for the package layout, or go straight to [Running experiments](../guides/running-experiments.md) for full-data training, Academy replays, and checkpoint screening.

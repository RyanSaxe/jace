# Contributing

## Setup and checks

```sh
uv sync
prek install               # install local git hooks matching CI
uv run --locked pytest -q  # unit tests
prek run --all-files       # run all repository checks
```

## Package layout

See [Architecture](../concepts/architecture.md) for the solution design and why
each seam exists.

| Package / path | Owns |
| --- | --- |
| `jace.train` | Gameplay self-play CLIs (`full_data`, `ten_decks`, `harness`) |
| `jace.pretrain` | Pretraining CLIs and helpers (`draft`, `decks`, `common`, `formats`, `schedules`, `supervised`, `deck_measures`) |
| `jace.forks` | Counterfactual forks, batched playouts, horizon audits (`collect`, `batch`, `audit`) |
| `jace.eval` | Mirrored deals, adaptive judging, forensics, probe diffs (`deals`, `parallel`, `forensics`, `diffs`) |
| `jace.collect` | v11 `ActorCritic` factory, deck/curriculum loading, collectors (`agent`, `decks`, `local`, `pool`) |
| `jace.rl` | Pure-tensor RL math (`credit`, `losses`, `select`, `schedules`, `ema`, `pool`, `prefetch`, `diagnostics`) and `loop` |
| `jace.draft` | Booster specs, multi-pod draft state machine, `DrafterModel` |
| `jace.decks` | `DeckBuilder`, `DeckScore`, `BuildRules`, batch-sampled deck search |
| `jace.storage` | Gameplay folders (`gameplay.py`) and locked, SHA-256 verified pretraining records (`pretrain.py`) |
| `jace.cards` | `CardPool`, program-based `CardEncoder`, frozen `CardTable` |
| `jace.data` | 17Lands/MTGJSON loaders, CRC-32 splits, program tokens, `catalog` seam |
| `jace.modules` | Shared transformer builder, packed `encode_real`, device selection |
| `scripts/` | Standalone PEP 723 scripts (`colab_run.py`) |

## Entry points

| Command | Purpose |
| --- | --- |
| `uv run python -m jace.train.ten_decks` | 10-deck self-play lab with optional forks |
| `uv run python -m jace.train.full_data` | Full 17Lands self-play with curricula, snapshots, resume |
| `uv run python -m jace.train.harness` | Warm-start screens and `--eval-only` adaptive head-to-head |
| `uv run python -m jace.pretrain.draft` | Pretrain `DrafterModel` and `CardEncoder` |
| `uv run python -m jace.pretrain.decks` | Pretrain `DeckScore` and `DeckBuilder` |
| `uv run scripts/colab_run.py` | Provision, sync, and stop Colab GPU training sessions |

## State machines

| Component | States | Owner |
| --- | --- | --- |
| Draft pod | `Picking` -> `Complete` | `jace.draft.pod.advance` |
| Pretraining run | `created` -> `running` -> `completed` / `failed` / `interrupted` | `jace.storage.pretrain.writing` |
| Historical pool | Bounded FIFO ring of CPU snapshots + EMA win rates | `jace.rl.pool.SnapshotPool` |
| Worker pool | Persistent spawned CPU workers (`req_queues` / `res_queue`) | `jace.collect.pool.MultiProcessCollector` |

## Design guidelines

- Follow the architectural seams in [Architecture](../concepts/architecture.md).
- Prefer functions and frozen dataclasses; add a class only for grouped fields
  or explicit state.
- Inject dependencies (`Callable` or `Protocol`) instead of module globals.
- Keep hot-path per-decision work batched in tensors or NumPy arrays.
- Record design-shaping papers, datasets, or repos in
  [Sources](../reference/sources.md) in the same commit.

## Tests

- Test through each module's public `__all__` exports.
- Use real tensors, small models, and synthetic `Trajectory` / `PolicyBatch`
  inputs instead of mocks.
- Keep unit tests fast and independent of 17Lands data.

# Architecture

jace splits Limited Magic learning into small packages plus standalone scripts
in `scripts/`, governed by a strict DAG.

## Data and model flow

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/architecture-flow-dark.svg">
  <img alt="End-to-end data and model flow across jace packages." src="../assets/architecture-flow-light.svg">
</picture>

## Package layers

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/modules-dark.svg">
  <img alt="Package layers of jace and the managym/manabot import seam." src="../assets/modules-light.svg">
</picture>

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

## Import boundaries

`.byor/rules/project/` enforces these `ast-grep` rules on every commit and CI:

| Rule | Contract |
| --- | --- |
| `jace.no-manabot-learner-recipes` | All RL updates, schedules, and loops stay in `jace.rl` |
| `jace.manabot-import-boundaries` | Only `data.catalog`, `rl.loop`, `collect`, `eval`, `forks`, `storage.gameplay`, `train` touch the engine |
| `jace.pure-rl-math-boundaries` | `jace.rl` (except `__init__`, `loop`) stays pure PyTorch/NumPy and never imports `rl.loop` |
| `jace.collect-stays-below-eval` | `jace.collect` never imports `eval`, `forks`, `pretrain`, `train` |
| `jace.leaf-packages-stay-leaf` | `cards`, `data`, `modules`, `storage.__init__`, `storage.pretrain` never import higher layers |
| `jace.only-entry-points-import-draft` | Only `draft`, `pretrain`, `train` import `jace.draft` |
| `jace.nothing-imports-pretrain` | Nothing outside `jace.pretrain` imports `jace.pretrain` |
| `jace.nothing-imports-train` | Nothing outside `jace.train` imports `jace.train` |

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

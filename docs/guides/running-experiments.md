# Running experiments

Every training and evaluation loop runs through `uv run python -m <module>`. Use `--help` on any module for its full flag list and defaults.

## 10-deck self-play lab

`jace.loops.train_ten_decks` trains on the 10 built-in closed-decklist `elemental-world-v2` decks with no external data:

```sh
uv run python -m jace.loops.train_ten_decks \
  --run-id lab-baseline \
  --updates 150 \
  --num-envs 32 \
  --collect-workers 8 \
  --eval-every 25
```

Runs write checkpoints, metrics, forensics, and Academy replays to `<runs-dir>/v2-ten-deck-lab/<run-id>/` (default `--runs-dir ../jace-runs`).

## Full-data 17Lands self-play

`jace.loops.train_full_data` trains across all 91,359 17Lands Avatar: The Last Airbender decks in `decks.pt`, evaluates every `--eval-every` updates on 200 mirrored games (`--eval-deals 100`) drawn strictly from the 10,137 held-out human decks, and writes atomic resume checkpoints:

```sh
uv run python -m jace.loops.train_full_data \
  --run-id full-tla-v1 \
  --decks-path ../jace-runs/data/decks.pt \
  --updates 1800 \
  --curriculum-stages 90:0,180:600,347:1200 \
  --curriculum-blend-updates 10 \
  --collect-workers 8 \
  --eval-workers 8
```

To continue or recover an interrupted run from its latest atomic `resume-state.pt` (restoring policy weights, Adam optimizer state, and `SnapshotPool`), pass `--resume` with the same `--run-id`:

```sh
uv run python -m jace.loops.train_full_data \
  --run-id full-tla-v1 \
  --decks-path ../jace-runs/data/decks.pt \
  --updates 3800 \
  --resume
```

To start a fresh run from an earlier `.safetensors` checkpoint with a new schedule, pass `--init-checkpoint <path>` and a new `--run-id`.

## Fast experiment screening and head-to-head evaluation

`jace.loops.harness` runs a short warm-start screen (default 25 updates) or scores two existing checkpoints with adaptive mirrored evaluation (`100 -> 200 -> 500` deals) and prints a JSON verdict (`PROMOTE`, `KILL`, or `INCONCLUSIVE`):

```sh
# Warm-start screen with counterfactual forks on declare_attacker
uv run python -m jace.loops.harness \
  --run-id fork-screen \
  --init-checkpoint ../jace-runs/v2-full-data/full-tla-v1/checkpoints/iter-1800.safetensors \
  --forks-per-update 32 \
  --fork-kinds declare_attacker

# Standalone adaptive head-to-head check between two checkpoints
uv run python -m jace.loops.harness \
  --eval-only \
  --candidate path/to/candidate.safetensors \
  --baseline path/to/baseline.safetensors
```

## Playing and replaying checkpoints in Academy

Every saved Stage A checkpoint includes a companion `.policy.json` file and native `managym` JSONL game recordings. `jace.runs.academy_commands` records the exact commands in each `forensics/iter-XXXX.json`:

```sh
# Play interactively against a saved checkpoint
uv run managym academy play \
  --policy ../jace-runs/v2-full-data/full-tla-v1/checkpoints/iter-1800.policy.json \
  --opponent iter-1800

# Inspect recorded evaluation games turn by turn
uv run managym academy replay \
  ../jace-runs/v2-full-data/full-tla-v1/games/eval-iter-1800.jsonl
```

## Pretraining Stage B models on 17Lands

First download Avatar: The Last Airbender data with `uv run spells add TLA`, then pretrain the drafter (and shared `CardEncoder`) followed by the deck score and deck builder:

```sh
# 1. Pretrain drafter + CardEncoder (also caches drafts.pt and text vectors in --data-dir)
uv run python -m jace.loops.pretrain \
  --data-dir ../jace-runs/data \
  --run-dir ../jace-runs/pretrain \
  --set TLA

# 2. Pretrain DeckScore + DeckBuilder on frozen card vectors (also caches decks.pt)
uv run python -m jace.loops.pretrain_decks \
  --data-dir ../jace-runs/data \
  --run-dir ../jace-runs/pretrain \
  --embedding-run ../jace-runs/pretrain/<drafter-run-id> \
  --set TLA
```

## Composing custom RL loops

`jace.rl` exposes plain composable callables rather than a monolithic trainer class:

| Extension point | Signature / protocol                                         | Built-in implementations                          |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------- |
| Credit          | `CreditFn: (Trajectory) -> CreditTargets`                    | `gae_wdl`, `monte_carlo`                          |
| Row selection   | `SelectFn: (Tensor) -> Tensor`                               | `top_fraction`, `all_rows`                        |
| Policy/value loss | `LossFn: (PolicyBatch) -> LossBreakdown`                   | `damped_policy_loss`, `ppo_loss`                  |
| Schedule        | `LossScheduleFn: (int) -> tuple[float, float, LossFn]`       | `DecaySchedule`, `DampedSchedule`                 |
| Collection      | `CollectFn(agent, *, opponent_entry, update_idx) -> RolloutBatch` | Local collector or `MultiProcessCollector.collect` |
| Auxiliary loss  | `PrepareAuxFn: (ActorCritic, int) -> AuxLossFn | None`       | `ForkCollectorPool`, `build_batched_fork_aux_loss` |

Pass any combination directly to `train_self_play` in `jace.rl.loop`:

```python
from jace.rl import TrainConfig, UpdateConfig, damped_policy_loss, gae_wdl, top_fraction, train_self_play

history = train_self_play(
    agent,
    optimizer,
    obs_space=obs_space,
    deck_source=deck_source,
    credit=gae_wdl(gamma=1.0, lam_policy=0.5, lam_value=0.8),
    select=top_fraction(0.5, minimum=0.01),
    loss=damped_policy_loss(clip=0.2, collection_kl=0.1, magnet_kl=0.01, value_coef=1.0),
    config=TrainConfig(updates=50, update=UpdateConfig(device="cpu")),
)
```

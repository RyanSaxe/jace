# Running experiments

Every training, pretraining, and evaluation loop runs through `uv run python -m <module>`. Pass `--help` to any module for its full flag list and defaults.

## 10-deck self-play lab

Train on the 10 built-in closed-decklist `elemental-world-v2` decks with no external data:

```sh
uv run python -m jace.loops.train_ten_decks \
  --run-id lab-baseline \
  --updates 150 \
  --num-envs 32 \
  --collect-workers 8 \
  --eval-every 25
```

Runs write checkpoints, metrics, forensics, and Academy replays under `<runs-dir>/v2-ten-deck-lab/<run-id>/` (default `--runs-dir ../jace-runs`).

## Full-data 17Lands self-play

Train across cached 17Lands decks (`decks.pt`) with a nested color-balanced card-pool curriculum, periodic held-out mirrored evaluation, and atomic resume checkpoints:

```sh
uv run python -m jace.loops.train_full_data \
  --run-id tla-self-play \
  --decks-path ../jace-runs/data/decks.pt \
  --updates 600 \
  --curriculum-stages 90:0,180:200,347:400 \
  --curriculum-blend-updates 10 \
  --collect-workers 8 \
  --eval-workers 8
```

Resume an interrupted or continuing run from its latest atomic `resume-state.pt` (restoring policy weights, Adam state, and `SnapshotPool`):

```sh
uv run python -m jace.loops.train_full_data \
  --run-id tla-self-play \
  --decks-path ../jace-runs/data/decks.pt \
  --updates 1000 \
  --resume
```

Pass `--init-checkpoint <path>` with a new `--run-id` to start a fresh schedule from an existing `.safetensors` checkpoint.

## Experiment screening and head-to-head judging

`jace.loops.harness` runs short warm-start screens or standalone adaptive mirrored head-to-head checks (`100 -> 200 -> 500` deals) and prints a JSON verdict (`PROMOTE`, `KILL`, or `INCONCLUSIVE`):

```sh
# Warm-start screen with counterfactual forks on declare_attacker
uv run python -m jace.loops.harness \
  --run-id fork-screen \
  --init-checkpoint path/to/baseline.safetensors \
  --forks-per-update 32 \
  --fork-kinds declare_attacker

# Standalone adaptive head-to-head evaluation between two checkpoints
uv run python -m jace.loops.harness \
  --eval-only \
  --candidate path/to/candidate.safetensors \
  --baseline path/to/baseline.safetensors
```

## Playing and replaying checkpoints in Academy

Every gameplay checkpoint includes a companion `.policy.json` file and native `managym` JSONL game recordings. Exact commands are recorded in each `forensics/iter-XXXX.json`:

```sh
# Play interactively against a saved checkpoint
uv run managym academy play \
  --policy path/to/checkpoints/iter-0100.policy.json \
  --opponent iter-0100

# Replay recorded evaluation games turn by turn
uv run managym academy replay path/to/games/eval-iter-0100.jsonl
```

## Pretraining the drafter and deck models on 17Lands

Download a set with `uv run spells add TLA`, then pretrain the drafter and shared program `CardEncoder`, followed by `DeckScore` and `DeckBuilder`:

```sh
# 1. Pretrain DrafterModel + CardEncoder (caches drafts.pt in --data-dir)
uv run python -m jace.loops.pretrain \
  --data-dir ../jace-runs/data \
  --run-dir ../jace-runs/pretrain \
  --set TLA

# 2. Pretrain DeckScore + DeckBuilder on frozen card vectors (caches decks.pt)
uv run python -m jace.loops.pretrain_decks \
  --data-dir ../jace-runs/data \
  --run-dir ../jace-runs/pretrain \
  --embedding-run ../jace-runs/pretrain/<drafter-run-id> \
  --set TLA
```

## Composing custom RL loops

`jace.rl` exposes composable callables rather than a monolithic trainer class:

| Extension point | Signature / protocol                                         | Built-in implementations                           |
| --------------- | ------------------------------------------------------------ | -------------------------------------------------- |
| Credit          | `CreditFn: (Trajectory) -> CreditTargets`                    | `gae_wdl`, `monte_carlo`                           |
| Row selection   | `SelectFn: (Tensor) -> Tensor`                               | `top_fraction`, `all_rows`                         |
| Policy/value loss | `LossFn: (PolicyBatch) -> LossBreakdown`                   | `damped_policy_loss`, `ppo_loss`                   |
| Schedule        | `LossScheduleFn: (int) -> tuple[float, float, LossFn]`       | `DecaySchedule`, `DampedSchedule`                  |
| Collection      | `CollectFn(agent, *, opponent_entry, update_idx) -> RolloutBatch` | Local collector or `MultiProcessCollector.collect` |
| Auxiliary loss  | `PrepareAuxFn: (ActorCritic, int) -> AuxLossFn | None`       | `ForkCollectorPool`, `build_batched_fork_aux_loss` |

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

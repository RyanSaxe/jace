# Running experiments

Entry points run via `uv run python -m <module>`; scripts live in `scripts/`.
Pass `--help` to any command for full flags and defaults.

## 10-deck self-play lab

Train on the 10 built-in `elemental-world-v2` decks (no external data):

```sh
uv run python -m jace.train.ten_decks \
  --run-id lab-baseline \
  --updates 150 \
  --num-envs 32 \
  --collect-workers 8 \
  --eval-every 25
```

Outputs go to `<runs-dir>/v2-ten-deck-lab/<run-id>/` (`--runs-dir` defaults to
`../jace-runs`).

## Full-data 17Lands self-play

Train on cached 17Lands decks (`decks.pt`) with a nested curriculum, held-out
mirrored evaluation, and atomic resume:

```sh
uv run python -m jace.train.full_data \
  --run-id tla-self-play \
  --decks-path ../jace-runs/data/decks.pt \
  --updates 600 \
  --curriculum-stages 90:0,180:200,347:400 \
  --curriculum-blend-updates 10 \
  --collect-workers 8 \
  --eval-workers 8
```

Resume from `resume-state.pt` (weights, Adam, EMA, `SnapshotPool`):

```sh
uv run python -m jace.train.full_data \
  --run-id tla-self-play \
  --decks-path ../jace-runs/data/decks.pt \
  --updates 1000 \
  --resume
```

Pass `--init-checkpoint <path>` with a new `--run-id` to start a fresh schedule
from a `.safetensors` checkpoint.

## Google Colab GPUs

`scripts/colab_run.py` provisions a Colab VM via
`uvx --from google-colab-cli colab`, uploads `decks.pt` and resume files
(chunked and SHA-256 verified over 48 MiB), bootstraps Rust and `uv` over SSH,
and launches `jace.train.full_data` under `nohup`. A tunnel-free `colab exec`
loop polls every 5 minutes (avoiding the 3-hour idle timeout), syncs
`metrics.jsonl`, `config.json`, and `checkpoints/resume-state.pt` every
`--sync-interval-s`, and stops the VM unless `--keep-vm` is set. Arguments after
`--` forward to `jace.train.full_data`.

```sh
# Dry-run: inspect commands and bootstrap script (drop --dry-run to launch)
uv run scripts/colab_run.py \
  --session tla-colab \
  --gpu L4 \
  --decks-path ../jace-runs/data/decks.pt \
  --backup-dir ../jace-runs/v2-colab/tla-self-play \
  --run-id tla-self-play \
  --dry-run \
  -- \
  --updates 600 \
  --curriculum-stages 90:0,180:200,347:400

# Resume an earlier session on a fresh VM from --backup-dir
uv run scripts/colab_run.py \
  --session tla-colab-resume \
  --gpu L4 \
  --decks-path ../jace-runs/data/decks.pt \
  --backup-dir ../jace-runs/v2-colab/tla-self-play \
  --resume-from ../jace-runs/v2-colab/tla-self-play \
  --run-id tla-self-play \
  -- \
  --updates 1000
```

## Screening and head-to-head judging

`jace.train.harness` runs warm-start screens or standalone adaptive matches
(`100 -> 200 -> 500` deals) and prints a JSON verdict (`PROMOTE`, `KILL`, or
`INCONCLUSIVE`):

```sh
# Warm-start screen with counterfactual forks on declare_attacker
uv run python -m jace.train.harness \
  --run-id fork-screen \
  --init-checkpoint path/to/baseline.safetensors \
  --forks-per-update 32 \
  --fork-kinds declare_attacker

# Standalone adaptive head-to-head
uv run python -m jace.train.harness \
  --eval-only \
  --candidate path/to/candidate.safetensors \
  --baseline path/to/baseline.safetensors
```

## Academy play and replays

Checkpoints ship with `.policy.json` specs and JSONL recordings; exact commands
appear in `forensics/iter-XXXX.json`:

```sh
uv run managym academy play \
  --policy path/to/checkpoints/iter-0100.policy.json \
  --opponent iter-0100

uv run managym academy replay path/to/games/eval-iter-0100.jsonl
```

## Pretraining drafter and deck models

Download a set (`uv run spells add TLA`), then pretrain the drafter + shared
`CardEncoder`, followed by `DeckScore` + `DeckBuilder`:

```sh
# 1. DrafterModel + CardEncoder (caches drafts.pt in --data-dir)
uv run python -m jace.pretrain.draft \
  --data-dir ../jace-runs/data \
  --run-dir ../jace-runs/pretrain \
  --set TLA

# 2. DeckScore + DeckBuilder on frozen card vectors (caches decks.pt)
uv run python -m jace.pretrain.decks \
  --data-dir ../jace-runs/data \
  --run-dir ../jace-runs/pretrain \
  --embedding-run ../jace-runs/pretrain/<drafter-run-id> \
  --set TLA
```

## Custom RL loops

`jace.rl` composes plain callables:

| Extension point | Signature | Built-ins |
| --- | --- | --- |
| Credit | `CreditFn: (Trajectory) -> CreditTargets` | `gae_wdl`, `monte_carlo` |
| Row selection | `SelectFn: (Tensor) -> Tensor` | `top_fraction`, `all_rows` |
| Loss | `LossFn: (PolicyBatch) -> LossBreakdown` | `damped_policy_loss`, `ppo_loss` |
| Schedule | `LossScheduleFn: (int) -> tuple[float, float, LossFn]` | `DecaySchedule`, `DampedSchedule` |
| Collection | `CollectFn(agent, *, opponent_entry, update_idx) -> RolloutBatch` | `LocalShardedCollector`, `MultiProcessCollector.collect` |
| Aux loss | `PrepareAuxFn: (ActorCritic, int) -> AuxLossFn \| None` | `ForkCollectorPool`, `build_batched_fork_aux_loss` |

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

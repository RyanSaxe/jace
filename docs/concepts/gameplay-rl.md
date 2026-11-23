# Gameplay RL

Stage A trains a gameplay policy over closed-decklist Limited Magic games on `managym` 0.2.3. jace reuses `manabot`'s observation encoder, `ActorCritic` backbone, and native rollout collector while implementing all credit assignment, row selection, policy losses, schedules, historical opponent pools, multi-process collection, and counterfactual forks in `jace.rl` and `jace.loops`.

## Policy backbone

`build_agent` in [`jace/loops/train_ten_decks.py`](../../src/jace/loops/train_ten_decks.py) instantiates `manabot`'s `ActorCritic` with:

| Setting               | Value                                                             |
| --------------------- | ----------------------------------------------------------------- |
| Observations          | `policy_input_version=11`, 80 cards and 64 permanents per player  |
| Card representation   | Semantic card encoder over `managym` engine program tokens        |
| Trunk                 | Width 160 (`train_full_data`) or 64 (`train_ten_decks`), 2 layers, 4 heads |
| Action conditioning   | `policy_film=True` (FiLM conditioning of legal action embeddings) |
| Value head            | `categorical_wdl` (3-class Loss / Draw / Win logits) with `masked_mean` pooling |

The width-160, 2-layer configuration has 2.38M parameters. Every game is played with `decklists="closed"`, so the player never observes the opponent's hidden library or unrevealed hand.

## One update step

`train_self_play` and `update_step` in [`jace/rl/loop.py`](../../src/jace/rl/loop.py) execute one on-policy iteration in six stages:

1. **Collect complete games.** CPU workers play `steps_per_env` target steps per stream with `batch_closure="complete-games"`, so every trajectory terminates cleanly under a single frozen weight snapshot and long games are never truncated.
2. **Assign credit (`jace.rl.credit`).** `gae_wdl` computes undiscounted (`gamma=1.0`) policy advantages with `lam_policy` (default `0.5`) and 3-class WDL lambda-returns with `lam_value` (default `0.8`). Reverse scans group rows by distance from each episode's terminal flag (`_reverse_waves`), scanning thousands of packed rows in as many iterations as the longest single game while cutting cleanly across game boundaries. `monte_carlo` (`gamma=1.0, lam=1.0`) is also available for ablations.
3. **Filter high-leverage rows (`jace.rl.select`).** `top_fraction` ranks flat decisions by `|advantage|` and keeps the top share (`top50` by default in `train_full_data`, `top25` in `train_ten_decks`) above `min_advantage` (`0.01`). Policy loss averages only over selected rows, while the critic trains on every row so value estimates stay calibrated across routine turns.
4. **Keep raw win-probability advantages.** By default in `damped_policy_loss` runs, advantages stay on their natural win-probability scale rather than being standardized to unit variance (`normalize_advantages=False`), preserving the relative weight of decisive turns over near-zero steps.
5. **Bucket minibatches by board size.** `bucketed_minibatch_order` shuffles rows, sorts them by observation size (`needs.sum(axis=1)`), slices them into `minibatch_size` buckets, and shuffles the bucket order. Grouping similar board sizes avoids padding early-game turns to late-game stalemate lengths on MPS.
6. **Cache semantic catalog encoding once per epoch.** `_epoch_catalog_cache` encodes the card program catalog once per epoch, accumulates gradients across minibatches into a leaf tensor, and runs one backward pass through the semantic card encoder at epoch end.

## Damped policy objective and schedules

`damped_policy_loss` in [`jace/rl/losses.py`](../../src/jace/rl/losses.py) implements the Ataraxos damped policy-gradient objective (Sokota et al., 2025) over plain `PolicyBatch` tensors:

```text
L = L_clip(ratio, A)
  + collection_kl * KL(pi || pi_behavior)
  + magnet_kl     * KL(pi || pi_reference)
  + value_coef    * CE(v_wdl, target_wdl)
```

- **Float64 ratio math.** Importance ratios and advantages are cast to `float64` (on CPU when training on MPS, which lacks `float64`, while preserving autograd) so clipped surrogates do not suffer precision drift.
- **Dual reverse-KL anchors.** `collection_kl` (default `0.1`) pulls the updated policy toward the frozen behavior distribution that collected the rollout. `magnet_kl` (`tau`) pulls toward `reference_distribution`, which defaults to `action_type_uniform` (uniform across legal action categories, then uniform within each category).
- **Schedules (`jace.rl.pool`).** `DecaySchedule` decays learning rate and magnet `tau` inverse-linearly across the run horizon (`initial / (1 + decay * progress)`). `DampedSchedule` maps updates onto a power-law horizon (`scale / pos**power`) matching Ataraxos Table S7. Standard `ppo_loss` with an entropy bonus is available for baselines.

## Historical opponents and gradient diagnostics

Pure self-play can chase strategic cycles. `SnapshotPool` in [`jace/rl/pool.py`](../../src/jace/rl/pool.py) keeps a bounded FIFO ring buffer of detached CPU weight dictionaries (default `max_size=16`, added every 10 updates). On `snapshot_fraction` of updates (default `0.25`), the learner plays one seat against a sampled past snapshot instead of two-seat current self-play.

Two matching modes are supported:

- **`uniform`:** Samples every stored snapshot with equal probability.
- **`even_odds` (default in `train_full_data`):** Tracks an exponential moving average win rate `p` against each snapshot (`ema_alpha=0.2`) and samples proportionally to `max(0.05, p * (1 - p))`, focusing compute on opponents near a 50% match rather than obsolete early checkpoints.

`GradientAgreementTracker` in [`jace/rl/diagnostics.py`](../../src/jace/rl/diagnostics.py) measures the cosine similarity between consecutive updates' mean gradient vectors on CPU float32 buffers, diagnosing whether wide deck diversity is washing out update direction.

## Multi-process collection and nested curricula

Single-process collection hits Python GIL limits well below a 16-core machine. `MultiProcessCollector` in [`jace/loops/collect_pool.py`](../../src/jace/loops/collect_pool.py) spawns persistent CPU worker processes (`torch.set_num_threads(1)` per worker), broadcasts detached CPU weights each update, collects complete-game `RolloutBatch` payloads in parallel, and merges them in deterministic rank order.

Key safeguards in the collector:

- **Engine error isolation.** Rare capacity overflows and engine errors are isolated per game through `GameErrorPolicy(threshold=0.05, window=200, warmup=100, recover_capacity=True)`. Worker exceptions and unexpected exits propagate cleanly to the parent process instead of hanging the pool.
- **Sharded 17Lands decks.** Each worker loads `decks.pt` once and builds a `UniformDecks` shard of `shard_size` (default `512`) decks rather than validating all 91,359 decks at startup, rotating shards every `shard_refresh_every` updates (default `50`).
- **Nested color-balanced curricula.** `nested_card_pools` in [`jace/eval/human_decks.py`](../../src/jace/eval/human_decks.py) grows card subsets greedily by adding whichever deck maximizes per-card log-balanced coverage across all 10 two-color pairs, keeping all five colors represented at small pool sizes (for example, 90 cards). Passing `--curriculum-stages 90:0,180:600,347:1200` transitions smoothly across stages over `--curriculum-blend-updates` updates (default `10`).
- **Single-writer replays.** Only rank 0 writes Academy game recordings to `games/train.jsonl` (scaling its recording fraction by `num_workers`) so workers never interleave JSONL lines.

## Batched counterfactual forks

Standard on-policy RL can leave low-entropy habits stuck in local optima, such as declaring every legal attacker 100% of the time. `jace.loops.fork_batch`, `jace.loops.fork_audit`, and `jace.rl.losses.fork_pairwise_loss` target these decisions directly using exact engine cloning (`env.clone_env()`):

1. **Harvest roots.** Spawned CPU workers play closed-decklist games and clone states matching `decision_kinds` (for example, `declare_attacker`) where the policy has at least two legal actions.
2. **Step paired rollouts in wide batches.** For each root, the engine forces the policy's top action `a0` against its runner-up `a1` across `R` paired playouts (`playouts_per_action`, default `4`) sharing identical continuation RNG streams (common random numbers). Every active branch across all harvested roots advances together in one batched forward pass per ply, either to game termination or to an optional `rollout_horizon` bootstrapped by the WDL value head.
3. **Update with bounded pairwise cross-entropy.** `fork_targets_from_returns` converts the paired rollout returns `(q0, q1)` into regularized targets `softmax(pair_returns / temperature)` and weights decisive forks by `|q0 - q1|` whenever the gap reaches `min_gap` (default `0.25`), zeroing out tied or noisy forks. `fork_pairwise_loss` computes cross-entropy restricted to `(a0, a1)` (whose gradient `p - target` stays full-strength even when a collapsed policy assigns `p = 0.999` to the wrong action) plus an `anchor_kl` reverse-KL penalty over all legal actions to protect shared trunk features.
4. **Reuse continuation value samples.** Full-game fork continuations reservoir-sample on-policy intermediate states (`value_samples_per_fork`) and train the WDL critic on their terminal outcomes (`build_batched_fork_aux_loss`). Before enabling truncated horizons, `audit_horizon_correlation` in [`jace/loops/fork_audit.py`](../../src/jace/loops/fork_audit.py) checks Pearson correlation and sign agreement between value-head gaps at candidate horizons and true full-game fork outcomes.

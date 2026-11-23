# Status

jace is developed in two stages on Avatar: The Last Airbender (`TLA`, 347 cards):

- **Stage A (active):** Self-play gameplay RL (`jace.rl`, `jace.eval`, `jace.runs`, `jace.loops.train_full_data`).
- **Stage B (pretrained, queued to reconnect):** Content card encoder, drafter, deck score, and deck builder (`jace.cards`, `jace.draft`, `jace.decks`, `jace.loops.pretrain*`).

## Gameplay player (Stage A)

Measured on fresh mirrored deals drawn strictly from the 10,137 held-out 17Lands TLA human decks (closed decklists, zero training overlap). All checkpoints are width-160, 2-layer `ActorCritic` models (2.38M parameters, `v11` observations) trained on an Apple M3 Max with 8 CPU collection workers and MPS updates.

| Checkpoint | Updates | Cumulative games / decisions | Train hours | vs. Heuristic (400g det.) | vs. Heuristic (400g stoch.) | Head-to-head |
| --- | --- | --- | --- | --- | --- | --- |
| 10-deck `u1200` (`w160-l2-overnight`) | 1,200 | 10 elemental decks | n/a | 55.0% (200g zero-shot) | 55.0% (200g) | baseline |
| `full-17lands-curr-1800u` (`u1800`) | 1,800 | 122,375 / 17.0M | 5.9 h | 61.8% [57.8%–65.7%] | 63.0% [59.1%–66.9%] | 62.3% vs. `u1200` (400g) |
| `full-17lands-cont-2000u` (`u3800`) | 3,800 | 258,696 / 34.2M | 12.0 h | **71.5%** [67.5%–75.5%] | **71.5%** [67.5%–75.5%] | 57.0% vs. `u1800` (400g), 71.5% vs. `u1200` (200g) |

Behavioral forensics on `u3800`:

- **Land drops:** 0.0% missed main-phase land drops.
- **Blocking:** 50.0% block rate (vs. Heuristic's 51.2%), unlocked by decaying magnet `tau` from `0.005` to `0.0017` across `u1800` to `u3800`.
- **Attacking:** 100.0% attack declaration rate remains pinned, which is the target of batched counterfactual forks (`jace.loops.fork_batch`).

## Pretrained Limited models (Stage B)

Measured on held-out 17Lands TLA data: 10,282 drafts (421,562 picks) for the drafter, 10,137 decks for the deck score, and 2,000 rule-legal pools for the deck builder. None of the learned models read 17Lands usage statistics.

| Model | Headline held-out metric | Baseline | Notes |
| --- | --- | --- | --- |
| Drafter (`jace.loops.pretrain`) | **68.4%** top-1, **87.5%** top-2, **94.6%** top-3 (0.8175 nats) | 45.8% top-1 (earliest average pick), 23.6% (random) | 4,000 steps, dropout 0.2, trains `CardEncoder` at 3x drafter LR |
| Deck score (`jace.loops.pretrain_decks`) | **0.6788** log loss per game (40% of oracle spread) | 0.6860 constant mean win rate (~0.6677 oracle incl. pilot skill) | 2,000 steps on frozen drafter card vectors |
| Deck builder (`jace.loops.pretrain_decks`) | **85.2%** non-basic card overlap, **78.1%** exact land-total match (68.31 nats) | 77.5% land-total match (always 17 lands) | Sinusoidal land-share head + Plackett-Luce spell rounds |

## What is built and tested

`uv run pytest -q` verifies every package without requiring external downloads:

| Package | Verified behavior |
| --- | --- |
| `jace.rl` | WDL GAE and Monte Carlo credit, top-fraction selection, length-bucketed minibatches, damped dual-KL and PPO losses, pairwise fork loss, schedules, even-odds `SnapshotPool`, and gradient-agreement tracking |
| `jace.eval` | Deterministic mirrored deals, parallel workers with engine panic/overflow isolation, adaptive sequential judge, forensics, probe diffs, 17Lands splitting, and color-pair-balanced nested curricula |
| `jace.loops` | Multi-process CPU collector with worker error surfacing, atomic resume, batched counterfactual forks with value reuse and horizon audit, fast promotion harness, and Stage B pretraining loops |
| `jace.runs` / `jace.tracking` | Academy-native `.safetensors` + `.policy.json` storage, atomic resume round-trips, OS-locked state transitions, and SHA-256 checkpoint verification |
| `jace.cards` / `jace.draft` / `jace.decks` / `jace.data` | Content `CardEncoder` and `CardTable`, multi-pod booster drafting, legal Plackett-Luce deck sampling with exact gradient log-probs, and 17Lands Parquet caching |

## Current gaps and open work

| Area | Current state |
| --- | --- |
| Attack selectivity | `u3800` still declares 100% of legal attacks. Batched counterfactual forks with continuation value reuse (`jace.loops.fork_batch`, `jace.loops.fork_audit`) are implemented and queued for training runs. |
| Reconnecting Stage B into self-play | Drafter, deck builder, and deck score pretrain cleanly on 17Lands data, but an end-to-end draft-build-play self-play loop on the v2 RL core is not wired yet. |
| Cross-set generalization | `CardEncoder` uses no card IDs or statistics, but it has only been trained and evaluated on `TLA`. Zero-shot evaluation on held-out cards or a second set is untested. |
| Booster rarities | `managym` does not expose rarity metadata, so wildcard and foil odds in `TLA_PLAY_BOOSTER` are placeholders. |
| Unplayable bonus cards | 3 of 347 `TLA` cards (181 of 101,496 human decks) are not implemented in `managym` 0.2.3. |

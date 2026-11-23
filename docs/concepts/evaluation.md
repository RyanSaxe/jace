# Evaluation

Magic has high deal variance: mana screw, mana flood, and play/draw order can hide real skill differences over small samples. `jace.eval` combines mirrored deals, parallel CPU execution, adaptive stopping, behavioral forensics, and fixed probe-state diffs so every training checkpoint is measured on both win rate and play quality.

## Mirrored paired deals

A `Deal` in [`jace/eval/deals.py`](../../src/jace/eval/deals.py) fixes two decks, their sideboards, a decklist disclosure mode (`"closed"` by default), and an engine seed. `play_match` plays every deal twice with seats swapped:

- **Leg 0:** Player A plays on the play with deck A, and Player B plays on the draw with deck B.
- **Leg 1:** Player B plays on the play with deck A, and Player A plays on the draw with deck B.

Each deal produces a `PairedDealResult` (`2-0` sweep won, `1-1` split, or `0-2` sweep lost). Because `1-1` mana-screw or matchup splits contribute zero paired variance, `summarize_paired_deals` in [`jace/eval/parallel.py`](../../src/jace/eval/parallel.py) computes the standard error and 95% confidence interval across per-deal win rates rather than treating individual games as independent.

`play_deals_parallel` shards deals round-robin across spawned CPU worker processes (`torch.set_num_threads(1)` per worker). Each worker loads checkpoints once under a cached `PolicyPlayer` inference scope and isolates rare engine panics or observation capacity overflows per leg without aborting the match.

### Held-out 17Lands splits

`load_human_deck_catalog` in [`jace/eval/human_decks.py`](../../src/jace/eval/human_decks.py) reads `decks.pt` using the exact deterministic CRC-32 `split_key` partition from `jace.data.store` (`held_out_share=0.1`). Full-data training samples exclusively from the 91,359 training decks, while `build_human_deals(..., split="held_out")` draws evaluation matchups strictly from the 10,137 unseen held-out decks.

## Adaptive sequential judging and experiment screening

Fixed-size evaluations either waste time on obvious blowouts or lack power on 3 to 5 percentage point gains. `evaluate_adaptive` and `play_adaptive_parallel` in [`jace/eval/parallel.py`](../../src/jace/eval/parallel.py) step through increasing mirrored deal stages (default `100 -> 200 -> 500` deals, or `200 -> 400 -> 1,000` games) on a shared deterministic deal stream:

| Stage            | Stopping rule                                                                    |
| ---------------- | -------------------------------------------------------------------------------- |
| Stage 1 (100 deals) | Stop as `WIN` if 95% CI > 50% and win rate >= 58%, or `LOSS` if CI < 50% and win rate <= 44% |
| Stage 2 (200 deals) | Stop as `WIN` or `LOSS` if 95% CI excludes 50%, or `TIE` if outside the close zone (48% to 56%) |
| Stage 3 (500 deals) | Final `WIN`, `LOSS`, or `TIE` from the 95% paired CI                             |

`jace.loops.harness` wraps this judge into a promotion gate (`PROMOTE`, `KILL`, or `INCONCLUSIVE`). It runs either a short warm-start or from-scratch screen followed by adaptive head-to-head scoring against a baseline checkpoint, or standalone `--eval-only` scoring between any two checkpoints. Any candidate whose missed land-drop rate regresses by more than 3 percentage points is killed automatically, even if headline win rate looks flat.

## Behavioral forensics

Win rate alone does not tell you *why* a checkpoint won or lost. `run_forensics` in [`jace/eval/forensics.py`](../../src/jace/eval/forensics.py) audits live mirrored games against `managym.Heuristic` (default 20 deals = 40 games) and records `ForensicsReport` metrics on every non-trivial decision:

| Metric                       | What it catches                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------- |
| `missed_land_drop_rate`      | Passing priority on an empty stack when playing a land is legal                 |
| `attack_rate` / `attack_declare_spread` | Attack selectivity vs. Heuristic and policy entropy on `declare_attacker` choices |
| `block_rate` / `block_agreement` | Blocker selectivity and agreement with Heuristic on `declare_blocker` choices |
| `wrong_side_target_rate`     | Targeting the opposite side of the board from Heuristic when both friendly and opponent targets are legal |
| `entropy_by_kind` / `heuristic_agreement_by_kind` | Per-action-space-type calibration and policy drift across priority, combat, and targeting |

When `games_path` is passed, both `play_match` and `run_forensics` capture every game through `managym`'s native `ScalarCapture` so you can step through individual turns in `uv run managym academy replay`.

## Fixed probe-state checkpoint diffs

Two checkpoints with similar win rates can play very differently. `capture_probes` in [`jace/eval/diffs.py`](../../src/jace/eval/diffs.py) records a deterministic bank of real decision states (priority, attack, block, and targeting) from heuristic games as `(deal, prefix_actions)` pairs. `diff_checkpoints` replays each probe state once and compares two `ActorCritic` checkpoints on the exact same observations, reporting:

- `mean_tv`: mean total-variation distance between action probability distributions
- `mean_kl`: mean KL divergence from `before` to `after`
- `argmax_flip_rate`: share of probes where the top action changed
- `mean_abs_value_delta`: mean absolute change in the value head's prediction
- `top_shifts`: the top-K largest policy shifts with human-readable action labels and probabilities

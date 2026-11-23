# Status

Jace v2 separates gameplay RL (Stage A) from drafting and deckbuilding (Stage B).
The gameplay player uses `manabot`'s `ActorCritic` with Jace's own RL core
(`jace.rl`), evaluation suite (`jace.eval`), multi-process collector, and
batched counterfactual fork engine. On 400 unseen 17Lands Avatar: The Last
Airbender Limited decks (347 unique cards, closed decklists), the `u3800`
player beats `managym`'s scripted Heuristic **71.5%** (95% CI 67.5% to 75.5%).
The drafter, deck score, and deck builder are pretrained on human data and
queued to reconnect once gameplay experiments settle.

Every number on this page is from Avatar: The Last Airbender. The text
before each table describes what it was measured on.

## What works

`uv run pytest -q` runs the test suite.

| Area        | Shown by the tests                                                  |
| ----------- | ------------------------------------------------------------------- |
| Gameplay RL (`jace.rl`) | GAE over WDL critics, top-fraction advantage selection, damped policy and pairwise fork losses, power-law and decay schedules, even-odds snapshot pool, and gradient-agreement tracking |
| Evaluation (`jace.eval`) | Deterministic mirrored deals, parallel workers with panic isolation, policy forensics, checkpoint diffs, and 17Lands human-deck splitting and nested curricula |
| Full-data training (`jace.loops`) | Multi-process CPU collection with worker error surfacing, atomic checkpoint resume, batched counterfactual fork rollouts with value reuse, and fast calibrated experiment harness |
| Run records (`jace.runs`) | Academy-compatible run layouts, atomic resume state round-trips, and JSONL metric logging |
| Drafting    | Pods draft by the rules, many at once, with one model call per pick |
| Deck build  | Every sampled deck is legal, a sampled deck's log-probability matches the one training computes for it, and the search returns distinct decks best first |
| Human data  | Drafts and decks load from 17Lands tables, and a drafter, deck score, and deck builder learn from them |

## Pretraining on human picks

The runs below ran on an Apple M3 Max GPU and were measured on 421,562
picks from 10,282 held-out drafts, which [Data](reference/data.md)
describes. Every run used the card encoder, 256 drafts a step and a
drafter 256 wide with 6 layers. The learning rates warm up over the
first twentieth of the steps, then fall in a straight line to a tenth of
their peak. The first row is `jace.loops.pretrain` with its default
settings.

Top 1 to Top 3 are the share of picks in which the human's card is among
the picker's best one, two or three. The log loss is the held-out negative
log-probability of the human's card, in nats a pick. The gap is how much
lower the same loss is on as many training drafts.

| Run                                                   | Steps | Top 1  | Top 2  | Top 3  | Log loss | Gap   |
| ----------------------------------------------------- | ----- | ------ | ------ | ------ | -------- | ----- |
| As set: dropout 0.2, encoder rate 1.5e-3              | 4,000 | 0.6844 | 0.8748 | 0.9457 | 0.8175   | 0.020 |
| Before tuning: no dropout, both rates 5e-4            | 2,500 | 0.6815 | 0.8719 | 0.9443 | 0.8260   | 0.021 |
| No dropout, a 5,000-step schedule stopped at 2,500    | 2,500 | 0.6784 | 0.8697 | 0.9429 | 0.8355   | 0.030 |
| Dropout 0.1                                           | 2,500 | 0.6823 | 0.8722 | 0.9443 | 0.8252   | 0.014 |
| Dropout 0.2                                           | 2,500 | 0.6817 | 0.8719 | 0.9442 | 0.8270   | 0.009 |
| Dropout 0.1, encoder rate 1.7e-4                      | 2,500 | 0.6796 | 0.8704 | 0.9432 | 0.8323   | 0.013 |
| Dropout 0.1, encoder rate 1.5e-3                      | 2,500 | 0.6827 | 0.8731 | 0.9448 | 0.8227   | 0.014 |
| Dropout 0.1, half the training drafts                 | 2,500 | 0.6783 | 0.8698 | 0.9429 | 0.8371   | 0.048 |
| The pack's card with the earliest average pick        |       | 0.4584 | 0.6725 | 0.8003 |          |       |
| A uniformly random card                               |       | 0.2362 | 0.3992 | 0.5256 | 1.7789   |       |

Without dropout, the drafter overfits. Its held-out log loss stops falling
near step 2,000, while its training loss keeps falling. Dropout in the
drafter's transformer slows the overfitting. Separately, the card encoder
learns better at three times the drafter's rate than at the same rate or a
third of it. With both changes, the drafter can train for more steps. The
run as set took 24 minutes. At its last step, its held-out log loss was
still falling slowly, and the gap was still growing. Training on half the
drafts cost 0.4 points of top-1 and 0.012 of log loss, which is more than
tuning gained. So more drafts would help the drafter more than further
tuning would.

The earliest-average-pick baseline ranks a pack's cards by their 17Lands
average pick position. That statistic is computed from every draft of the
set, the held-out drafts included, so the baseline's row is a little too
high. The drafter reads no statistics. In the table above, a card that ties
with the human's card counts against the pick.

This earlier comparison of the drafter's card inputs was measured on the
same held-out drafts, before the card encoder existed and before tuning:

| Picker                                  | Top 1 | Top 2 | Top 3 | Reads 17Lands statistics |
| --------------------------------------- | ----- | ----- | ----- | ------------------------ |
| Starting vectors from 17Lands statistics | 0.661 | 0.856 | 0.934 | Yes |
| From card text                          | 0.681 | 0.872 | 0.944 | No  |
| From card text and engine program       | 0.682 | 0.873 | 0.944 | No  |
| From all three                          | 0.683 | 0.873 | 0.945 | Yes |
| From engine program alone               | 0.619 | 0.817 | 0.907 | No  |

Those rows were measured before ties counted against the pick.

## Pretraining on human decks

The runs below ran on the same GPU, each training one model, and were
measured on held-out decks. The runs read the card vectors of the drafter
run from before tuning. The first row of each table is
`jace.loops.pretrain_decks` with its default settings, on the tuned
drafter's vectors.

The deck score is measured by its log loss a game on 10,137 held-out decks.
Predicting the training decks' overall win rate for every deck scores
0.6860. Beyond the noise of its few games, a deck's true win rate varies
with an SD of about 0.094, and that spread includes the pilot's skill. An
oracle that knew each deck's true rate would score about 0.6677, so no deck
score can gain more than about 0.0183 a game. A run's share is its gain
over the overall rate, as a fraction of that 0.0183. The gap is how much lower
the log loss is on as many training decks. Every run has 256 decks a step
and a model 128 wide with 3 layers, at rate 5e-4, unless the row names
another setting.

| Deck score run                       | Steps | Log loss | Share | Lowest log loss, at step | Gap    |
| ------------------------------------ | ----- | -------- | ----- | ------------------------ | ------ |
| As set, on the tuned drafter's vectors | 2,000 | 0.6788 | 40%   | 0.6788 at 2,000          | 0.001  |
| The same settings: seeds 0, 1 and 2  | 2,000 | 0.6785, 0.6787, 0.6793 | 41%, 40%, 37% | 0.6785 at 2,000 | 0.001 to 0.002 |
| Half as many steps                   | 1,000 | 0.6792   | 37%   | 0.6792 at 1,000          | 0.001  |
| Twice as many steps                  | 4,000 | 0.6823   | 21%   | 0.6788 at 2,250          | 0.012  |
| Rate 2e-4                            | 2,000 | 0.6787   | 40%   | 0.6787 at 2,000          | 0.000  |
| Rate 1e-3                            | 2,000 | 0.6791   | 38%   | 0.6788 at 1,250          | 0.002  |
| 64 wide, 1 layer                     | 2,000 | 0.6789   | 39%   | 0.6789 at 2,000          | 0.000  |
| 64 wide, 2 layers                    | 2,000 | 0.6788   | 40%   | 0.6788 at 2,000          | 0.000  |
| 256 wide, 3 layers                   | 2,000 | 0.6794   | 36%   | 0.6790 at 1,250          | 0.005  |
| Weight decay 0.1                     | 2,000 | 0.6785   | 41%   | 0.6785 at 2,000          | 0.001  |
| Dropout 0.1                          | 2,000 | 0.6786   | 41%   | 0.6786 at 2,000          | 0.001  |
| Dropout 0.3                          | 2,000 | 0.6788   | 39%   | 0.6788 at 2,000          | 0.001  |
| Dropout 0.3                          | 4,000 | 0.6790   | 39%   | 0.6785 at 2,250          | 0.002  |
| 1,024 decks a step                   | 500   | 0.6787   | 40%   | 0.6787 at 500            | 0.000  |
| 1,024 decks a step                   | 2,000 | 0.6845   | 9%    | 0.6793 at 750            | 0.015  |
| Half the training decks              | 2,000 | 0.6816   | 24%   | 0.6794 at 500            | 0.010  |

No setting does better than the spread across three seeds. The score
overfits after about 5.6 passes over the training decks, even with the
settings meant to slow overfitting. Every run that trained on all the
training decks reached its lowest log loss between 0.6785 and 0.6792.
Training on half the decks cost about 0.0009 at the best step, so more
decks would help a little more than any setting tried. Most of the gain
the score cannot reach is likely the pilot's skill. The oracle's 0.0183
includes that skill, but the deck score reads only the deck.

The builder is measured on the first 2,000 held-out pools whose human deck
the deck rules allow. Its log loss is the negative log-probability of the
human build, in nats a deck. Each build's cards are put in one random
order, and every measurement uses that same order. Overlap is the share of
the human deck's cards, basic lands aside, that the builder's likeliest
deck also contains. Lands is the share of pools whose likeliest deck has
the same land total as the human deck. Always choosing 17, the most common
total, scores 0.775. The gap is how much lower the log loss is on as many
training decks. Every run has 128 decks a step and a model 128 wide with 3
layers, at rate 5e-4, unless the row names another setting. Every run but
the last used an earlier land head, which read a land total only as its
share of the deck.

| Builder run                          | Steps | Log loss | Gap  | Overlap | Lands |
| ------------------------------------ | ----- | -------- | ---- | ------- | ----- |
| As set but for the earlier land head, on the tuned drafter's vectors | 9,000 | 68.71 | 0.29 | 0.833 | 0.026 |
| The steps before tuning              | 3,000 | 69.08    | 0.28 | 0.828   | 0.009 |
| Rate 1e-3                            | 3,000 | 68.98    | 0.34 | 0.828   | 0.014 |
| 256 wide                             | 3,000 | 68.91    | 0.37 | 0.830   | 0.010 |
| 256 decks a step                     | 3,000 | 69.00    | 0.30 | 0.831   | 0.012 |
| Three times as many steps            | 9,000 | 68.79    | 0.43 | 0.835   | 0.039 |
| Half the training decks             | 3,000 | 69.14    | 0.54 | 0.828   | 0.010 |
| Land head reading sines and cosines of the land share, as in the builder's code | 3,000 | 68.31 | 0.28 | 0.852 | 0.781 |

The builder is under-trained, not overfit. In every 3,000-step run, the
held-out loss was still falling at the last step. A higher rate, a wider
model and larger batches each lowered it a little more, as more steps do.
At 9,000 steps, the held-out loss is 0.3 to 0.4 nats a deck above the
training loss, out of about 69. Across all these runs, the overlap moves by
less than a point.

The earlier land head limited the builder. Because it read a land total
only as its share of the deck, the odds it learned rose with the total:
0.16, 0.22, 0.28 and 0.35 for 15 to 18 lands in the tuned run. The shares
of human decks with those totals are 0.01, 0.21, 0.77 and 0.01. The
builder's likeliest deck had 18 lands for 95% of pools, and had the human
deck's land total for 1 to 4 percent of pools, against 77.5 percent for
always choosing 17. When the land head also read sines and cosines of the
land share, the builder matched the human land total for 78 percent of
pools in 3,000 steps. Its log loss dropped 0.77 nats a deck and its overlap
rose 2.5 points, more than any setting did. The builder reads its land
total this way, so a builder checkpoint saved with the earlier land head
does not load. Training on half the decks cost only 0.06 nats a deck at
3,000 steps, so more decks would help the builder less than the land head
did.


## Training on real games (Jace v2)

The runs below trained a width-160, 2-layer `ActorCritic` (2.38M parameters,
`v11` observations, closed decklists) on an Apple M3 Max using 8 CPU
collection workers and MPS learning (`jace.loops.train_full_data`). All
win rates below were measured on fresh mirrored deals drawn strictly from
the 10,137 held-out 17Lands TLA human decks (347 unique cards, zero training
overlap).

| Checkpoint | Cumulative updates | Cumulative games / decisions | Train-loop hours | vs. Heuristic (400g det.) | vs. Heuristic (400g stoch.) | Head-to-head |
| ---------- | ------------------ | ---------------------------- | ---------------- | ------------------------- | --------------------------- | ------------ |
| 10-deck `u1200` (`w160-l2-overnight`) | 1,200 | 10 elemental decks | — | 55.0% (200g zero-shot) | 55.0% (200g) | baseline |
| `full-17lands-curr-1800u` (`u1800`) | 1,800 | 122,375 / 17.0M | 5.92 h | 61.8% [57.8%–65.7%] | 63.0% [59.1%–66.9%] | 62.3% vs. 10-deck `u1200` (400g) |
| `full-17lands-cont-2000u` (`u3800`) | 3,800 | 258,696 / 34.2M | 12.00 h | **71.5%** [67.5%–75.5%] | **71.5%** [67.5%–75.5%] | 57.0% vs. `u1800` (400g), 71.5% vs. `u1200` (200g) |

Key behavior verified on `u3800`:

- Missed main-phase land drops reached **0.0%** and stayed near zero throughout the run.
- Raising magnet `tau` (`0.005` down to `0.0017`) during `u1800` to `u3800` unlocked selective blocking (**50.0%** block rate vs. Heuristic's 51.2%).
- Attack declarations remain pinned at **100.0%**, which is the target of the next batched counterfactual fork experiment (`jace.loops.fork_batch`).

## The guess model

These numbers come from one run of `train_limited` for four iterations.
Each iteration had 2 pods and 4 blocks of 25 games per deck, with one pass
over each block's games. Two checks then scored every guess in the table
on the same decisions. The first used the scoring harness's 25 deals of
held-out human decks against the three scripted players, 300 games in all,
with a player from 150 iterations of an earlier run choosing the actions.
The second used 48 self-play games of freshly drafted and built decks.

Hand loss is the log loss per decision and card. The base rate gives each
card its frequency in hand over the same decisions, so it is fitted to the
very decks being scored.

| Guess                                   | Harness hand loss | Self-play hand loss |
| --------------------------------------- | ----------------- | ------------------- |
| Day-0 meta from booster odds, no network | 0.0520           | 0.0519              |
| Meta of the human decks, no network     | 0.0412            | 0.0450              |
| The guess model after four iterations   | 0.0411            | 0.0437              |
| Base rate fitted on the scored decisions | 0.0436           | 0.0424              |

The guess model's training loss, in its one pass over each iteration's
games, fell from 19.85 to 18.71 over the four iterations. The loss of the
earlier guess model, which read the decks being played, fell from 18.68 to
16.08 over the same four.

## Measured speed

On an Apple M3 Max CPU:

| Setup                                                  | Decisions a second |
| ------------------------------------------------------ | ------------------ |
| Random play through the adapter, simple decks, 16 games in a batch | About 16,400 |
| Random play through the adapter, real decks            | About 2,000 to 4,000 |
| The untrained player and guess model choosing actions, 16 games in a batch | 366 |

The build step of `train_limited` for 16 pools took 4.2 seconds an
iteration with an earlier search of 8 proposals and one round of swaps.
With the current search, which samples 256 decks a pool, it took 2.5
seconds. Each figure is from one short run.

The table below times the play models' update on the M3 Max's GPU, before
and after a change that made the transformers read only the real tokens,
packed without padding, through Metal attention kernels. Every measurement
used the same block of 39,398 decisions and the same weights, and the
losses agreed to five digits. Each figure is from a single run on a Mac
that was doing other work, so each before and after pair was measured side
by side.

| Measure                                                        | Before (ea08bbe) | After |
| -------------------------------------------------------------- | ---------------- | ----- |
| Forward and backward of a 512-row slice: player, win chance, guess | 123, 115, 114 ms | 82, 75, 69 ms |
| One block's update, warm                                        | 32.9 s           | 21.5 s |
| One block's update in a new process                             | 38.9 s           | 28.6 s |
| Update seconds per 1,000 decisions, two iterations of `train_limited` at the overnight run's settings | 1.21 | 0.82 |

## Not built yet

| Gap                         | Detail                                                     |
| --------------------------- | ---------------------------------------------------------- |
| Cards from other sets       | The card encoder reads only a card's text, engine program and printed facts, which exist before anyone plays it. But it has been trained only on the cards of the training drafts, so how it handles a card with no play data is untested |
| Booster rarities            | The engine has no rarity. The Avatar booster's wildcard and foil odds are placeholders |
| Scoring the other models in self-play | The training loops score the player on paired deals against random play, scripted greedy and the heuristic player, and record the guess model's loss on those games. Apart from the deck-score check, which tests both deck scores on each iteration's decks, nothing scores the other models in self-play |
| A constructed loop          | `train_player` plays human decks. No loop builds its own constructed decks |
| The card vectors in self-play | Self-play does not train them, as a working choice. A drafter meant to be frozen fell from 65.5% to 50.7% agreement with human picks when another model's loss moved the earlier embedding, so only the drafter's pretraining trains the card encoder. Whether and how the card vectors should learn from self-play is undecided |
| GPU                         | Pretraining, the updates of both self-play loops, and the drafting and deck building of `train_limited` run on `--device`, which defaults to the Apple GPU when there is one. The worker processes play every game on the CPU |

## Known problems

| Problem                     | Detail                                                     |
| --------------------------- | ---------------------------------------------------------- |
| Library labels              | The engine reports each seat's hand but not its library. The adapter works each library out from the deck, the hand and the board the acting seat sees. A library is one card off while the other seat has a foretold card, which that board hides, or while one seat controls a permanent the other owns and both decks contain that card |
| Abandoned games             | A game the engine fails on, cuts short, or whose board outgrows the board arrays is abandoned. Its decisions are left out of training, and its matchup is not played again. With board arrays of 80 cards and 64 permanents a seat, 15 of 5,491 games of a scripted player on human decks were abandoned |
| Held-out statistics         | The 17Lands statistics are computed from the held-out drafts too. They inflate the held-out score of the earliest-average-pick baseline, the one picker that still reads them, and of every measurement above marked as reading them |
| Schedule numbers            | The self-play rates of the drafter and the deck models, and the defaults of the deck-score check, are guesses that no measurement supports. Pretraining's settings were tuned in the runs above |

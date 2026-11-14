# Write a loop

A training loop in jace is a function you write from the package's pieces.
This guide reads the limited loop, then shows where to change it.

Start with the small one. It trains a player on the toy game in about
twenty lines of loop:

```sh
uv run python -m jace.loops.toy
```

Its source is [`jace/loops/toy.py`](../../src/jace/loops/toy.py).

## The limited loop

`limited_iteration` in
[`jace/loops/limited.py`](../../src/jace/loops/limited.py) runs one round of
draft, build, play and learn. This is its body, shortened:

```python
drafter = model_drafter(learners.drafter, generator=generator)
records = _draft(setup, drafter=drafter, other_seats=other_seats, rng=rng)
# Each pool's best `setup.blocks` decks, one row per pool and block.
played = _build(setup, learners, records=records, check=check, generator=generator)

# Block b plays every pool's b-th deck, then updates the player, its win chance and the guess model on its games.
blocks = _play_blocks(setup, learners, decks=_block_decks(played.decks, blocks=setup.blocks), ...)

after, checked = check_scores(check, played=tested, human=learners.human_score, learning=learners.score, ...)
metrics |= train_learning_score(learners.score, learners.human_score, recent=after.recent, ...)
if check.reads_learning and dials.builder_rate > 0:
    metrics |= _update_builder(setup, learners, played=played, wins=wins, games=games, dials=dials)
if dials.drafter_rate > 0:
    metrics |= _update_drafter(...)
return metrics, after
```

| Step  | What happens                                                              |
| ----- | ------------------------------------------------------------------------- |
| Draft | Every pod drafts at once. The learner fills the seats `other_seats` does not |
| Build | The search samples many decks for each pool and keeps the best few by the deck score |
| Play  | Block `b` plays every pool's `b`-th deck. In each block, every deck plays the same number of games, against decks of other pods, each game to its end |
| Learn | The player, its win chance and the guess model take a step after every block. After the last, both deck scores are tested on the iteration's decks, the learning one trains, the builder learns once the search reads the learning score, and the drafter learns |

[Deck building](../concepts/deck-building.md) explains the search, the two
deck scores and the switch between them.

`jace.loops.train_limited` runs this loop on real games. It builds the
models, loads checkpoints trained on human data, plays the games in worker
processes, calls `limited_iteration` once per iteration, and every few
iterations scores the player on paired deals against three scripted
players:

```sh
uv run python -m jace.loops.train_limited --help
```

[Status](../status.md) has what it has trained so far.

## What you hand it

| Argument      | Holds                                                         |
| ------------- | ------------------------------------------------------------- |
| `setup`       | What stays fixed: the card pool, the limited format, deck rules, sizes |
| `learners`    | The models, the frozen human-trained ones they are pulled toward, and one optimizer for each |
| `check`       | The deck-score check the iteration before returned, or `FIRST_CHECK` |
| `dials`       | This iteration's values from the schedule                     |
| `other_seats` | Drafters for seats the learner does not draft, such as past versions |
| `play`        | A function that plays a list of matchups to the end and returns the decisions |

The player, its win chance and the guess model arrive as functions, and
the games through `play`, so the same loop runs on manabot in worker
processes and on the toy game in one. The iteration returns its metrics and
the check to hand the next one.

`GuessModel` guesses from a field, the decks its opponents may hold, and
the loop does not set it. The `play` of `train_limited` sets the decks of
the matchups it is given before it sends the weights to the workers, so the
field holds while the games are played and while the guess model learns
from them:

```python
models.guess.set_field([deck for matchup in matchups for deck in matchup])
```

Build each model's optimizer from `own_parameters(model, embedding)`. No
optimizer of this loop holds the shared embedding, so a model whose rate is
zero does not change. [Training](../concepts/training.md) has the reason.

## Changing it

| To change                        | Edit                                                    |
| -------------------------------- | ------------------------------------------------------- |
| How a dial moves over training   | Write another schedule function                         |
| Which decks are played for a pool | `_build`, which keeps the search's best `setup.blocks` |
| When the search trusts the learning deck score | `ScoreRule`, or `check_scores` in `jace.loops.checked_score` |
| The builder's reward             | `_update_builder`, where it is a build's win rate minus its pool's mean |
| The drafter's reward             | `_update_drafter`, where the reward is the win rate of the pool's best build |
| Who drafts the other seats       | Pass different drafters as `other_seats`                |
| The order of updates             | Reorder the calls                                       |

## Constructed

A constructed loop skips the draft and build steps. It starts at `play`
with matchups of decks you supply. `jace.loops.train_player` is the
smallest one: it plays human decks and updates the player, its win chance
and the guess model. Nothing in it imports `jace.draft`.

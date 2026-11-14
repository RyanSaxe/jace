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
draft, build, play and learn. This is its body:

```python
drafter = model_drafter(learners.drafter, generator=generator)
records = _draft(setup, drafter=drafter, other_seats=other_seats, rng=rng)
played_decks, best_decks = _build(setup, learners, records=records, dials=dials, rng=rng)
decks = [candidate.deck for candidate in played_decks]

# Each block plays its games, then updates the player, its win chance and the guess model on them.
blocks = _play_blocks(setup, learners, decks=decks, play=play, dials=dials, rng=rng, generator=generator)
wins, games_played = blocks.wins, blocks.games_played

metrics |= _update_deck_score(setup, learners, decks=decks, wins=wins, games_played=games_played)
metrics |= _update_builder(setup, learners, records=records, best_decks=best_decks, generator=generator)
if dials.drafter_rate > 0:
    metrics |= _update_drafter(...)
```

| Step  | What happens                                                              |
| ----- | ------------------------------------------------------------------------- |
| Draft | Every pod drafts at once. The learner fills the seats `other_seats` does not |
| Build | Each pool gets ranked candidate decks, and one is chosen to play          |
| Play  | In each block, every deck plays the same number of games, against decks of other pods, each game to its end |
| Learn | The player, its win chance and the guess model take a step after every block; the deck score, builder and drafter take one after the last |

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
| `setup`       | What stays fixed: the card pool, draft format, deck rules, sizes |
| `learners`    | The models and one optimizer for each                         |
| `dials`       | This iteration's values from the schedule                     |
| `other_seats` | Drafters for seats the learner does not draft, such as past versions |
| `play`        | A function that plays a list of matchups to the end and returns the decisions |

The player, its win chance and the guess model arrive as functions, and
the games through `play`, so the same loop runs on manabot in worker
processes and on the toy game in one.

Build each model's optimizer from `own_parameters(model, embedding)`. No
optimizer of this loop holds the shared embedding, so a model whose rate is
zero does not change. [Training](../concepts/training.md) has the reason.

## Changing it

| To change                        | Edit                                                    |
| -------------------------------- | ------------------------------------------------------- |
| How a dial moves over training   | Write another schedule function                         |
| Which deck is played for a pool  | The two lines in `_build` that use `explore_share`      |
| The drafter's reward             | `_update_drafter`, where the reward is the deck's win rate |
| Who drafts the other seats       | Pass different drafters as `other_seats`                |
| The order of updates             | Reorder the calls                                       |

## Constructed

A constructed loop skips the draft and build steps. It starts at `play`
with matchups of decks you supply. `jace.loops.train_player` is the
smallest one: it plays human decks and updates the player, its win chance
and the guess model. Nothing in it imports `jace.draft`.

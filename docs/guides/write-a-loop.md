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

collected = collect(
    start(matchup),
    None,
    learner=learners.policy,
    opponent=opponent,
    guesser=learners.guesser,
    cards=len(setup.pool),
    history=setup.history,
    decisions=setup.decisions,
    generator=generator,
)
wins, played = _deck_records(decks, collected.games)

metrics |= _update_play_models(setup, learners, rollout=collected.rollout, dials=dials)
metrics |= _update_deck_score(setup, learners, decks=decks, wins=wins, played=played)
metrics |= _update_builder(setup, learners, records=records, best_decks=best_decks, generator=generator)
if dials.drafter_rate > 0:
    metrics |= _update_drafter(...)
```

| Step  | What happens                                                              |
| ----- | ------------------------------------------------------------------------- |
| Draft | Every pod drafts at once. The learner fills the seats `other_seats` does not |
| Build | Each pool gets ranked candidate decks, and one is chosen to play          |
| Play  | Random pairs of those decks play until enough decisions are collected     |
| Learn | The player, guess model, deck score, builder and drafter each take a step |

The loop has only run on the toy game, in a test. It has not trained
anything on Magic.

## What you hand it

| Argument      | Holds                                                         |
| ------------- | ------------------------------------------------------------- |
| `setup`       | What stays fixed: the card pool, draft format, deck rules, sizes |
| `learners`    | The models and one optimizer for each                         |
| `dials`       | This iteration's values from the schedule                     |
| `opponent`    | The policy in the seat the learner does not play              |
| `other_seats` | Drafters for seats the learner does not draft, such as past versions |
| `start`       | A function that starts a batch of games on your engine        |

The player and the guess model arrive as functions, so the same loop runs
on manabot and on the toy game.

## Changing it

| To change                        | Edit                                                    |
| -------------------------------- | ------------------------------------------------------- |
| How a dial moves over training   | Write another schedule function                         |
| Which deck is played for a pool  | The two lines in `_build` that use `explore_share`      |
| The drafter's reward             | `_update_drafter`, where the reward is the deck's win rate |
| Who drafts the other seats       | Pass different drafters as `other_seats`                |
| The order of updates             | Reorder the calls                                       |

## Constructed

A constructed loop skips the draft and build steps. It starts at `collect`
with decks you supply, and updates the player, the guess model and the deck
score. Nothing in those steps imports `jace.draft`. No constructed loop is
written yet.

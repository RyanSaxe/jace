# Write a loop

This guide walks through `limited_iteration`, the example limited loop in
`src/jace/loops/limited.py`, one step at a time. It then explains how to
change the reward, the explore rule and the other seats, and what a
constructed loop would skip.

The limited loop has run only on the toy game, for one iteration, in a test.
It shows that the pieces fit together. It has not trained anything on Magic.

## A loop is a function

jace has no trainer to configure. A loop is a function that calls the pieces
in the order it wants. To change what a loop does, edit the function or write
your own. The pieces it calls stay the same.

`jace.loops` holds two examples. `jace.loops.toy` is the smallest: it calls
`collect` and `update_player` in a `for` loop on the toy game, and you can
run it with `uv run python -m jace.loops.toy`. `jace.loops.limited` uses
every module, and the rest of this guide follows it.

## What the limited loop takes

```python
def limited_iteration(
    setup: LimitedSetup,
    learners: Learners,
    *,
    dials: Dials,
    opponent: Policy,
    other_seats: Sequence[Drafter],
    start: StartBatch,
    rng: random.Random,
) -> Metrics
```

| Argument      | Holds                                                                                                             |
| ------------- | ----------------------------------------------------------------------------------------------------------------- |
| `setup`       | What stays fixed for a run: the card pool, the draft format, the booster, the deck rules and the sizes.           |
| `learners`    | The models being trained, and one optimizer for each of them and for the embedding.                               |
| `dials`       | This iteration's values from the schedule.                                                                        |
| `opponent`    | The policy that plays the seat the learner does not.                                                              |
| `other_seats` | Drafters for the last seats of every pod, such as past versions. The learner drafts the rest.                     |
| `start`       | A function that starts a batch of games on whatever engine you use.                                               |
| `rng`         | The source of every random choice in the iteration.                                                               |

It returns the metrics of every update as a mapping from a name to a number.

A caller asks a schedule for the dials and calls the function once per
iteration. The test in `tests/test_loops.py` does it for one iteration on
the toy game:

```python
metrics = limited_iteration(
    setup,
    learners,
    dials=phased_schedule(drafter_starts=0, pull_ends=10)(5),
    opponent=policy,
    other_seats=[],
    start=lambda matchups: ToyBatch(matchups, games=8, seed=0),
    rng=random.Random(0),
)
```

## The steps

### 1. Set the learning rates

```python
_set_rate(learners.optimizers["player"], dials.player_rate)
_set_rate(learners.optimizers["drafter"], dials.drafter_rate)
```

The schedule owns the two learning rates, so the loop writes them into the
optimizers first.

### 2. Draft

```python
drafter = model_drafter(learners.drafter, generator=generator)
records = _draft(setup, drafter=drafter, other_seats=other_seats, rng=rng)
```

`_draft` opens a booster for every seat of every round of `setup.pods` pods,
and `run_drafts` drafts all the pods in step. The learner's drafter fills
every seat that `other_seats` does not, and it picks for all of them in one
forward pass per pick.

Each `DraftRecord` holds the pool every seat ended with, and every view a
seat had when it owed a pick. The drafter's update needs those views later.

### 3. Build a deck from each pool

```python
played_decks, best_decks = _build(setup, learners, records=records, dials=dials, rng=rng)
```

For each pool, `_build` calls `build_decks` with the deck builder's model as
the proposer and the deck score as the scorer. It gets back the candidate
decks, best first, and chooses one to play:

```python
explore = len(candidates) > 1 and rng.random() < dials.explore_share
played.append(rng.choice(candidates[1:]) if explore else candidates[0])
best.append(candidates[0])
```

The loop keeps both lists. The played decks go to the games, and the
top-scored decks teach the deck builder.

### 4. Play

```python
def matchup() -> tuple[Deck, Deck]:
    first, second = rng.sample(decks, SEATS)
    return first, second


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
```

Each game is played with two decks drawn at random, without replacement,
from the decks of the iteration. `collect` plays until the finished games hold
`setup.decisions` learner decisions. The loop passes `None` for `pending`, so
the games still running at that point are discarded.

### 5. Learn

The excerpt is abridged.

```python
wins, played = _deck_records(decks, collected.games)

metrics |= _update_play_models(setup, learners, rollout=collected.rollout, dials=dials)
metrics |= _update_deck_score(setup, learners, decks=decks, wins=wins, played=played)
metrics |= _update_builder(setup, learners, records=records, best_decks=best_decks, generator=generator)
if dials.drafter_rate > 0:
    metrics |= _update_drafter(...)
```

`_deck_records` counts each deck's wins and games. A draw counts as half a
win for each deck.

| Update                | Learns from                                                 | Loss                                                    |
| --------------------- | ----------------------------------------------------------- | ------------------------------------------------------- |
| `_update_play_models` | The rollout                                                 | `update_player`, then `update_guess`                    |
| `_update_deck_score`  | Each played deck's wins and games                           | `score_loss`                                            |
| `_update_builder`     | The top-scored deck the search found for each pool          | `choice_loss` with weights of one, on `writing_steps`   |
| `_update_drafter`     | The picks of the learner's seats, and each deck's win rate  | `choice_loss` with the win rate as the reward           |

Every update passes the embedding's optimizer to `apply` beside the model's
own, so the embedding moves with each of them. A deck that played no game
teaches the deck score and the drafter nothing.

## Change the reward

The player's reward is the game's result, and `advantages` in `jace.train`
turns it into a signal per decision. To reward the player differently, write
your own update from `player_loss` and `apply` in place of `update_player`.

The drafter's reward is the win rate of the deck built from its pool. It is
set in `_update_drafter`:

```python
rewards.extend([float(win_rates[deck])] * len(record.views[seat]))
...
loss = choice_loss(logits, torch.tensor(chosen), weights=reward - reward.mean())
```

Every pick of a seat gets the same reward. To change it, compute another
number per pick and pass it as `weights`. The two terms after that line are
separate: the entropy bonus weighted by `dials.drafter_entropy`, and the pull
toward `learners.human_drafter` weighted by `dials.human_pull`. The pull is
skipped when `human_drafter` is `None`.

## Change the explore rule

The deck explore rule is the two lines of `_build` shown in step 3. With
chance `explore_share`, the loop plays a candidate other than the top-scored
one, chosen evenly. `build_decks` returns every candidate with its score, so
any rule that reads scores fits there, such as sampling in proportion to
them.

[Exploration](../concepts/training.md#exploration) lists the other three
places a loop explores, and the dial for each.

## Change the other seats

Two arguments set who the learner faces.

`other_seats` is a sequence of `Drafter` functions for the last seats of
every pod. A `Drafter` takes seat views and returns one card per view, so a
past version of the model, a model trained on human picks, or a plain
function that takes the first card all fit. With an empty sequence, the
learner drafts every seat. The drafter's update learns only from the seats
the learner drafted.

`opponent` is the `Policy` that plays the seat the learner does not. Pass the
learner's own policy for self-play, a past version of it, or
`random_policy(engine_to_card)` from `jace.play`.

## What a constructed loop would skip

No constructed loop is built yet. It would be another short function that
leaves out two of the steps above:

- It skips `_draft` and `_update_drafter`, and does not import `jace.draft`.
- Its pools come from a collection in place of a draft. `build_decks` takes
  a pool as a plain tuple of card ids, so the build step is unchanged.

Playing, the player's and the guess model's updates, and the deck score's
update do not depend on where a deck came from. A lint rule keeps it that
way: only `jace.loops` may import `jace.draft`.

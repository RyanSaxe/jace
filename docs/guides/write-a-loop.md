# Write a loop

A training loop in jace is a function you write from the package's pieces.
This guide explains the limited loop, then shows where to change it.

Start with the toy loop. It trains a player on High Card, a toy game that
runs without the engine, and the whole loop is one function of about 60
lines:

```sh
uv run python -m jace.loops.toy
```

Its source is [`jace/loops/toy.py`](../../src/jace/loops/toy.py).

## The limited loop

`limited_iteration` in
[`jace/loops/limited.py`](../../src/jace/loops/limited.py) runs one
iteration of draft, build, play and learn. This is its body, shortened:

```python
drafter = model_drafter(learners.drafter, generator=generator)
records = _draft(setup, drafter=drafter, other_seats=other_seats, rng=rng)
# Each pool's best `setup.blocks` decks, one row per pool and block.
played = _build(setup, learners, records=records, check=check, generator=generator)

# Each block plays one build of every pool, mixing ranks, then updates the player, its win chance and the guess model.
builds = block_builds(len(played.decks) // setup.blocks, blocks=setup.blocks)
blocks = _play_blocks(setup, learners, decks=_block_decks(played.decks, builds=builds), ...)

after, checked = check_scores(check, played=tested, human=learners.human_score, learning=learners.score, ...)
if dials.score_rate > 0:
    metrics |= train_learning_score(learners.score, learners.human_score, recent=after.recent, ...)
if check.reads_learning and dials.builder_rate > 0:
    metrics |= _update_builder(setup, learners, played=played, wins=wins, games=games, dials=dials)
if dials.drafter_rate > 0:
    # A pool's deck is its best build, the one the deck score ranked first.
    metrics |= _update_drafter(..., win_rates=best_wins / best_games.clamp_min(1.0), ...)
return metrics, after, tested
```

The loop has two deck scores. One was trained on human decks and stays
frozen. The other, the learning deck score, trains on the decks the loop
plays. The search ranks decks by the human-trained score until the learning
score's test loss has been lower by a margin over several iterations. From
then on it ranks them by the learning score.

| Step  | What happens                                                              |
| ----- | ------------------------------------------------------------------------- |
| Draft | Every pod drafts at once. The learner's drafter takes every seat that `other_seats` does not fill |
| Build | For each pool, the search samples `setup.deck_samples` decks and keeps the `setup.blocks` decks that the deck score ranks best |
| Play  | Each block plays one build of every pool, mixing ranks across pools. In a block, every deck plays `setup.games_per_block` games, each to its end. With two pods or more, every game is between decks of different pods |
| Learn | The player, its win chance and the guess model update after every block. After the last block, both deck scores are tested on the iteration's decks, and then the learning deck score trains. The builder learns only in an iteration whose search read the learning score. The drafter learns last |

[Deck building](../concepts/deck-building.md) explains the search, the two
deck scores and the switch between them.

`jace.loops.train_limited` runs this loop on real games. It builds the
models, and can start the drafter and the deck models from checkpoints
trained on human data. It plays the games in worker processes and calls
`limited_iteration` once per iteration. Every `--evaluate-every`
iterations, it scores the player against random play, scripted greedy and
the heuristic player. Each score is over paired deals. A deal is two
held-out human decks, played four times so that each player plays each
deck from each seat.

```sh
uv run python -m jace.loops.train_limited --help
```

[Status](../status.md) gives the results of its runs so far.

## What you hand it

| Argument      | What it is                                                    |
| ------------- | ------------------------------------------------------------- |
| `setup`       | What stays fixed from one iteration to the next, such as the card pool, the limited format, the deck rules and the sizes |
| `learners`    | The models being trained, one optimizer for each, and the frozen human-trained drafter and deck score that the drafter and the learning deck score are pulled toward |
| `check`       | The deck-score check the previous iteration returned, or `FIRST_CHECK` for the first iteration. Its `reads_learning` is true once the search reads the learning deck score |
| `dials`       | This iteration's values from the schedule: the learning rates and loss weights that change over training |
| `other_seats` | Drafters for the last seats of every pod, such as past versions. The learner's drafter takes the other seats |
| `play`        | A function that plays a list of matchups to the end and returns the decisions, on the device the models train on |
| `rng`         | The random source that opens the boosters, schedules each block's games and seeds the iteration's PyTorch generator |

The loop calls the player, its win chance and the guess model only as
functions, and it plays games only through `play`. That is why the same
loop runs on manabot in worker processes and on the toy game in one
process. `limited_iteration` returns its metrics, the check to pass to the
next iteration, and the decks that finished a game.

The guess model starts each guess from a meta. A meta is a set of
statistics over a format's decks: how many copies of each card they
contain, and which cards they contain together. `limited_iteration` does
not change the meta.
`train_limited` sets it to the human decks' meta before the first
iteration. After each iteration, it blends that iteration's decks into the
meta, so no game is guessed with a meta that contains its own decks. Those
are the decks `limited_iteration` returns:

```python
metrics, after, played = limited_iteration(setup, learners, check=check, ...)
meta = blend_meta(models.guess.meta, played.decks, decay=arguments.meta_decay)
models.guess.set_meta(meta)
```

Build each model's optimizer from `own_parameters(model, embedding)`, which
leaves out the shared card embedding. Every model reads cards through that
embedding, so an optimizer that stepped it would change every other model's
input, even that of a model whose rate is zero.
[Training](../concepts/training.md) explains why the embedding stays fixed
in self-play.

## Changing it

| To change                        | Edit                                                    |
| -------------------------------- | ------------------------------------------------------- |
| How a dial moves over training   | Write another schedule function                         |
| Which decks are played for a pool | `_build`, which keeps the search's best `setup.blocks` |
| When the search switches to the learning deck score | `ScoreRule`, or `check_scores` in `jace.loops.checked_score` |
| The builder's reward             | `_update_builder`, where it is a build's win rate minus the mean win rate of its pool's builds |
| The drafter's reward             | `limited_iteration`, which passes `_update_drafter` the win rate of each pool's best build |
| Who drafts the other seats       | Pass different drafters as `other_seats`                |
| The order of updates             | Reorder the calls                                       |

## Constructed

A constructed loop skips the draft and the deck building. It starts at
`play`, with matchups of decks you supply. `jace.loops.train_player` is
one. It plays human decks and updates the player, its win chance and the
guess model. Nothing in it imports `jace.draft`.

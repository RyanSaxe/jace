# Status

jace is at an early stage. This page lists what works today and how it was
checked, the speeds measured so far, and the known limits.

A first version of every module exists and is tested. Nothing has been
trained beyond single update steps.

## What works

Each row is checked by the tests named in it. `uv run pytest -q` runs all
45 in about 3 seconds.

| What works                                                                                                               | Checked by                |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------- |
| A draft pod deals, passes and finishes by the rules, and rejects illegal picks and boosters that do not fit the format.  | `tests/test_pod.py`       |
| Many pods draft at once, with one model call per pick for all the seats a drafter fills. Boosters follow their slot odds. | `tests/test_draft.py`     |
| The deck builder makes only legal decks, and finds a best deck that a swap search from one start cannot reach.           | `tests/test_decks.py`     |
| The manabot adapter plays real games: the same seed replays the same game, a finished slot deals a new game, a seat never sees the opponent's decklist, and the hidden cards it reports agree with the board. | `tests/test_engine.py` |
| The player and the guess model read real engine boards, give illegal actions no probability, and ignore padding.         | `tests/test_play.py`      |
| The deck score, the deck builder, the player and the guess model keep their answers for old cards when the card pool grows. The drafter drafts over a larger pool than it was built with. | `tests/test_decks.py`, `tests/test_play.py`, `tests/test_draft.py` |
| Advantages and earlier guesses follow a seat through its own game. A game still running when `collect` stops comes back whole from a later call. | `tests/test_train.py` |
| A small player learns to pick its highest card on the toy game.                                                          | `tests/test_train.py`     |
| Real games on manabot train the real player and guess model for one step.                                                | `tests/test_real_games.py` |
| One limited iteration on the toy game updates every model.                                                               | `tests/test_loops.py`     |
| A win rate's range narrows as games are added, and an honest guess reads as honest.                                      | `tests/test_eval.py`      |
| A run's record survives a killed writer, refuses a second writer, and rejects a tampered checkpoint.                     | `tests/test_tracking.py`  |

The toy loop shows learning on a game whose best play is known.
`uv run python -m jace.loops.toy` trains a small player for 40 iterations,
and the share of decisions where it picks its best card rises from 0.36 to
0.78.

## Measured speed

These were measured on the CPU of an Apple M3 Max, with 16 games in a batch.
A decision is one action by one seat.

| Setup                                                                             | Decisions a second |
| --------------------------------------------------------------------------------- | ------------------ |
| Random play through the manabot adapter, simple decks                             | About 16,400       |
| Random play through the manabot adapter, random two-colour decks                  | About 3,400        |
| The untrained, small player and guess models choosing every action                | 366                |

## Known limits

### Data and pretraining

- jace does not load 17Lands data, and no model is pretrained on human
  drafts or games.
- A card's starting vector is not built from its rules text or its engine
  program yet. The tests use random vectors.

### Drafting

- A real booster needs each card's rarity, and the engine has none. You must
  add rarity from another source before you build a `CardPool`.
- `TLA_PLAY_BOOSTER`, the Avatar: The Last Airbender booster, is simplified.
  The odds of its wildcard and foil slots are placeholders.

### Deck building

- The swap search is slow. At 200 proposals and full depth, one pool takes
  about two million deck scores and 24 seconds on one CPU.

### Games

- The manabot adapter steps one scalar game per slot in a Python loop,
  because manabot's batched mode does not expose hidden hands yet.
- The engine raises an error when a board holds more than 40 permanents. The
  adapter does not catch it, so the error stops the run.
- manabot is pinned to one git commit, and it has no licence file yet.

### Training

- The limited loop has run only on the toy game. On manabot, only game
  collection and the player's and the guess model's updates have run.
- No constructed loop is built yet.
- The numbers for the drafter and the explore share in
  `jace.loops.schedules` are first guesses with no source.
- Nothing has run on a GPU.

### Evaluation

- `win_rate` resamples single games and treats them as independent. It does
  not yet replay one deal four ways, so its range overstates certainty when
  many games share a deck or a deal.

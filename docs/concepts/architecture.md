# Architecture

jace is a set of modules that a training loop calls. This page describes
what each module contains and how games reach the code that learns from
them.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/modules-dark.svg">
  <img alt="The modules of jace in rows, where a module imports only from the rows below it. Top: jace.loops. Then jace.draft, jace.decks, jace.train and jace.eval. Then jace.play. Then jace.engine, the only module that imports manabot, the game engine. At the bottom, jace.cards. jace.tracking stands alone." src="../assets/modules-light.svg">
</picture>

| Module          | Contains                                                  |
| --------------- | --------------------------------------------------------- |
| `jace.cards`    | Card records, card ids, the card encoder and the table frozen from it |
| `jace.engine`   | `GameBatch`, the interface games are played through, and the code that plays a match between two players |
| `jace.play`     | The guess model, the player and the win chance            |
| `jace.draft`    | Boosters, the draft pod and the drafter                   |
| `jace.decks`    | The deck score and the deck builder                       |
| `jace.train`    | Game collection, the losses and the update step           |
| `jace.eval`     | Paired deals, win rates and the check on the guess model  |
| `jace.data`     | Human drafts and decks from 17Lands, and the card encoder's inputs |
| `jace.loops`    | Training loops, limited formats, schedules and the command-line entry points |
| `jace.tracking` | A training run's record on disk                           |

`jace.data` is not in the diagram. It imports only `jace.cards`, and only
`jace.loops` imports it.

## Games go through one interface

Every game played in training goes through `GameBatch`. A batch is made
for a fixed list of matchups, each a pair of decks, and plays exactly
those games, a few at a time, until all are over.

```python
class GameBatch(Protocol):
    def decisions(self) -> Decisions: ...
    def act(self, actions: npt.NDArray[np.int64]) -> tuple[Finished, ...]: ...
    def hidden_cards(self) -> HiddenCards: ...
```

| Method         | Returns                                                             |
| -------------- | ------------------------------------------------------------------- |
| `decisions`    | The pending decision of every game in play: which game, who acts, the board they see |
| `act`          | The games that ended on this step, with their decks and outcome     |
| `hidden_cards` | Both seats' true hands and libraries, for training the guess model and the win chance |

When a game ends, the batch starts the next matchup of the list in its
place. Once none is left, `decisions` returns no rows. `Decisions` and
`Finished` identify each game by its position in the list.

Two modules implement the interface:

- `jace.engine.manabot` plays real Magic through the batched games of
  managym, manabot's engine. Its `board_layout` returns the layout of the
  engine's board arrays as a `BoardLayout`, which the play models are
  built from.
- `jace.engine.toy` plays High Card, a game in which each seat holds three
  cards, picks one, and the higher pick wins. Tests and the smallest
  example use it, so they need no engine.

## Scoring plays one game at a time

Scoring is the only part of jace that plays games without `GameBatch`.
`jace.engine.scoring` plays a match between any two players, such as jace
checkpoints or manabot's scripted players. It plays one game at a time for
two reasons. manabot's scripted players read manabot's own game object,
which a batch does not give them. Two different players also each need a
seat with its own memory of the game.

Each game also runs in a child process. A step of the engine can take very
long on a rare board, and only killing its process ends that step. When a
game runs past its time limit, `play_games` kills the process, records the
game as hung and plays the rest in a new process.

`jace.eval` defines which games make up a match and how their results are
scored. A deal is two decks and an engine seed. `legs` lists every deal as
four games, so each player plays each deck in each seat. `score_match`
gives a win rate with a range found by resampling whole deals, because the
four games of a deal share their decks and shuffles.

`jace.engine.manabot` and `jace.engine.scoring` are the only files that
import managym or manabot, so a change in either package affects only
those two files.

## Pieces, not a framework

jace has no trainer class. `jace.train` contains functions: play these
games, compute this loss, take this step. A training loop is a function you
write by calling them. `jace.loops` contains two such loops: `toy`, which
trains a player on High Card, and `limited`, which runs one round of
draft, build, play and learn.

Seven modules in `jace.loops` run from the command line:

| Entry point                 | Does                                            |
| --------------------------- | ----------------------------------------------- |
| `jace.loops.toy`            | Trains a player on High Card and prints how often it picks its best card |
| `jace.loops.pretrain`       | Trains the drafter and the card encoder, on human picks |
| `jace.loops.pretrain_decks` | Trains the deck score and the deck builder, on human decks |
| `jace.loops.train_limited`  | Trains every model, on real games. The drafter and the deck models can start from pretraining checkpoints |
| `jace.loops.keep_running`   | Runs `train_limited` and restarts it from its newest checkpoints whenever it stalls |
| `jace.loops.train_player`   | Trains only the player, the win chance and the guess model, on fixed human decks |
| `jace.loops.score`          | Scores a run's checkpoints against each other and against manabot's scripted players |

Only `jace.loops` imports `jace.draft`, so constructed play, which has no
draft, needs no drafting code. A loop for it would call the same pieces
and leave out the draft.

## The things that have state

Most of jace is functions over values. Five things have real states:

| Thing                | States                                           | Moved by          |
| -------------------- | ------------------------------------------------ | ----------------- |
| A draft pod          | `Picking`, `Complete`                            | `advance`         |
| A play worker        | `Starting`, `Ready`, `Playing`                   | `play_on_workers` |
| A training run       | created, running, completed, failed, interrupted | `writing`         |
| A game               | A pending decision, or finished                  | `GameBatch.act`   |
| The deck-score check | The search reads the human-trained score, then the learning one | `check_scores` |

A draft pod and a play worker have one type for each state.
[Training](training.md#playing-games-in-worker-processes) describes the
workers. A run's states are the values of `RunState`, and each change is one
line of the run's `state.jsonl`. A game is pending while it has a row in
`decisions`, and finished once `act` returns it as a `Finished`. The
deck-score check is one `ScoreCheck` record, which the limited loop passes
from each iteration to the next. Its `reads_learning` becomes true once and
never becomes false again.

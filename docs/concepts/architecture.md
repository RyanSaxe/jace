# Architecture

jace is a set of modules that a training loop calls. This page shows what
each module owns and how games reach the code that learns from them.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../assets/modules-dark.svg">
  <img alt="The modules of jace in rows, where a module imports only from the rows below it. Top: jace.loops. Then jace.draft, jace.decks, jace.train and jace.eval. Then jace.play. Then jace.engine, whose adapter is the only code that imports manabot. At the bottom, jace.cards. jace.tracking stands alone." src="../assets/modules-light.svg">
</picture>

| Module          | Owns                                                      |
| --------------- | --------------------------------------------------------- |
| `jace.cards`    | Card records, card ids, the card encoder and the table frozen from it |
| `jace.engine`   | `GameBatch`, the interface games are played through, and the games that score two players |
| `jace.play`     | The guess model, the player and the win chance            |
| `jace.draft`    | Boosters, the draft pod and the drafter                   |
| `jace.decks`    | The deck score and the deck builder                       |
| `jace.train`    | Game collection, the losses and the update step           |
| `jace.eval`     | Paired deals, win rates and the check on the guess model  |
| `jace.data`     | Human drafts and decks from 17Lands, and the card encoder's inputs |
| `jace.loops`    | Training loops, schedules and the command-line entry points |
| `jace.tracking` | A training run's record on disk                           |

The diagram leaves out `jace.data`. It imports only `jace.cards`, and only
`jace.loops` imports it.

## Games go through one interface

Everything that plays a game in training uses `GameBatch`. A batch is made
for a fixed list of matchups and plays exactly those games, a few at a
time, until all are over.

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

When a game ends, the next matchup of the list takes its place. Once none
is left, `decisions` has no rows. A game is known by its position in the
list.

Two things implement the interface:

- `jace.engine.manabot` plays real Magic through the batched games of
  managym, manabot's engine. It also reports the layout of the engine's
  board arrays as a `BoardLayout`, which the play models are built from.
- `jace.engine.toy` plays High Card, a three-card game. Tests and the
  smallest example use it, so they need no engine.

## Scoring plays one game at a time

Scoring a player is the one thing that does not go through `GameBatch`.
`jace.engine.scoring` plays a match between any two players, which can be
jace checkpoints or manabot's scripted players. Each game runs alone, in a
child process, for two reasons. manabot's scripted players read manabot's
own game object. And a step of the engine can take very long on a rare
board, which only killing its process ends: `play_games` records such a
game as hung and plays the rest.

`jace.eval` decides which games a match holds and what they mean. `legs`
lists every deal as four games, so each player holds each deck in each seat,
and `score_match` gives a win rate whose range resamples whole deals.

`jace.engine.manabot` and `jace.engine.scoring` are the only files that
import managym or manabot, so a change in either reaches two files.

## Pieces, not a framework

jace has no trainer class. `jace.train` holds functions: play these games,
compute this loss, take this step. A training loop is a function you write
by calling them. `jace.loops` holds two: `toy`, a player on High Card, and
`limited`, one round of draft, build, play and learn.

Five modules there run from the command line:

| Entry point                 | Does                                            |
| --------------------------- | ----------------------------------------------- |
| `jace.loops.pretrain`       | Trains the drafter, on human picks              |
| `jace.loops.pretrain_decks` | Trains the deck score and the deck builder, on human decks |
| `jace.loops.train_limited`  | Trains every model, on real games, from those checkpoints |
| `jace.loops.train_player`   | Trains only the player and the guess model, on human decks |
| `jace.loops.score`          | Scores a run's checkpoints against each other and against manabot's scripted players |

Because only `jace.loops` imports `jace.draft`, constructed play is the same
code with the draft step left out.

## Where state lives

Most of jace is functions over values. Three things have real states, and
each is written as one type per state:

| Thing           | States                                           | Moved by               |
| --------------- | ------------------------------------------------ | ---------------------- |
| A draft pod     | `Picking`, `Complete`                            | `advance`              |
| A training run  | created, running, completed, failed, interrupted | `writing`              |
| A game          | A pending decision, or finished                  | `GameBatch.act`        |

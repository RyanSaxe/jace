# jace

Learn to draft, build decks and play Magic: The Gathering by self-play.

The project trains its own models on the game engine from
[manabot](https://github.com/loopflowstudio/manabot). A drafter picks cards, a
deck builder turns each pool into a 40-card deck, and a player plays the games.
Game results train all three.

The first target is limited. Constructed uses the same code without the
drafter, because the deck builder and the player do not depend on it.

## Status

A first version of every module exists, with tests. A full limited iteration
runs end to end on a toy game. Nothing has been trained on real Magic games
yet.

## Layout

| Module | Holds |
|---|---|
| `jace.cards` | Card records, the pool that defines a card id, and the one card embedding every model reads |
| `jace.engine` | `GameBatch`, the interface every caller plays games through. `jace.engine.manabot` implements it on manabot and `jace.engine.toy` on a tiny game for tests |
| `jace.draft` | Boosters, the draft pod, whole drafts across pods, and the drafter model |
| `jace.decks` | The deck score, and a deck builder that proposes whole decks and improves them |
| `jace.play` | The guess model, which predicts the opponent's deck and hand, and the player model |
| `jace.train` | The pieces a loop is written from: game collection, advantages, losses and the update step |
| `jace.eval` | Win rates with a range, and the guess model's honesty check |
| `jace.tracking` | Run records as plain files |
| `jace.loops` | Example loops written from the pieces, and the schedules that drive them |

## Writing a loop

The package is pieces, not a framework. A training loop is a short function
that calls them in the order it wants. `jace.loops.limited` is one example:
it drafts pods, builds a deck from each pool, plays the decks against each
other and updates every model. A schedule is a function from the iteration
number to that iteration's dials, such as learning rates and how much to
explore.

## Develop

Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest
uvx pre-commit install
```

The pre-commit hook and the `byor gate` workflow run the same checks: the
ast-grep rules in `.byor/rules`, ruff, ty, and three documentation checks.

## Conventions

- Start from functions and plain values. A type is added only when it has
  several fields or states a rule the type checker should enforce.
- A real state machine is explicit: one immutable type per state and one
  function per transition. A sequence of calls is a function.
- A dependency is handed in as an argument. One behaviour is a callable
  type, and a protocol is for an injected object with several methods.
- Two lint rules guard the boundaries: only `jace.engine.manabot` imports
  manabot, and only `jace.loops` imports `jace.draft`.
- A module's public interface is its `__all__`. Tests use only that interface.

## Notice

jace is unofficial Fan Content permitted under the Fan Content Policy. Not
approved/endorsed by Wizards. Portions of the materials used are property of
Wizards of the Coast. ©Wizards of the Coast LLC.

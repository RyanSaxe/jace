# mtg-rl

Learn to draft, build decks and play Magic: The Gathering by self-play.

The project trains its own models on the game engine from
[manabot](https://github.com/loopflowstudio/manabot). A drafter picks cards, a
deck builder turns each pool into a 40-card deck, and a player plays the games.
Game results train all three.

The first target is limited. Constructed uses the same code without the
drafter, because the deck builder and the player do not depend on it.

## Status

The repository holds the first module, `mtg_rl.draft`, which runs a draft pod
as an explicit state machine. Nothing is trained yet.

## Layout

| Module | Holds |
|---|---|
| `mtg_rl.draft` | The draft pod: seats, packs, passing and pools |

Modules for cards, the engine, decks, play, training and evaluation are added
as each is built.

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

- Every state machine is explicit. A state is an immutable value with its own
  type, and each transition is one function from a state to the next.
- A module's public interface is its `__all__`. Tests use only that interface.

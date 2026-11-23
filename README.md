# jace

jace learns to play, draft, and build decks in Limited Magic: The Gathering by self-play. It plays its games on `managym` 0.2.3, the Rust rules engine from [manabot](https://github.com/loopflowstudio/manabot).

On 400 unseen 17Lands Avatar: The Last Airbender Limited decks (347 cards, closed decklists), the gameplay player wins **71.5%** of mirrored games against `managym`'s scripted Heuristic (95% CI 67.5% to 75.5%) after 258,696 cumulative self-play games. The drafter and deck models are pretrained on 17Lands human data and queued to reconnect in Stage B. See [Status](docs/status.md) for the full measurements.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/approach-dark.svg">
  <img alt="Jace v2 two-stage architecture. Stage A (active gameplay self-play) loops through Collect, Learn in jace.rl, and Evaluate in jace.eval. Stage B (pretrained on 17Lands) connects jace.cards to jace.draft and jace.decks." src="assets/approach-light.svg">
</picture>

## Core ideas

- **Own the RL algorithms, reuse the engine.** jace uses `managym` for game rules and `manabot`'s `ActorCritic` backbone (`v11` observations, semantic card program encoder, FiLM action conditioning, 3-class Win/Draw/Loss critic). Every RL algorithm lives as plain PyTorch and NumPy code in `jace.rl`, and AST rules prevent importing upstream training recipes.
- **Stabilize self-play on real Limited decks.** Training combines undiscounted WDL generalized advantage estimation, top-fraction advantage filtering, a damped clipped policy loss with collection and magnet reverse-KL anchors, AlphaStar-style even-odds historical snapshots, nested color-balanced card-pool curricula, and batched counterfactual forks.
- **Represent cards from content alone.** For drafting and deck building, `CardEncoder` builds each card's vector from its printed text, engine program tokens, and printed facts, with no 17Lands usage statistics and no fixed per-card output head.

## Quick start

jace requires Python 3.13 and [uv](https://docs.astral.sh/uv/). Prebuilt `managym` wheels exist for Apple Silicon macOS and x86-64 Linux. Other platforms compile the engine from source and need a Rust toolchain.

```sh
git clone https://github.com/RyanSaxe/jace.git
cd jace
uv sync
uv run pytest -q                                       # unit tests
uv run python -m jace.loops.train_ten_decks --help     # 10-deck lab loop (no external data)
uv run python -m jace.loops.train_full_data --help     # full 17Lands self-play loop
```

`train_ten_decks` runs closed-decklist self-play on the 10 built-in elemental decks without downloading external data. `train_full_data` trains across 91,359 17Lands Avatar: The Last Airbender decks in `decks.pt` and evaluates on 10,137 unseen human decks. See [Install](docs/getting-started/install.md) for dataset setup.

## Documentation

| Page                                                   | Topic                                                         |
| ------------------------------------------------------ | ------------------------------------------------------------- |
| [Install](docs/getting-started/install.md)             | Requirements, tests, 10-deck quick run, and 17Lands data      |
| [Architecture](docs/concepts/architecture.md)          | Package layers, AST import boundaries, and CLI entry points   |
| [Gameplay RL](docs/concepts/gameplay-rl.md)            | Policy backbone, `jace.rl`, multi-process collection, forks   |
| [Evaluation](docs/concepts/evaluation.md)              | Mirrored deals, adaptive harness, forensics, and probe diffs  |
| [Limited models](docs/concepts/limited-models.md)      | Content card encoder, drafter, deck builder, and deck score   |
| [Running experiments](docs/guides/running-experiments.md) | Training runs, Academy replays, screening, and custom loops |
| [Data](docs/reference/data.md)                         | 17Lands and MTGJSON pipeline and Avatar set counts            |
| [Run records](docs/reference/run-records.md)           | Academy-native `jace.runs` and locked `jace.tracking` layouts |
| [Sources](docs/reference/sources.md)                   | Papers, repositories, and datasets behind the design          |
| [Status](docs/status.md)                               | Current results, verified components, and open work           |
| [Contributing](docs/contributing/README.md)            | Dev setup, automated checks, boundary rules, and tests        |

## Licence

A licence has not been chosen yet.

## Notice

jace is unofficial Fan Content permitted under the Fan Content Policy. Not
approved/endorsed by Wizards. Portions of the materials used are property of
Wizards of the Coast. ©Wizards of the Coast LLC.

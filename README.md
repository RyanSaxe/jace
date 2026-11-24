# jace

jace learns to draft, build decks, and play Limited Magic: The Gathering by
self-play on the `managym` Rust rules engine.

> [!NOTE]
> Work in progress. Development currently focuses on Phase A (gameplay
> self-play); Phase B connects drafting and deck building into the end-to-end
> loop.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/approach-dark.svg">
  <img alt="Jace architecture: shared program card encoder feeding Draft, Deck Build, Play, Self-Play RL, and Evaluation." src="assets/approach-light.svg">
</picture>

## Approach

Limited Magic couples three imperfect-information decisions across hundreds of
cards: drafting from unseen boosters, building a 40-card deck, and playing with
hidden hands, libraries, and decklists.

- **Shared program card encoder.** Models read `managym` engine program tokens
  alone, with no per-card IDs, fixed card heads, or usage stats, so unseen cards
  generalize from their mechanics.
- **Draft, build, and play.** A causal transformer drafts in 8-seat pods, a
  3-stage masked set transformer builds and scores legal decks, and a
  closed-decklist actor-critic plays with a 3-class Win/Draw/Loss (WDL) critic.
- **Stabilized self-play and evaluation.** Training pairs undiscounted WDL GAE,
  top-fraction advantage filtering, a damped dual-KL loss, even-odds snapshots,
  color-balanced curricula, and counterfactual forks. Evaluation uses mirrored
  paired deals, adaptive stopping, forensics, and probe diffs.

## Quick start

Requires Python 3.13, [uv](https://docs.astral.sh/uv/), and Rust (to compile
`managym`).

```sh
git clone https://github.com/RyanSaxe/jace.git
cd jace
uv sync
uv run pytest -q                              # unit tests
uv run python -m jace.train.ten_decks --help  # 10-deck lab (no external data)
uv run python -m jace.train.full_data --help  # full 17Lands self-play
uv run python -m jace.pretrain.draft --help   # drafter & card encoder pretraining
uv run python -m jace.pretrain.decks --help   # deck builder & score pretraining
uv run scripts/colab_run.py --help            # Google Colab GPU launcher
```

See [Install](docs/getting-started/install.md) for setup and 17Lands data.

## Documentation

| Page                                                      | Topic                                             |
| --------------------------------------------------------- | ------------------------------------------------- |
| [Install](docs/getting-started/install.md)                | Setup, tests, smoke run, 17Lands data             |
| [Architecture](docs/concepts/architecture.md)             | Data flow, package layers, boundaries, CLIs       |
| [Gameplay RL](docs/concepts/gameplay-rl.md)               | Policy backbone, self-play loop, curricula, forks |
| [Limited models](docs/concepts/limited-models.md)         | Card encoder, drafter, deck builder, deck score   |
| [Evaluation](docs/concepts/evaluation.md)                 | Mirrored deals, adaptive judge, forensics, diffs  |
| [Running experiments](docs/guides/running-experiments.md) | Training, Colab, screening, replays, custom loops |
| [Data](docs/reference/data.md)                            | 17Lands/MTGJSON loaders, caching, CRC-32 splits   |
| [Run records](docs/reference/run-records.md)              | Gameplay folders, resume, locked pretraining runs |
| [Sources](docs/reference/sources.md)                      | Papers, repos, and datasets behind the design     |
| [Contributing](docs/contributing/README.md)               | Setup, static checks, boundaries, testing         |

## Licence

A licence has not been chosen yet.

## Notice

jace is unofficial Fan Content permitted under the Fan Content Policy. Not
approved/endorsed by Wizards. Portions of the materials used are property of
Wizards of the Coast. ©Wizards of the Coast LLC.

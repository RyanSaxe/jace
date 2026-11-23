# jace

jace learns to draft, build decks, and play Limited Magic: The Gathering by self-play on the `managym` Rust rules engine.

> [!NOTE]
> Work in progress. Development is currently focused on Phase A (gameplay self-play); Phase B will connect drafting and deck building into the end-to-end loop.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/approach-dark.svg">
  <img alt="Jace unified architecture: a shared program card encoder feeds Draft, Deck Build, and Play, closing the loop through Self-Play RL and Evaluation." src="assets/approach-light.svg">
</picture>

## The problem and approach

Limited Magic combines three coupled decisions under imperfect information: drafting a card pool from unseen boosters, assembling a legal 40-card deck, and playing games with hidden hands, libraries, and decklists across hundreds of distinct cards.

- **One shared program card encoder.** Every model reads cards from their `managym` engine program tokens alone, with no per-card IDs, no fixed card output heads, and no usage statistics, so unseen cards generalize directly from their mechanics.
- **Draft, build, and play in one loop.** A causal transformer drafts in 8-seat pods, a 3-stage masked set transformer builds and scores legal decks, and an actor-critic policy plays closed-decklist games with a 3-class Win/Draw/Loss critic.
- **Stabilized self-play and paired evaluation.** Self-play combines undiscounted WDL GAE, top-fraction advantage filtering, a damped dual-KL policy loss, even-odds historical snapshots, color-balanced card-pool curricula, and batched counterfactual forks. Evaluation measures checkpoints with mirrored paired deals, adaptive sequential stopping, behavioral forensics, and probe-state diffs.

## Quick start

jace requires Python 3.13, [uv](https://docs.astral.sh/uv/), and a Rust toolchain to compile `managym`.

```sh
git clone https://github.com/RyanSaxe/jace.git
cd jace
uv sync
uv run pytest -q                                       # unit tests
uv run python -m jace.loops.train_ten_decks --help     # 10-deck lab (no external data)
uv run python -m jace.loops.train_full_data --help     # full 17Lands self-play
uv run python -m jace.loops.pretrain --help            # drafter & card encoder pretraining
uv run python -m jace.loops.pretrain_decks --help      # deck builder & deck score pretraining
```

See [Install](docs/getting-started/install.md) for setup and 17Lands dataset instructions.

## Documentation

| Page                                                    | Topic                                                       |
| ------------------------------------------------------- | ----------------------------------------------------------- |
| [Install](docs/getting-started/install.md)              | Requirements, tests, quick lab run, and 17Lands setup       |
| [Architecture](docs/concepts/architecture.md)           | System flow, package layers, boundaries, and entry points   |
| [Gameplay RL](docs/concepts/gameplay-rl.md)             | Policy backbone, self-play update loop, curricula, and forks |
| [Limited models](docs/concepts/limited-models.md)       | Program card encoder, drafter, deck builder, and deck score |
| [Evaluation](docs/concepts/evaluation.md)               | Mirrored deals, adaptive judging, forensics, and probe diffs |
| [Running experiments](docs/guides/running-experiments.md) | Training, pretraining, screening, replays, and custom loops |
| [Data](docs/reference/data.md)                          | 17Lands and MTGJSON loaders, caching, and deterministic splits |
| [Run records](docs/reference/run-records.md)            | Gameplay run folders, atomic resume, and locked pretraining runs |
| [Sources](docs/reference/sources.md)                    | Papers, repositories, and datasets behind the design        |
| [Contributing](docs/contributing/README.md)             | Development setup, static checks, boundaries, and testing   |

## Licence

A licence has not been chosen yet.

## Notice

jace is unofficial Fan Content permitted under the Fan Content Policy. Not
approved/endorsed by Wizards. Portions of the materials used are property of
Wizards of the Coast. ©Wizards of the Coast LLC.

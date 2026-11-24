# Documentation

Start with [Architecture](concepts/architecture.md) for the end-to-end system
flow and package layout, then dive into [Gameplay RL](concepts/gameplay-rl.md),
[Limited models](concepts/limited-models.md), and
[Evaluation](concepts/evaluation.md).

| Page                                               | Topic                                                          |
| -------------------------------------------------- | -------------------------------------------------------------- |
| [Install](getting-started/install.md)              | Requirements, tests, quick lab run, and 17Lands setup          |
| [Architecture](concepts/architecture.md)           | System flow, package layers, boundaries, and entry points      |
| [Gameplay RL](concepts/gameplay-rl.md)             | Policy backbone, self-play update loop, curricula, and forks   |
| [Limited models](concepts/limited-models.md)       | Program card encoder, drafter, deck builder, and deck score    |
| [Evaluation](concepts/evaluation.md)               | Mirrored deals, adaptive judging, forensics, and probe diffs   |
| [Running experiments](guides/running-experiments.md) | Training, pretraining, screening, replays, and custom loops  |
| [Data](reference/data.md)                          | 17Lands and MTGJSON loaders, caching, and deterministic splits |
| [Run records](reference/run-records.md)            | Gameplay run folders, atomic resume, and locked pretraining runs |
| [Sources](reference/sources.md)                    | Papers, repositories, and datasets behind the design           |
| [Contributing](contributing/README.md)             | Development setup, static checks, boundaries, and testing      |

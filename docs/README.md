# Documentation

Start with [Architecture](concepts/architecture.md) for the package layout, then read [Gameplay RL](concepts/gameplay-rl.md) and [Evaluation](concepts/evaluation.md) to see how Stage A trains and measures the gameplay player. Read [Limited models](concepts/limited-models.md) for the pretrained Stage B drafter and deck models.

| Page                                               | Topic                                                         |
| -------------------------------------------------- | ------------------------------------------------------------- |
| [Install](getting-started/install.md)              | Requirements, tests, 10-deck quick run, and 17Lands data      |
| [Architecture](concepts/architecture.md)           | Package layers, AST import boundaries, and CLI entry points   |
| [Gameplay RL](concepts/gameplay-rl.md)             | Policy backbone, `jace.rl`, multi-process collection, forks   |
| [Evaluation](concepts/evaluation.md)               | Mirrored deals, adaptive harness, forensics, and probe diffs  |
| [Limited models](concepts/limited-models.md)       | Content card encoder, drafter, deck builder, and deck score   |
| [Running experiments](guides/running-experiments.md) | Training runs, Academy replays, screening, and custom loops |
| [Data](reference/data.md)                          | 17Lands and MTGJSON pipeline and Avatar set counts            |
| [Run records](reference/run-records.md)            | Academy-native `jace.runs` and locked `jace.tracking` layouts |
| [Sources](reference/sources.md)                    | Papers, repositories, and datasets behind the design          |
| [Status](status.md)                                | Current results, verified components, and open work           |
| [Contributing](contributing/README.md)             | Dev setup, automated checks, boundary rules, and tests        |

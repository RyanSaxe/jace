# Agents

## Package layout

jace separates pure models and RL math from engine collection, evaluation,
storage, and runnable CLIs. Imports flow downward only and are enforced by
`.byor/rules/project/jace.*.yml`.

| Package / path | Owns | Engine imports |
| --- | --- | --- |
| `src/jace/train/` | Top-level gameplay self-play CLIs (`full_data`, `ten_decks`, `harness`) | Yes |
| `src/jace/pretrain/` | Supervised pretraining CLIs and helpers (`draft`, `decks`, `common`, `formats`, `schedules`, `supervised`, `deck_measures`) | No |
| `src/jace/forks/` | Counterfactual fork collection, batched playouts, and horizon audits (`collect`, `batch`, `audit`) | Yes |
| `src/jace/eval/` | Mirrored paired deals, parallel adaptive judging, forensics, and probe diffs (`deals`, `parallel`, `forensics`, `diffs`) | Yes |
| `src/jace/collect/` | v11 `ActorCritic` factory, deck/curriculum loading, and in-process / multi-process collectors (`agent`, `decks`, `local`, `pool`) | Yes |
| `src/jace/rl/` | Pure-tensor RL math (`credit`, `losses`, `select`, `schedules`, `ema`, `pool`, `prefetch`, `diagnostics`) and collection bridge (`loop`) | Only `loop.py` |
| `src/jace/draft/` | Booster specs, multi-pod draft state machine, and `DrafterModel` | No |
| `src/jace/decks/` | `DeckBuilder`, `DeckScore`, `BuildRules`, and batch-sampled deck search | No |
| `src/jace/storage/` | Gameplay run folders (`gameplay.py`) and file-locked pretraining run records (`pretrain.py`, leaf) | Only `gameplay.py` |
| `src/jace/cards/` | `CardPool`, program-based `CardEncoder`, and frozen `CardTable` (leaf) | No |
| `src/jace/data/` | 17Lands and MTGJSON loaders, deterministic splits, program tokens, and engine `catalog.py` seam (leaf) | Only `catalog.py` |
| `src/jace/modules.py` | Shared transformer builder, packed `encode_real`, and device selection (leaf) | No |
| `scripts/` | Standalone operational scripts with PEP 723 metadata (such as `colab_run.py`) | No |

## Engine and manabot boundaries

- Keep `manabot` and `managym` imports strictly inside `jace.data.catalog`,
  `jace.rl.loop`, `jace.collect`, `jace.eval`, `jace.forks`,
  `jace.storage.gameplay`, and `jace.train`.
- Never import `manabot.training.updates`, `manabot.training.loops`, or
  `manabot.recipes`; jace owns its RL algorithms in `jace.rl`.
- Keep `jace.collect` free of `jace.eval` imports, and keep
  `jace.storage.pretrain`, `jace.cards`, `jace.data`, and `jace.modules` free of
  higher-layer imports.
- Outside `jace.draft`, only `jace.pretrain` and `jace.train` may import
  `jace.draft`, so constructed play requires no drafting code.
- Nothing outside `jace.pretrain` may import `jace.pretrain`, and nothing
  outside `jace.train` may import `jace.train`.

## Sources

`docs/reference/sources.md` lists the papers, repositories, datasets and
documents behind jace's design.

- When a source changes what you build, add it in the same commit.
- Each entry gives the linked title (the arXiv abstract page when there is one),
  the authors and year, what the source found, and which part of jace uses it.
- Add only sources you have read, and check the title, authors, year and finding
  against the source itself.
- When jace stops following a source, keep the entry: say what jace does
  instead, or mark it "Decided against" with the reason.

# Agents

## Setup and checks

```sh
uv sync --extra studio
prek install
prek run --all-files
uv run --locked --extra studio pytest -q
npm --prefix web test && npm --prefix web run build
```

## Workflow

- Work lands on `main` through pull requests. There is no release process yet.
- Write commit messages in Conventional Commits format.
- Ask before adding a dependency, merging a pull request, or approving a
  workflow run.
- Do not add backwards-compatibility shims before the first release.
- Keep unit tests fast, self-contained, and independent of downloaded 17Lands
  data.

## Package layout

jace separates pure models and RL math from engine collection, evaluation,
storage, and runnable CLIs. Imports flow downward only and are enforced by
`.byor/rules/project/jace.*.yml`.

| Package / path | Owns | Engine imports |
| --- | --- | --- |
| `src/jace/train/` | Top-level gameplay self-play CLIs (`full_data`, `ten_decks`, `harness`) | Yes |
| `src/jace/studio/` | Local evaluation workbench backend: FastAPI, DuckDB/Parquet index, `managym` replay, board serialization, playout/compare analysis, Arena/puzzle benchmarks, belief/search API, Scryfall cache, and Fork API (`app`, `runs`, `index`, `board`, `replay`, `inference`, `analysis`, `benchmarks`, `belief_api`, `scryfall`) | Yes |
| `src/jace/pretrain/` | Supervised pretraining CLIs and helpers (`draft`, `decks`, `common`, `formats`, `schedules`, `supervised`, `deck_measures`) | No |
| `src/jace/forks/` | Counterfactual fork collection, batched playouts, closed-decklist local search, search Arena sweeps, and horizon audits (`collect`, `batch`, `search`, `arena`, `audit`) | Yes |
| `src/jace/belief/` | Pure-tensor closed-decklist opponent belief models (`FormatPrior`, `BayesianDeckPoolPrior`, `SetTransformerPrior`, `BeliefNet`, `project_belief_overrides`) and Turn-0 `MatchupPredictor` `V_0` (`types`, `prior`, `model`, `project`, `v0`) | No |
| `src/jace/eval/` | Mirrored paired deals, parallel adaptive judging, Bradley-Terry Arena ledger, local report, tactical puzzles, forensics, and probe diffs (`deals`, `parallel`, `arena`, `report`, `puzzles`, `forensics`, `diffs`) | Yes |
| `src/jace/collect/` | v11 `ActorCritic` factory, deck/curriculum loading, and in-process / multi-process collectors (`agent`, `decks`, `local`, `pool`) | Yes |
| `src/jace/rl/` | Pure-tensor RL math (`credit`, `losses`, `select`, `schedules`, `ema`, `pool`, `prefetch`, `diagnostics`) and collection bridge (`loop`) | Only `loop.py` |
| `src/jace/draft/` | Booster specs, multi-pod draft state machine, and `DrafterModel` | No |
| `src/jace/decks/` | `DeckBuilder`, `DeckScore`, `BuildRules`, and batch-sampled deck search | No |
| `src/jace/storage/` | Gameplay run folders (`gameplay.py`) and file-locked pretraining run records (`pretrain.py`, leaf) | Only `gameplay.py` |
| `src/jace/cards/` | `CardPool`, program-based `CardEncoder`, and frozen `CardTable` (leaf) | No |
| `src/jace/data/` | 17Lands and MTGJSON loaders, deterministic splits, program tokens, and engine `catalog.py` seam (leaf) | Only `catalog.py` |
| `src/jace/modules.py` | Shared transformer builder, packed `encode_real`, and device selection (leaf) | No |
| `scripts/` | Standalone operational scripts with PEP 723 metadata (such as `colab_run.py`) | No |

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

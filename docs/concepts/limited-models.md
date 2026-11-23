# Limited models

Drafting and deck building share one program-based card representation so every model scores cards from their mechanics alone, with no per-card IDs, no fixed card output heads, and no usage statistics.

## Shared program card encoder

```mermaid
flowchart TD
    Catalog["managym Engine Programs\n((kind, value) token pairs)"] --> Emb["Token + Position Embeddings\n(padding / no-program / unknown)"]
    Emb --> Tx["Packed Transformer\n(encode_real)"]
    Tx --> Pool["Masked Mean Pool + LayerNorm\n(CardEncoder -> frozen CardTable)"]
    Pool --> Drafter["DrafterModel\n(Causal Transformer)"]
    Pool --> Builder["DeckBuilder\n(3-Stage Masked Set Transformer)"]
    Pool --> Score["DeckScore\n(Set Transformer Win Predictor)"]
```

| Symbol / constant   | Role                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------- |
| `CardEncoder`       | Encodes each card's `managym` program tokens into a `width`-dim vector (default `64`)  |
| `CardTable`         | Parameter-free frozen snapshot (`frozen_table(encoder)`) shared by downstream models   |
| `CardVectors`       | Protocol satisfied by both `CardEncoder` and `CardTable`                               |
| Special token IDs   | `0` = `PADDING_TOKEN`, `1` = `NO_PROGRAM_TOKEN`, `2` = `UNKNOWN_TOKEN`, `3+` = catalog pairs (`MAX_PROGRAM_TOKENS = 256`) |
| `own_parameters`    | Excludes shared card-encoder weights from a model's optimizer so embeddings step once  |

## Drafter (`jace.draft`)

`DrafterModel` in [`jace/draft/model.py`](../../src/jace/draft/model.py) is a causal transformer (default width `256`, `6` layers, `8` heads, dropout `0.2`) over a seat's pick sequence:

```mermaid
flowchart LR
    History["Earlier Picks\n(picked card + mean pack + position)"] --> CausalTx["Causal Transformer\n(DrafterModel)"]
    Query["Current Pack Query\n(learned query + mean pack + position)"] --> CausalTx
    CausalTx --> Dot["Scaled Dot Product\nwith candidate(card_vec)"]
    Dot --> Mask["Mask Outside Pack (-inf)\n-> Pick Logits"]
```

- **Single-pass training (`whole_draft`).** Interleaves picked tokens and query tokens under one `2t`-length causal mask, producing exact pick-by-pick logits for an entire draft in one forward pass.
- **Parallel pods (`jace.draft.pod`).** `advance` steps batches of 8-seat booster pods from `Picking` to `Complete` in one batched model call per pick.

## Deck builder and deck score (`jace.decks`)

Both models read distinct cards and `log1p`-scaled copy counts through `CardSetEncoder` (permutation-invariant transformer, default width `128`, `3` layers, `4` heads).

```mermaid
flowchart LR
    Pool["Drafted Pool\n(pool & deck counts)"] --> S1["1. Land Total Head\n(sinusoidal share over\nmin_lands..max_lands)"]
    S1 --> S2["2. Spells & Nonbasics\n(Plackett-Luce Gumbel-max\nin rounds of 6)"]
    S2 --> S3["3. Basic Land Split\n(multinomial over\nremaining land slots)"]
    S3 --> Legal["Legal 40-Card Decks\n(batch sampled)"]
    Legal --> Rank["DeckScore\n(deduplicate & rank best first)"]
```

| Component       | Details                                                                                         |
| --------------- | ----------------------------------------------------------------------------------------------- |
| `DeckScore`     | Pools encoded deck tokens to a single win-probability logit; trained with binomial `score_loss` |
| `DeckBuilder`   | Enforces `BuildRules` via hard masks at every stage so every sampled deck is legal              |
| `sample_builds` | Samples batches of legal decks with exact log-probabilities                                     |
| `build_log_probs` | Recomputes differentiable log-probabilities in one pass for teacher forcing or policy gradient |
| `build_decks`   | Samples candidates, deduplicates, scores each unique deck once with `DeckScore`, keeps top `k`  |

## Limited formats (`jace.loops.formats`)

`LimitedFormat` centralizes format geometry so no model hardcodes set sizes:

| Format        | Seats | Boosters          | Deck size | Land range |
| ------------- | ----- | ----------------- | --------- | ---------- |
| `TLA_LIMITED` | 8     | 3 × 14-card packs | 40        | 15 to 18   |

# Limited models

Stage B covers the three models needed outside a game: a shared content-based card encoder (`jace.cards`), a booster-pod drafter (`jace.draft`), and an autoregressive deck builder plus deck score (`jace.decks`). Each model is pretrained on 17Lands human data and reads cards without a fixed per-card output head so unseen cards from a new set can be scored from their own content.

## Shared content-based card representation

`CardEncoder` in [`jace/cards/encoder.py`](../../src/jace/cards/encoder.py) maps every card to a single vector (default width `64`) from three inputs known before anyone plays the card:

| Input     | Source                                                                          |
| --------- | ------------------------------------------------------------------------------- |
| `text`    | 384-dim embedding of the card's printed text from `sentence-transformers/all-MiniLM-L6-v2` (card name masked to `"this card"`) |
| `program` | Tokenized `managym` engine program (`(kind, value)` pairs read by a 1-layer transformer with learned positions) |
| `printed` | Standardized mana value, power, toughness (with missing flags), plus 0/1 flags for color, type, rarity, and bonus sheet |

No 17Lands usage statistics enter the card vector. `CardEncoder` projects the text and printed inputs, encodes the program tokens with a transformer (`encode_real` in `jace.modules`), joins the three parts through a GELU MLP, and normalizes the result.

Drafter pretraining (`jace.loops.pretrain`) trains `CardEncoder` alongside `DrafterModel` using separate optimizers (`own_parameters` excludes the shared embedding from the drafter's optimizer). Downstream deck pretraining freezes the encoder output into a parameter-free `CardTable` (`frozen_table(encoder)`). Every model accepts either one through the `CardVectors` protocol.

## Drafter

`DrafterModel` in [`jace/draft/model.py`](../../src/jace/draft/model.py) is a causal transformer (default width `256`, `6` layers, `8` heads, dropout `0.2`) over a seat's draft history:

- Each earlier pick becomes one token: projected picked-card vector + projected mean vector of the pack it came from + learned pick-position embedding.
- The current pack becomes a query token (using a learned `current` vector in place of a picked card) attending causally to all earlier picks.
- Each card in the current pack is scored by the scaled dot product of the query with `candidate(card_embedding)`, masked to `-inf` outside the pack.

`DrafterModel.whole_draft` scores every pick of a recorded draft in a single parallel forward pass by placing picked tokens and query tokens into one `2t`-length causal attention mask, producing the exact logits `forward` gives pick by pick.

For simulated drafting, `jace.draft.pod` runs many 8-seat booster pods in parallel (`advance` moves pods from `Picking` to `Complete`), batching all active seats into one model call per pick.

## Deck builder and deck score

Both deck models read cards through `CardSetEncoder` in [`jace/decks/score.py`](../../src/jace/decks/score.py), a permutation-invariant transformer over distinct cards with `log1p`-scaled copy counts (default width `128`, `3` layers, `4` heads).

### Deck score

`DeckScore` reads a deck's distinct cards and copy counts, averages their encoded tokens, and outputs a single win-probability logit. `jace.loops.pretrain_decks` trains it with binomial log loss (`score_loss`) against each human deck's event wins and losses.

### Deck builder

`DeckBuilder` in [`jace/decks/builder.py`](../../src/jace/decks/builder.py) reads every non-basic card in a pool with two counts (copies in pool, copies already in deck) and builds a legal deck in three masked stages:

| Stage                      | Head / sampling                                                                  |
| -------------------------- | -------------------------------------------------------------------------------- |
| 1. Land total              | Softmax over legal totals (`[min_lands, max_lands]`), reading the land share through sinusoidal frequencies `(1, 2, 4, 8, 16)` so the head can peak cleanly at interior counts such as 17 |
| 2. Spells and nonbasic lands | Drawn in rounds of `spells_per_round` (default `6`) without replacement via Plackett-Luce (Gumbel-max over remaining copies), re-encoding the deck each round so curve and color choices stay coherent |
| 3. Basic lands             | Multinomial split of the remaining land slots across legal basic lands           |

Masks enforce `BuildRules` (deck size, pool copy limits, and land range) at every step, so every sampled deck is legal. `sample_builds` draws whole batches of decks with their exact log-probabilities, and `build_log_probs` recomputes those log-probabilities with gradients in one batched call for teacher forcing (`builds_from_decks` shuffles spells into rounds with a spell last) or policy-gradient updates.

`build_decks` in [`jace/decks/search.py`](../../src/jace/decks/search.py) samples many candidate decks from a pool in one batch, deduplicates them, scores each distinct deck once with `DeckScore`, and returns the top `keep` builds best first.

## Limited formats

`LimitedFormat` in [`jace/loops/formats.py`](../../src/jace/loops/formats.py) collects all format-specific sizes in one record so no model hardcodes them:

- `TLA_LIMITED`: 8 seats, 3 boosters of 14 picks (`TLA_PLAY_BOOSTER`), 40-card decks with 15 to 18 lands.
- `deck_rules(limited_format("TLA"), cards)` produces the `BuildRules` for that set's basic and nonbasic land IDs.

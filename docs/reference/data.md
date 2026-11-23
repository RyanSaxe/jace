# Data

jace reads public Premier Draft exports from [17Lands](https://www.17lands.com/public_datasets) (Creative Commons Attribution 4.0) and printed card attributes from [MTGJSON](https://mtgjson.com/).

## Downloading a set

The [spells](https://github.com/oelarnes/spells) package downloads 17Lands files and MTGJSON attributes and converts them to Parquet:

```sh
uv run spells add TLA
```

Files live in `~/.local/share/spells` (or `$SPELLS_DATA_HOME`). Avatar: The Last Airbender (`TLA`) takes about 300 MB. jace never downloads data automatically. If a set is missing, `jace.data.read_set` raises an error naming `uv run spells add <set>`.

| Table           | One row per | Columns jace reads                                                     |
| --------------- | ----------- | ---------------------------------------------------------------------- |
| Draft data      | Pick        | Pack contents, card taken, player skill, event result                  |
| Game data       | Game        | Main deck, sideboard, build index, game outcome                        |
| Card attributes | Card        | Rarity, colors, types, mana value, power, toughness, rules text        |

## Building cached tensors

`jace.data` turns Parquet tables and engine definitions into cached PyTorch archives so training loops never re-parse Parquet on startup:

```python
from jace.data import card_table, load_decks, load_drafts, read_set
from jace.data.catalog import engine_cards

data = read_set("TLA")
programs = {card.name: card.program for card in engine_cards()}
cards = card_table(data.attributes, data.statistics, programs=programs)
drafts = load_drafts(data.picks, pool=cards.pool, cache_dir=cache_dir, held_out_share=0.1)
decks = load_decks(data.games, pool=cards.pool, cache_dir=cache_dir, held_out_share=0.1)
```

`jace.data.catalog` is the one module in `jace.data` allowed to import `managym`. Its `engine_cards` lists every deck-eligible card with its engine program, and `program_vocabulary` numbers every `(kind, value)` pair in `managym`'s token catalog.

### Deterministic train / held-out split

Both `load_drafts` and `load_decks` assign every draft a deterministic CRC-32 `split_key` (`jace.data.store.SPLIT_KEYS`). A draft belongs to the held-out split when `split_key < round(held_out_share * SPLIT_KEYS)` (`0.1` by default). Stage A gameplay evaluation (`jace.eval.human_decks.load_human_deck_catalog`) reads `decks.pt` using the exact same mask, so held-out decks stay disjoint from training across pretraining, self-play, and evaluation.

### Card encoder inputs

`jace.loops.entry_points.card_inputs` assembles `CardInputs` from three sources that exist before any games are played:

- **Printed facts (`cards.printed`):** Standardized mana value, power, and toughness (with missing-value indicator columns) plus binary flags for color, card type, rarity, and bonus sheet.
- **Text embeddings (`text_vectors`):** `card_texts` formats mana cost, type line, power/toughness, rarity, and rules text with the card's name replaced by `"this card"`. `jace.data.hugging_face.hugging_face_encoder` embeds those strings with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) and caches the result in `cache_dir`. Keeping the Hugging Face wrapper in its own module avoids importing `transformers` on ordinary `jace.data` imports.
- **Engine programs (`program_tokens`):** Maps each card's `(kind, value)` pairs to token IDs using `program_vocabulary()`. Token IDs `0`, `1`, and `2` are reserved for `PADDING_TOKEN`, `NO_PROGRAM_TOKEN`, and `UNKNOWN_TOKEN`.

`cards.ata` keeps 17Lands average pick position strictly for the untrained earliest-pick baseline in `jace.loops.pretrain`. No model input includes any 17Lands usage statistic.

### Deck extraction

Because players can edit their deck between Arena games, `load_decks` keeps one representative deck per draft: the `build_index` played in the most games (breaking ties by lowest `build_index`), with `pools = decks + sideboards` and the event's total `wins` and `losses`.

## Avatar: The Last Airbender (`TLA`)

| Quantity                             | Count                                               |
| ------------------------------------ | --------------------------------------------------- |
| Unique cards in set                  | 347 (286 main set + 61 bonus sheet)                 |
| Engine-playable cards in `managym`   | 344 (excludes Heartbeat of Spring, Lita, Mechanical Engineer, Release to Memory) |
| Recorded human drafts / picks        | 102,793 drafts / 4,214,513 picks (10,282 held-out drafts) |
| Recorded human decks in `decks.pt`   | 101,496 total (91,359 train + 10,137 held-out, of which 181 contain an unplayable bonus card) |

Arena's log omitted pack-1 pick-1 rows in the Avatar export, though the second pick's pool still contains the card taken first. `load_drafts` reconstructs pick 1 from the pick-2 pool with weight `0` so the drafter sees it in draft history without training or scoring on it.

# Data

jace reads the free public datasets that
[17Lands](https://www.17lands.com/public_datasets) publishes for each card
set under a Creative Commons Attribution 4.0 licence. They record how
people drafted and played on Arena.

## Where it comes from

The [spells](https://github.com/oelarnes/spells) package downloads the
files and converts them to Parquet. Download a set once:

```sh
uv run spells add TLA
```

spells keeps the files in `~/.local/share/spells`, or in the folder named
by `SPELLS_DATA_HOME`. Avatar: The Last Airbender (`TLA`) takes about
200 MB. jace never downloads: `read_set` raises an error that names this
command when a file is missing.

| File            | One row per | jace reads                                         |
| --------------- | ----------- | -------------------------------------------------- |
| Draft data      | Pick        | The pack, the card taken, the player's skill, the event's result |
| Game data       | Game        | The deck, the sideboard and who won                |
| Card attributes | Card        | Rarity, colours, types, mana value, rules text (from [MTGJSON](https://mtgjson.com/)) |

## What jace makes of it

`jace.data` turns the tables into a card pool and tensors:

```python
data = read_set("TLA")
cards = card_table(data.attributes, data.statistics, programs=programs)
drafts = load_drafts(data.picks, pool=cards.pool, cache_dir=folder, held_out_share=0.1)
decks = load_decks(data.games, pool=cards.pool, cache_dir=folder, held_out_share=0.1)
```

`cards` holds the `CardPool` and a starting vector for each card. The
vector holds the card's 17Lands statistics, such as its win rate when in
hand and the average pick it is taken at, and a few printed facts. A
missing statistic becomes zero, and a separate column marks it as missing.

17Lands aggregates the statistics over every draft and game of the set,
held-out drafts included. A model that reads them has seen a summary of
the held-out data, so its held-out scores are a little too high.

A card from a set nobody has played has no statistics, so two other
starting vectors use only what exists before release:

```python
from jace.data.hugging_face import hugging_face_encoder

texts = card_texts(data.attributes)
encode = hugging_face_encoder(TEXT_MODEL, device=torch.device("cpu"))
text = text_vectors(texts, encode=encode, model_name=TEXT_MODEL, cache_dir=folder)
program = program_vectors(cards.pool, width=64)
```

`card_texts` writes each card as its mana cost, type line, power and
toughness, rarity and rules text. It leaves the name out and writes "this
card" where the rules text names the card. `text_vectors` embeds the texts
with a frozen language model, by default `all-MiniLM-L6-v2` (Apache 2.0,
384 numbers per card), and saves the result in `cache_dir`, so the model
runs once per card list. `hugging_face_encoder` has its own module so that
importing `jace.data` does not load `transformers`.

`program_vectors` reads the engine program stored in the pool for each
card. It counts the program's entries and adjacent pairs of entries,
weights the ones few cards share, and keeps the 64 directions along which
cards differ most. A last column marks a card the engine has no program
for. Both functions standardise over the cards they are given, so vectors
made for different card lists are not comparable.

`python -m jace.loops.pretrain --starting-vectors text,program` trains on
any combination of `statistics`, `text` and `program`.

`drafts` holds every pack, every pick and a weight for each pick. `decks`
holds one deck per draft, its pool, and the event's wins and losses.

`load_drafts` and `load_decks` save their tensors in `cache_dir` and load
them on later calls. Use one folder per card set. Drafts are held out by a
checksum of the draft id, so the split is the same in every run.

## Avatar: The Last Airbender

The set has 347 cards: 286 of its own and 61 bonus cards. The engine can
play 288 of them, and the 59 it cannot are all bonus cards. The data holds
4,214,513 picks from 102,793 drafts, and 101,496 decks.

17Lands has no first pick for any Avatar draft. The pool at the second
pick names the card taken, so jace restores the first pick as history and
never trains or scores a model on it.

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

`cards` holds the `CardPool` and the facts printed on each card: its mana
value, power and toughness, a column that marks a missing power and one
that marks a missing toughness, and flags for its colours, types, rarity
and the bonus sheet. The three numbers are standardised over the cards that
have one. Of the 17Lands statistics, `cards` keeps only each card's average
pick position, which a baseline picker ranks by. No card vector holds a
statistic.

17Lands aggregates the statistics over every draft and game of the set,
held-out drafts included. The baseline that ranks by average pick position
has therefore seen a summary of the held-out data, so its held-out score is
a little too high.

The card encoder reads two more things about each card. Both exist before
anyone has played the card:

```python
from jace.data.hugging_face import hugging_face_encoder

texts = card_texts(data.attributes)
encode = hugging_face_encoder(TEXT_MODEL, device=torch.device("cpu"))
text = text_vectors(texts, encode=encode, model_name=TEXT_MODEL, cache_dir=folder)
program = program_tokens(cards.pool, vocabulary=program_vocabulary())
```

`card_texts` writes each card as its mana cost, type line, power and
toughness, rarity and rules text. It leaves the name out and writes "this
card" where the rules text names the card. `text_vectors` embeds the texts
with a frozen language model, by default `all-MiniLM-L6-v2` (Apache 2.0,
384 numbers per card), and saves the result in `cache_dir`, so the model
runs once per card list. `hugging_face_encoder` has its own module so that
importing `jace.data` does not load `transformers`.

`text_vectors` standardises over the texts it is given, so vectors made
for different card lists are not comparable.

`program_tokens` writes the engine program stored in the pool for each card
as token ids, one for each (kind, value) pair. `program_vocabulary`, in
`jace.engine.manabot`, numbers every pair in the engine's catalog, so a
pair's id does not depend on the pool. A card the engine cannot play gets
one token that says so, and a pair the vocabulary lacks gets a token that
marks it unknown. On Avatar the longest program has 149 pairs, and every
pair is in the vocabulary.

`card_inputs`, in `jace.loops.entry_points`, puts the text vectors, the
program tokens and the printed facts together as the `CardInputs` the
encoder reads.

`drafts` holds every pack, every pick and a weight for each pick. `decks`
holds one deck per draft, its pool, and the event's wins and losses.

`load_drafts` and `load_decks` save their tensors in `cache_dir` and load
them on later calls. Use one folder per card set. Drafts are held out by a
checksum of the draft id, so the split is the same in every run.

## Avatar: The Last Airbender

The set has 347 cards: 286 of its own and 61 bonus cards. The engine can
play 344 of them. The three it cannot are bonus cards: Heartbeat of Spring,
Lita, Mechanical Engineer and Release to Memory. A deck that holds one is
left out of play, which drops 181 of the 101,496 human decks. The data holds
4,214,513 picks from 102,793 drafts, and 101,496 decks.

17Lands has no first pick for any Avatar draft. The pool at the second
pick names the card taken, so jace restores the first pick as history and
never trains or scores a model on it.

# Data

jace reads the free public datasets that
[17Lands](https://www.17lands.com/public_datasets) publishes for each card
set under a Creative Commons Attribution 4.0 licence. They contain the
picks and games of people who drafted on Arena. The facts printed on each
card come from [MTGJSON](https://mtgjson.com/).

## Where it comes from

The [spells](https://github.com/oelarnes/spells) package downloads the
17Lands Premier Draft files and the MTGJSON card data, and converts them to
Parquet. Download a set once:

```sh
uv run spells add TLA
```

spells stores the files in `~/.local/share/spells`, or in the folder named
by `SPELLS_DATA_HOME`. Avatar: The Last Airbender (`TLA`) takes about
300 MB. jace never downloads anything itself. When a file is missing,
`read_set` raises an error that names this command.

| File            | One row per | jace reads                                         |
| --------------- | ----------- | -------------------------------------------------- |
| Draft data      | Pick        | The pack, the card taken, the player's skill, the event's result |
| Game data       | Game        | The deck, the sideboard and who won                |
| Card attributes | Card        | Rarity, colours, types, mana value, rules text (from [MTGJSON](https://mtgjson.com/)) |

## What jace makes of it

`jace.data` turns the tables into a card pool and tensors:

```python
data = read_set("TLA")
programs = {card.name: card.program for card in engine_cards()}
cards = card_table(data.attributes, data.statistics, programs=programs)
drafts = load_drafts(data.picks, pool=cards.pool, cache_dir=folder, held_out_share=0.1)
decks = load_decks(data.games, pool=cards.pool, cache_dir=folder, held_out_share=0.1)
```

`engine_cards`, in `jace.engine.manabot`, lists every card the engine can
play, and `programs` maps each one's name to its engine program.

`cards` contains the `CardPool` and a tensor of the facts printed on each
card. These are its mana value, power and toughness, standardised over the
cards that have a value, and 0/1 flags for its colours, types and rarity and
for whether it is on the bonus sheet. A card with no power gets zero there
and a 1 in a column that marks the power missing, and the same is true for
toughness.

`data.statistics` contains each card's 17Lands statistics, which spells
computes from every draft and game of the set. `cards` contains only one of
them, each card's average pick position, which a baseline picker ranks
cards by. No card vector includes a statistic. Because the held-out drafts
are among those the statistics are computed from, the baseline's held-out
score is a little too high.

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
with a frozen language model, here `TEXT_MODEL`, which is
`sentence-transformers/all-MiniLM-L6-v2` (Apache 2.0, 384 numbers per
card). It saves the result in `cache_dir`, so the model runs only once for
each list of texts. `hugging_face_encoder` has its own module so that
importing `jace.data` does not load `transformers`.

`text_vectors` standardises over the texts it is given, so vectors made
for different card lists are not comparable.

`program_tokens` writes each card's engine program as token ids, one for
each (kind, value) pair. `program_vocabulary`, in `jace.engine.manabot`,
numbers every pair in the engine's catalog, so a pair's id does not depend
on the pool. A card the engine cannot play gets one token that marks it
unplayable, and a pair the vocabulary lacks gets a token that marks it
unknown. On Avatar the longest program has 149 pairs, and every pair is in
the vocabulary.

`card_inputs`, in `jace.loops.entry_points`, puts the text vectors, the
program tokens and the printed facts together as the `CardInputs` the
encoder reads.

`drafts` contains every pack, every pick and a weight for each pick.
`decks` contains one deck for each draft that has game data, with its pool
and the games the player won and lost in the event. The deck is the build
the player used in the most games, or on a tie the one with the lowest
`build_index`. The pool is that deck plus its sideboard.

`load_drafts` and `load_decks` save their tensors in `cache_dir` and load
them on later calls. Use one folder per card set. A draft is held out by a
checksum of its id, so the split is the same in every run, and a draft
held out of the picks is held out of the decks too.

## Avatar: The Last Airbender

The set has 347 cards: 286 of its own and 61 bonus cards. The engine can
play 344 of them. The three it cannot play are bonus cards:

- Heartbeat of Spring
- Lita, Mechanical Engineer
- Release to Memory

jace does not play a human deck that contains one of them, which leaves
out 181 of the 101,496 human decks. The data contains 4,214,513 picks from
102,793 drafts, and 101,496 decks.

17Lands has no first pick for any Avatar draft, because the game client
never logged it. The pool at the second pick still contains the card taken
first. So `load_drafts` rebuilds the first pick as history. A model reads
that pick but is never trained or scored on it.

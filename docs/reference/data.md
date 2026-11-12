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
hand and the average pick it is taken at, and a few printed facts. It
stands in for a planned encoder that reads rules text. A missing statistic
becomes zero, and a separate column marks it as missing.

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

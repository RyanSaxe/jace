# Status

jace is at an early stage. A first version of every module exists and is
tested. The drafter has been trained to copy human picks. Nothing else has
been trained on Magic beyond single update steps.

## What works

`uv run pytest -q` runs 45 tests in about three seconds.

| Area        | Shown by the tests                                                  |
| ----------- | ------------------------------------------------------------------- |
| Drafting    | Pods draft by the rules, many at once, with one model call per pick |
| Deck build  | Every deck is legal, and the search escapes a local best            |
| Real games  | The manabot adapter replays a seed exactly and hides the opponent's decklist |
| Models      | All five run on a larger card pool than they were built with        |
| Training    | A player learns the toy game, and unfinished games carry over       |
| End to end  | One limited round on the toy game updates every model               |
| Real engine | The player and guess model take one training step on real games     |
| Records     | A run survives a killed writer and rejects a tampered checkpoint    |
| Human data  | Drafts and decks load from 17Lands tables, and a drafter learns a pick rule from them |

On the toy game, the share of decisions where the player picks its best
card rises from 0.36 to 0.78 in 40 iterations.

## Pretraining on human picks

One run on Avatar: The Last Airbender, on an Apple M3 Max GPU: 2,500 steps
of 256 drafts in 17 minutes, measured on 421,562 picks from 10,282 held-out
drafts. [Data](reference/data.md) describes the data.

| Picker                                  | Top 1 | Top 2 | Top 3 |
| --------------------------------------- | ----- | ----- | ----- |
| The pretrained drafter                  | 0.661 | 0.856 | 0.934 |
| The pack's card with the earliest average pick | 0.458 | 0.672 | 0.800 |
| A uniformly random card                 | 0.236 | 0.399 | 0.526 |

## Measured speed

On an Apple M3 Max CPU, with 16 games in a batch:

| Setup                                                  | Decisions a second |
| ------------------------------------------------------ | ------------------ |
| Random play through the adapter, simple decks          | About 16,400       |
| Random play through the adapter, random two-colour decks | About 3,400      |
| The untrained player and guess model choosing actions  | 366                |

## Not built yet

| Gap                         | Detail                                                     |
| --------------------------- | ---------------------------------------------------------- |
| Human data                  | Human decks load, but the deck score and deck builder are not pretrained on them |
| Card starting vectors       | Built from 17Lands statistics, not from rules text or engine programs |
| Booster rarities            | The engine has no rarity. The Avatar booster's wildcard and foil odds are placeholders |
| The limited loop on Magic   | It has run only on the toy game                            |
| A constructed loop          | Not written                                                |
| Paired evaluation           | Win rates resample single games, which overstates certainty |
| GPU                         | Only drafter pretraining has run on one                    |

## Known problems

| Problem                     | Detail                                                     |
| --------------------------- | ---------------------------------------------------------- |
| Slow deck search            | About two million deck scores and 24 seconds per pool at 200 proposals and full depth |
| Library labels              | manabot's batched games report hands but not libraries. The adapter works a library out from the board, and is one card off while the other seat holds a foretold card or controls a permanent that both decks contain |
| Board size limit            | The engine raises an error past 40 permanents, and the run stops |
| Schedule numbers            | The drafter's rates and the explore share are guesses with no source |

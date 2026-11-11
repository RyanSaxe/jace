# Status

jace is at an early stage. A first version of every module exists and is
tested. Nothing has been trained on Magic beyond single update steps.

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

On the toy game, the share of decisions where the player picks its best
card rises from 0.36 to 0.78 in 40 iterations.

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
| Human data                  | No 17Lands loading, and no pretraining of any model        |
| Card starting vectors       | Not built from rules text or engine programs. Tests use random vectors |
| Booster rarities            | The engine has no rarity. The Avatar booster's wildcard and foil odds are placeholders |
| The limited loop on Magic   | It has run only on the toy game                            |
| A constructed loop          | Not written                                                |
| Paired evaluation           | Win rates resample single games, which overstates certainty |
| GPU                         | Nothing has run on one                                     |

## Known problems

| Problem                     | Detail                                                     |
| --------------------------- | ---------------------------------------------------------- |
| Slow deck search            | About two million deck scores and 24 seconds per pool at 200 proposals and full depth |
| Scalar games                | The adapter steps games one by one in Python, because manabot's batched mode does not expose hidden hands |
| Board size limit            | The engine raises an error past 40 permanents, and the run stops |
| Schedule numbers            | The drafter's rates and the explore share are guesses with no source |

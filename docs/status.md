# Status

jace is at an early stage. A first version of every module exists and is
tested. The drafter has been trained to copy human picks, and the deck
score and deck builder have been trained on human decks. The whole loop has
run on real games once, for 60 iterations, and the player it trained beats
random play.

Every number on this page is from Avatar: The Last Airbender. Each table
says what it was measured on.

## What works

`uv run pytest -q` runs the tests.

| Area        | Shown by the tests                                                  |
| ----------- | ------------------------------------------------------------------- |
| Drafting    | Pods draft by the rules, many at once, with one model call per pick |
| Deck build  | Every deck is legal, and the search escapes a local best            |
| Real games  | The adapter on managym 0.2.3 replays a seed exactly and hides the opponent's decklist |
| Models      | All six run on a larger card pool than they were built with         |
| Training    | A player learns the toy game, a round plays exactly its list of games, and a model whose rate is zero does not change |
| End to end  | One limited round on the toy game updates every model               |
| Real engine | The player, its win chance and the guess model take one training step on real games |
| Records     | A run survives a killed writer and rejects a tampered checkpoint    |
| Human data  | Drafts and decks load from 17Lands tables, and a drafter learns a pick rule from them |
| Human decks | A deck score learns which card wins, and a deck builder learns a deck rule, on held-out decks |

On the toy game, the share of decisions where the player picks its best
card rises from 0.34 to 0.81 in 40 iterations.

## Pretraining on human picks

One run on Avatar: The Last Airbender, on an Apple M3 Max GPU: 2,500 steps
of 256 drafts in 17 minutes, measured on 421,562 picks from 10,282 held-out
drafts. [Data](reference/data.md) describes the data.

These runs read the earlier card embedding: a fixed starting vector for
each card, made from the sources each row names, adjusted by a head shared
by every card. The card encoder replaced it and has not been measured on
held-out picks yet.

| Picker                                  | Top 1 | Top 2 | Top 3 | Reads 17Lands statistics |
| --------------------------------------- | ----- | ----- | ----- | ------------------------ |
| The pretrained drafter, starting vectors from 17Lands statistics | 0.661 | 0.856 | 0.934 | Yes |
| The same, from card text                | 0.681 | 0.872 | 0.944 | No  |
| The same, from card text and engine program | 0.682 | 0.873 | 0.944 | No  |
| The same, from all three                | 0.683 | 0.873 | 0.945 | Yes |
| The same, from engine program alone     | 0.619 | 0.817 | 0.907 | No  |
| The pack's card with the earliest average pick | 0.458 | 0.672 | 0.800 | Yes |
| A uniformly random card                 | 0.236 | 0.399 | 0.526 | No  |

The rows marked Yes are a little too high. 17Lands aggregates its
statistics over every draft of the set, the held-out drafts included, so
those pickers have seen a summary of the picks they are scored on. The
other rows have not.

A card that ties with the human's card counts against the pick. The
drafter rows were measured before that rule and have not been measured
again; the earliest-average-pick row was, and did not change.

## Pretraining on human decks

Two runs on the same set and GPU, both starting from the pretrained
drafter's card embedding, the earlier one built from 17Lands statistics:
2,000 steps of 256 decks for the deck score and 3,000 steps of 32 decks for
the deck builder, in about 10 minutes. One run trained the embedding and
the other kept it fixed. Deck pretraining now always keeps the card vectors
fixed.

| Measure on held-out decks                           | Embedding trained | Embedding fixed | Baseline |
| --------------------------------------------------- | ----------------- | --------------- | -------- |
| Deck score: log loss per game, 10,137 decks         | 0.6789            | 0.6788          | 0.6860, the overall win rate for every deck |
| Deck score: correlation with the win rate, 6,816 decks of five or more games | 0.258 | 0.259 | 0 |
| Deck builder: share of the human's non-basic cards in its likeliest deck, 300 pools | 0.745 | 0.734 | 0.628, a random legal deck |
| Deck builder: the same share in one sampled deck    | 0.791             | 0.786           | 0.628    |
| Drafter: top-1 pick accuracy with this embedding    | 0.578             | 0.661           |          |

Every model in this table reads the statistics vectors, and those include
card win rates over the held-out decks' games. The deck score's numbers are
therefore a little too good, as the rows marked Yes above are.

Training the embedding here costs the drafter 8 points of pick accuracy
and gains the deck models almost nothing. Taking the likeliest card at
every step builds a worse deck than sampling, because it adds too many
basic lands: 17.3 on average where the humans play 15.3.

## Training on real games

Two early runs of `jace.loops.train_limited` on a Mac CPU. The player was
scored against uniformly random play on held-out human decks, about 80
games per point. They came before the loop played every game to its end,
shuffled its updates and stopped training the embedding, so they say
little about the loop as it is now.

| Player's settings                                           | Iterations | Win rate against random |
| ----------------------------------------------------------- | ---------- | ----------------------- |
| Credit horizon 0.95, learning rate 3e-4, no advantage filter | 60, in 24 minutes and 5,004 games | From 54% to 77% |
| Ataraxos's published settings: credit horizon 0.5, learning rate 1e-4, the top quarter of advantages | 42 | Stayed near 50% |

The credit horizon is `--lam-advantage`, and the filter is
`--keep-quantile`. Which of the three settings makes the difference is
being measured and is not known yet.

Both runs came before a bug was fixed. The optimizers of the deck score
and the deck builder also held the card embedding, so each of their
updates stepped the embedding twice.

## Measured speed

On an Apple M3 Max CPU:

| Setup                                                  | Decisions a second |
| ------------------------------------------------------ | ------------------ |
| Random play through the adapter, simple decks, 16 games in a batch | About 16,400 |
| Random play through the adapter, real decks            | About 2,000 to 4,000 |
| The untrained player and guess model choosing actions, 16 games in a batch | 366 |

## Not built yet

| Gap                         | Detail                                                     |
| --------------------------- | ---------------------------------------------------------- |
| Cards from other sets       | The card encoder reads only what a card has before anyone plays it, but it has been trained only on cards from the training drafts. A card with no play data is untested |
| Booster rarities            | The engine has no rarity. The Avatar booster's wildcard and foil odds are placeholders |
| Scoring the other models in self-play | The training loops score only the player, on paired deals against random play, scripted greedy and the heuristic player |
| A constructed loop          | `train_player` plays human decks. None builds its own constructed decks |
| The card vectors in self-play | Not trained, as a working choice. A drafter meant to be frozen fell from 65.5% to 50.7% agreement with human picks when another model's loss moved the earlier embedding, so only the drafter's pretraining trains the card encoder. Whether and how the card vectors should learn from self-play is undecided |
| GPU                         | Pretraining and the update of `train_player` can run on one. `train_limited` uses the CPU |

## Known problems

| Problem                     | Detail                                                     |
| --------------------------- | ---------------------------------------------------------- |
| Slow deck search            | About two million deck scores and 24 seconds per pool at 200 proposals and full depth |
| Library labels              | manabot's batched games report hands but not libraries. The adapter works a library out from the board, and is one card off while the other seat holds a foretold card or controls a permanent that both decks contain |
| Abandoned games             | A game the engine fails on, cuts short, or whose board outgrows the board arrays is abandoned and replaced. Its decisions teach nothing. With 80 cards and 64 permanents a seat, that was 15 of 5,491 games of a scripted player on human decks |
| Held-out statistics         | 17Lands statistics cover the held-out drafts too. They inflate the held-out score of the earliest-average-pick baseline, the one picker that still reads them, and of every measurement above marked as reading them |
| Schedule numbers            | The drafter's rates and the explore share are guesses with no source |

# Deck building

The deck builder samples whole decks from a pool, the deck score ranks
them, and both models learn from the decks the loop plays. This page
follows one pool from its format's rules to the updates its games lead to.

## The format sets the sizes

`LimitedFormat`, in `jace.loops.formats`, is a record of every number that
sizes one limited format:

- the seats of a pod, the boosters each seat opens and the picks it takes
  from each
- the booster spec, the odds of every slot of a booster
- the deck size and the range of lands a deck may have

`TLA_LIMITED` is the format of the set Avatar: The Last Airbender. It has
eight seats, three boosters of 14 picks, and 40-card decks of 15 to 18
lands. Every entry point looks its format up by set code with
`limited_format`, and no model fixes any of these numbers.

`deck_rules(limited_format, cards)` turns the format into the `BuildRules`
of one card set, in which the basics and nonbasic lands are that set's
card ids. A legal deck has exactly `size` cards and a land count within
the land range. It has no more copies of a card than its pool has, except
basics, which are unlimited. When a pool has too few spells to fill a deck
within the land range, its deck instead contains every spell the pool
has, and lands fill the slots left.

## How the builder writes a deck

`DeckBuilder` reads each card of the pool except the basics as a token
with two amounts: its copies in the pool and its copies already in the
deck. It writes a deck in three kinds of choice:

| Choice           | Made by                     | Odds                                          |
| ---------------- | --------------------------- | --------------------------------------------- |
| The land total   | The land head, once         | A softmax over the totals the rules allow     |
| Spells and nonbasic lands | Each card's own encoding, a round at a time | Plackett-Luce over the copies left |
| The basics       | The basics head, once       | A multinomial split of the land slots left    |

The spell slots are the deck size less the land total. The builder fills
them in rounds. Each round draws up to `spells_per_round` copies, one at a
time and without replacement, from the scores of one call of the model.
Every copy left is an option with its card's score, so a card with more copies left
is more likely to be drawn. This is the Plackett-Luce model of drawing
without replacement. The next round encodes the pool again with those
copies in the deck, so the model can score a sixth four-drop lower than
the first. A nonbasic land can be drawn with the spells while the deck has
fewer nonbasic lands than its land total. The rounds stop when the spell
slots are full. Every choice is masked by the rules, so every deck is
legal.

```python
builds, log_probs = sample_builds(
    builder, pool_counts.expand(256, -1), rules=rules, spells_per_round=6, temperature=1.0, generator=generator
)
```

`builds` contains every choice of every deck. `log_probs` contains each
deck's log-probability under the sampling at `temperature`, summed over
its land choice, its Plackett-Luce draws and its basics split.
`build_log_probs` gives the same numbers with gradients for any recorded
builds, along with each build's entropy. It encodes every round of every
build in one call, because the recorded order of the draws gives the deck
at the start of each round.

## The search

`build_decks` samples `--deck-samples` decks for a pool in one batch, 256
by default, and scores each distinct deck once with the deck score. It
returns at most `--blocks` distinct decks, best first. When a pool has
fewer distinct decks than blocks, the limited loop cycles through the
ones it has.

An iteration of the limited loop plays its games in `--blocks` blocks, and
in each block every pool plays one of its builds. In block `b`, pool `p`
plays its build `(p + b) % blocks`, as `block_builds` assigns them, so each
of a pool's builds plays in exactly one block. With at least as many pools
as blocks, every block then contains builds of every rank. A pool's builds
then play against the same mix of opponents, and their win rates can be compared. If
block `b` played every pool's build `b` instead, the builds in a block would
all share one rank. Each block's win rates would then average one half,
whatever that rank, and no rank could be shown to be better than another.

A pool's best build is the one the deck score ranked first, and its win
rate is the drafter's reward.

## Two deck scores

The deck score trained on human decks predicts how human decks did in human
hands. The loop never updates that score. From the first iteration, it
trains a learning copy of it on the decks the loop plays.

The loop tests both scores on each iteration's decks before the learning
copy trains on them. Each score predicts those decks' records, and its log
loss on them is its test loss. Before the loss is taken, each score's
logits are shifted so that their mean over the decks, each deck weighted
by its games, is zero. The test then compares only how the two scores rank
decks, which is all the search uses them for.

Without the shift, the test would also reward a learning copy that only
corrects a bias. Every self-play game is between two decks of the same
iteration, so those decks win half their games on average. The human
decks the frozen score was fit to won 56% of theirs. A score that predicts
56% where the truth is 50% loses about 0.007 nats a game to that bias
alone, which is more than the switch margin below. Correcting a constant
bias does not change how the search ranks a pool's decks, so it should
not cause the switch.

The search reads the human-trained score until the learning copy's mean
test loss over the last `--switch-iterations` iterations is at least
`--switch-margin` nats a game below the human-trained score's. From then
on it reads the learning copy for good. The switch is one-way so that the
search cannot flip between two rankings from one noisy iteration to the
next.

Every iteration, the loop logs both test losses and whether the search
reads the learning copy, as `deck_score/human_test_loss`,
`deck_score/learning_test_loss` and `deck_score/reads_learning`.

| Flag                  | Default | Meaning                                                |
| --------------------- | ------- | ------------------------------------------------------ |
| `--score-window`      | 8       | Iterations of played decks the learning copy trains on. With 2 pods and 4 blocks, that is 512 decks, played by recent versions of the player |
| `--score-passes`      | 4       | Gradient steps over the window each iteration          |
| `--score-pull`        | 1.0     | The weight of the KL divergence from the learning copy's Bernoulli prediction to the human-trained one's, on the same decks |
| `--switch-iterations` | 10      | Iterations of test loss the switch averages            |
| `--switch-margin`     | 0.002   | How far the learning copy's mean test loss must fall below the human-trained score's, in nats a game. It is about two and a half times the noise of the averaged difference, by the estimate below |

To estimate that noise, take the defaults. An iteration has 2 pods of 8
seats, so 16 decks, and each deck plays 25 games in each of 4 blocks. That
gives 1,600 deck games, counting each game once for each of its two decks.
When the two scores' chances differ by about 0.05, one game's loss
difference varies by about 0.1 nats, so the mean over 10 iterations varies
by about 0.0008.

The test losses and the switch are stored in a `ScoreCheck`, defined in
`jace.loops.checked_score`, which `limited_iteration` takes and returns. A
run starts from `FIRST_CHECK`. A resumed run rebuilds the test losses and
the switch from its curve file, `<run id>.curve.json`, which `train_limited`
rewrites after every iteration. The played decks are not saved, so the
learning copy's window starts empty.

## The builder learns from games

The builder learns only in iterations whose search read the learning copy. A
build's advantage is its win rate minus the mean win rate of its pool's
builds that played, a shared baseline that POMO (Kwon et al. 2020, arXiv
2010.16011) also uses. Each step raises each build's log-probability in
proportion to its advantage. The loss also adds a bonus of `builder_entropy`
times the entropy of the builder's choices, so the builder's sampling stays
spread out. The schedule sets the builder's learning rate.

## Pretraining on human decks

`jace.loops.pretrain_decks` trains the builder by teacher forcing. The
builder learns to make the choices that would have written the human's
deck. `builds_from_decks` lays out each human deck as those choices. The
deck's spells and nonbasic lands go in a random order, cut into rounds of
`spells_per_round`, with a spell last, because sampling stops once the
spell slots are full. The land total and the basics split are the
human's.

The loss is the negative log-probability of those builds, each weighted by
the share of games its event won, with a floor of `jace.data.RESULT_FLOOR`.
Only human decks that the deck rules allow are used to train or test the
builder.

The deck score pretrains with `score_loss` on every human deck's event
record. `jace.loops.deck_measures` measures both models on held-out decks:

- `build_log_loss` is the builder's negative log-probability of the human
  builds, unweighted.
- `deck_overlap` is the share of the human deck's non-basic cards that the
  builder's likeliest deck for the same pool also contains.
- `land_total_match` is the share of pools whose likeliest deck has the
  human deck's land total.
- `score_fit` is the deck score's log loss on the games each deck won and
  lost, with the correlation of its predicted and actual win rates.

Both models' log losses are also taken on as many training decks, to show
when a model starts to fit its training decks better than decks it has not
seen.

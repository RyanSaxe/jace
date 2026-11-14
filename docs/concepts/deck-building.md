# Deck building

The builder samples whole decks from a pool, the deck score ranks them, and
the decks the loop plays teach both models. This page follows one pool from
its format's rules to the gradient steps its games give.

## The format sets the sizes

A `LimitedFormat` in `jace.loops.formats` holds every number that sizes one
limited format: the seats of a pod, the boosters each seat opens, the picks
from each, the booster spec, the deck size and the land range. Avatar's is
`TLA_LIMITED`: eight seats, three boosters of 14 picks, 40-card decks of 15
to 18 lands. Every entry point looks its format up by set code with
`limited_format`, and no model fixes any of these numbers.

`deck_rules(limited_format, cards)` turns the format into the `BuildRules`
of one card set, whose basics and nonbasic lands are that set's card ids. A
deck is exactly `size` cards and holds no more copies of a card than its
pool, except basics, which are unlimited. A pool with too few spells to
fill a deck within the land range plays every spell it has and lands in
the slots left.

## How the builder writes a deck

`DeckBuilder` reads the pool's non-basic cards as tokens, each with two
amounts: its copies in the pool and its copies already in the deck. It
writes a deck in three kinds of choice:

| Choice           | Made by                     | Odds                                          |
| ---------------- | --------------------------- | --------------------------------------------- |
| The land total   | The land head, once         | A softmax over the totals the rules allow     |
| Spells and nonbasic lands | Each card's own encoding, a round at a time | Plackett-Luce over the copies left |
| The basics       | The basics head, once       | A multinomial split of the land slots left    |

A round draws up to `spells_per_round` copies without replacement from one
call of the model, and the next round encodes the pool again with those
copies taken. So a sixth four-drop scores lower than the first: the model
has seen the other five. Nonbasic lands are drawn with the spells, up to
the land total, and the rounds stop when the spell slots are full. Every
choice is masked by the rules, so every deck is legal.

```python
builds, log_probs = sample_builds(
    builder, pool_counts.expand(256, -1), rules=rules, spells_per_round=6, temperature=1.0, generator=generator
)
```

`builds` holds every choice of every deck, and `log_probs` each deck's
log-probability: the land choice, the Plackett-Luce terms and the basics
split, under the sampling at `temperature`. `build_log_probs` gives the
same numbers with gradients for any recorded builds. It encodes every
round of every build in one call, because the deck at the start of each
round is known.

## The search

`build_decks` samples `--deck-samples` decks for a pool in one batch,
default 256, and scores each distinct deck once with the deck score. It
returns the best `--blocks` distinct decks, best first. In the limited
loop, each block plays one build of every pool: pool `p` plays its build
`(p + b) % blocks` in block `b` (`block_builds`). Every block then mixes
builds of every rank, so a pool's builds meet the same mix of opponents and
their win rates can be compared. Were block `b` to play every pool's build
`b`, each block's win rates would average one half whatever the builds'
ranks. The pool's best build is the
one the deck score ranked first, and its win rate is the drafter's reward.

## Two deck scores

The deck score trained on human decks predicts how humans' decks did in
humans' hands. The loop keeps that score frozen and trains a learning copy
of it from the first iteration. Before the learning copy trains on an
iteration's decks, both predict those decks' records, and their log loss
on them is that iteration's test. Every self-play game has one winner, so
an iteration's decks win half their games on average, while the human
decks the frozen score was fit to won more. Each score is tested with its
mean logit over the decks moved to zero, so the test compares how the two
rank decks, and learning that average alone does not earn the switch.

The search reads the human-trained score until the learning copy's mean
test loss over the last `--switch-iterations` iterations is below it by
`--switch-margin` nats a game. From then on it reads the learning copy, for
good. The metrics log both losses every iteration and the switch, as
`deck_score/human_test_loss`, `deck_score/learning_test_loss` and
`deck_score/reads_learning`.

| Flag                  | Default | Why                                                    |
| --------------------- | ------- | ------------------------------------------------------ |
| `--score-window`      | 8       | Iterations of played decks the learning copy trains on. With 2 pods and 4 blocks, that is 512 decks, played by recent versions of the player |
| `--score-passes`      | 4       | Gradient steps over the window each iteration          |
| `--score-pull`        | 1.0     | Weighs the KL divergence from the learning copy's Bernoulli prediction to the human-trained one's, on the same decks |
| `--switch-iterations` | 10      | Iterations of test loss the switch averages            |
| `--switch-margin`     | 0.002   | About two and a half times the noise of the averaged difference, by the estimate below |

The noise estimate: an iteration of 2 pods, 4 blocks and 25 games a block
gives 1,600 deck games. When the two scores' chances differ by about 0.05,
one game's loss difference varies by about 0.1 nats, so the mean over 10
iterations varies by about 0.0008.

Self-play decks win half their games on average, and the human decks won
56% of theirs. A score that predicts 56% where the truth is 50% loses about
0.007 nats a game to that bias alone. So a learning copy that only corrects
the human-trained score's bias on self-play decks may pass the margin,
though correcting a constant bias does not change how the search ranks a
pool's decks.

`jace.loops.checked_score` holds this state as a `ScoreCheck`, which
`limited_iteration` takes and returns. A run starts from `FIRST_CHECK`. A
resumed run rebuilds the losses and the switch from its curve, and its
learning copy's window starts empty.

## The builder learns from games

The builder learns only in iterations whose search read the learning
copy. Each build's reward is its win rate minus the mean win rate of its
pool's builds that played, the shared baseline of POMO (Kwon et al. 2020,
arXiv 2010.16011). The step raises each build's log-probability in
proportion to its advantage, with a bonus of `builder_entropy` times the
entropy of its choices, which keeps the sampling spread out. The schedule
sets the builder's rate.

## Pretraining on human decks

`jace.loops.pretrain_decks` teaches the builder the human deck by teacher
forcing. `builds_from_decks` puts each deck's spells and nonbasic lands in
a random order with a spell last, cut into rounds of `spells_per_round`,
with the human's land total and basics split. The loss is the negative
log-probability of those builds. `jace.loops.deck_measures` measures the
builder three ways on held-out decks. `build_log_loss` is that negative
log-probability, unweighted. `deck_overlap` is the share of the human
deck's non-basic cards that the builder's likeliest deck for the same pool
also holds. `land_total_match` is the share of pools whose likeliest deck
has the human deck's land total. The deck score pretrains on every human
deck's event record with `score_loss` and is measured by `score_fit`. Both
models' log losses are also taken on as many training decks, so a run
shows when a model starts to fit its training decks better than decks it
has not seen.

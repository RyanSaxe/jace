# Models

jace trains six models. Each reads cards through one shared embedding, and
each is a small transformer.

| Model        | Reads                                         | Returns                                        |
| ------------ | --------------------------------------------- | ---------------------------------------------- |
| Drafter      | Every pack it has seen and every pick so far  | A score for each card in the current pack      |
| Deck builder | A pool, and the deck written so far           | A score for each card, each land total and each basic |
| Deck score   | A deck                                        | One number: how likely the deck is to win      |
| Guess model  | The meta, the board, recent events and its earlier guesses | The opponent's deck and hand, as probabilities |
| Player       | The board and the guess                       | A score for each legal action                  |
| Win chance   | The board and the opponent's true hidden cards | The acting seat's chance to win               |

## The card embedding

```python
vocabulary = FIRST_PROGRAM_TOKEN + len(program_vocabulary())
encoder = CardEncoder(inputs, width=64, program_vocabulary=vocabulary, program_layers=1, heads=4)
vectors = encoder(card_ids)  # [..., 64]
table = frozen_table(encoder)  # the same vectors, with nothing to train
```

`inputs` is a `CardInputs`, which contains three tensors with one row per
card:

| Field     | Contains, for each card                                    |
| --------- | ---------------------------------------------------------- |
| `text`    | The vector a frozen language model gives its printed text  |
| `program` | Its engine program, as token ids                           |
| `printed` | Its printed facts, such as mana value, colours and rarity  |

The encoder maps the text vector and the printed facts to 64 numbers each.
It gives each token of the program a learned position, reads the tokens
with a one-layer transformer and averages the transformer's output over
them. A small network then joins the three vectors into one, and a layer
norm normalizes the result. A card's vector depends only on that card's
inputs, so the vectors do not change when other cards are added to the pool.

The drafter's pretraining trains the encoder. Deck pretraining and
self-play read `frozen_table(encoder)`, a `CardTable` that contains every
card's vector and has no parameters, so no later loss can change the
vectors the drafter learned. A model takes either one as `CardVectors`,
the interface of anything that gives the number of cards, the width and
the device, and turns card ids into vectors.

An engine program is the engine's own definition of a card, a list of
(kind, value) integer pairs. `program_vocabulary` numbers every pair in
the engine's catalog, so a pair's id does not depend on the pool. A pair's
token id is `FIRST_PROGRAM_TOKEN` plus its number, and the three ids below
it have meanings of their own:

| Token id              | Stands for                          |
| --------------------- | ----------------------------------- |
| `PADDING_TOKEN`, 0    | Padding after the end of a program  |
| `NO_PROGRAM_TOKEN`, 1 | A card the engine cannot play       |
| `UNKNOWN_TOKEN`, 2    | A pair the vocabulary lacks         |

## Scoring cards by embedding

No model has an output layer with one slot per card. The drafter and the
guess model each produce a query vector and score a card by the dot
product of the query with a projection of that card's embedding. The deck
builder scores a card of the pool from that card's own encoding, the
transformer's output at the card's token. It scores a basic land by the
dot product of a query with a projection of the basic's embedding.

No weight therefore depends on the card pool. Tests build each model on a
small pool and then grow the pool. Every model still runs, and every model
but the drafter gives the old cards the same scores.

## The six models

**Drafter**, `DrafterModel.forward(seen, picks, pack=pack)`. It reads one
token for each earlier pick, made from the card taken, the cards of the
pack it came from and the pick's position, plus one token for the current
pack. Reading the whole draft lets it see signals: which cards were still
in a pack when it came around.

**Deck builder**, `DeckBuilder.forward(pools, taken, rules=rules)`. It
reads one token for each card of the pool except the basics, with the
card's copies in the pool and its copies already in the deck. A land head
scores each land total from a summary of the pool and the deck so far,
and from the total's share of the deck. It reads that share through sines
and cosines, so that it can prefer a total in the middle of the legal
range. A basics head scores each basic. `sample_builds` writes many decks
in one batch, a few cards a round. Masks, not learning, enforce the rules
the builder cannot break: the deck size, the pool's copies and the land
range.

`build_decks` samples many decks for a pool, scores each distinct one once
with the deck score, and returns the best first.
[Deck building](deck-building.md) follows the whole process.

**Deck score**, `DeckScore.forward(decks)`. It reads one token per
distinct card, with its count, and returns the logit of the deck's chance
to win a game. It learns from which decks won, so its score is a chance to
win in the hands that played those decks. The copy trained on human decks
predicts wins in human hands. The copy that learns from self-play predicts
wins in the hands of the current player.

**Guess model**, `GuessModel.guess(boards, earlier)`. It returns a `Guess`
of two vectors over cards. `deck` is each card's share of the opponent's
unseen cards, its hand and library. `hand` is the chance that the
opponent's hand contains at least one copy of each card.

The guess model reads only what a player can know:

- the meta, described below
- the board, including the acting seat's own deck
- the sizes of the opponent's hand and library
- every card the opponent has shown or revealed
- the recent events
- its own last `--history` guesses in the same game, 4 by default

It starts from the meta guess, which it works out from the meta and the
cards seen, without its network. The network learns only corrections to
the meta guess.

```python
meta = fit_meta(human_decks, fallback=day_zero_meta(booster_copies, deck_size=40))
guess_model.set_meta(meta)  # what decks of the format look like
start = guess_model.meta_guess(boards)  # the guess with no network
guess = guess_model.guess(boards, earlier)  # the meta guess plus the network's corrections
meta = blend_meta(meta, played_decks, decay=0.9)  # after a round, with older decks weighted down
```

The meta is a `Meta`, a record of card co-occurrence over the format's
decks. For each card, it contains the mean copies per deck. For each ordered
pair of cards i and j, it contains the mean copies of j in the decks that
contain at least one i. It also contains which cards the format's boosters
contain, the number of decks it was fitted or blended from, and how many of
those decks contain each card. In a blended meta an older deck counts for
less than one, so these numbers are effective numbers. Nothing in the meta
names an archetype or a colour. What decks have in common shows only in
which cards they contain together.

| Meta | Fitted from | Used by |
| ---- | ----------- | ------- |
| Day 0 | How often the boosters contain each card, with every card independent of the others. A card that makes up a share s of a booster's cards has deck size × s copies per deck | A format with no decklists, and any card that none of a meta's decks contain |
| Human | The 17Lands decks of the training split, the drafts the drafter trains on | The start of every self-play run |
| Blended | The meta before, with the weight of each of its decks multiplied by `--meta-decay`, plus an iteration's played decks | Each self-play iteration after the first, and scoring, through the meta saved in a checkpoint |

The self-play loops blend an iteration's decks into the meta only after
its games are over, so no game is guessed with its own decks. Scoring
reads the meta saved with the guess model's checkpoint, never the decks
dealt in the games it scores.

For the meta guess, a card is known to be in the opponent's deck while a
copy of it is on the opponent's battlefield, in its graveyard, in exile or
on the stack. A card is also known while the engine reports a copy of it
revealed in the opponent's hand. Each known card i gives every card j a
lift, the ratio of j's mean copies in the decks that contain i to j's mean
copies in all decks. `meta_guess` works out, card by card:

| Step | The meta guess's value for a card |
| ---- | --------------------------------- |
| Expected copies | Its mean copies, times the exponential of the known cards' log lifts for it, summed and divided by the square root of their number. A known card has at least its mean copies in the decks that contain it |
| Unseen copies | Its expected copies less the copies known, at least a thousandth of a copy for a legal card and none for any other. They are then scaled so that all cards' unseen copies add up to the opponent's hidden cards that are not known |
| Deck | Its copies known to be in hand plus its unseen copies, as a share of the opponent's hidden cards |
| Hand | 1 when a copy is known to be in hand, and otherwise 1 - exp(-p u) |

Here p is the share of the opponent's unknown hidden cards that are in its
hand, and u is the card's unseen copies.

The hand chance takes the unseen copies as a Poisson number with mean u,
because u is a mean and not a whole count. The chance for whole copies,
1 - (1 - p)^u, would put every legal card in hand once the opponent's
library is empty, since p is then 1.

Dividing the sum of the log lifts by the square root of their number is a
middle way. A plain sum would count the known cards as independent
evidence, which cards of one deck are not. A plain mean would give twenty
known cards no more weight than two. On held-out human decks with some of
their cards shown, the meta guess predicted the hidden cards better with
the square root than with either.

The opponent's deck may always contain more copies of a legal card than
expected, so `meta_guess` gives every legal card at least a thousandth of
an unseen copy. While some of the opponent's hidden cards are not known,
every legal card therefore has a share above zero, and only a card outside
the format has none.

The network reads each card on the board together with that card's row of
the meta, given as the mean card vector of the decks that contain the
card. It adds a correction to the log of each card's share and to the
logit of each hand chance. The corrections start at zero, so an untrained
model's guess is the meta guess. A share of 0, and a chance of 0 or 1,
have an infinite log or logit, so no correction changes them.

`guess_loss` is the deck guess's cross-entropy plus the hand guess's binary
cross-entropy, each summed over cards, so a decision's hand guess counts
as much as its deck guess.

The meta is stored in buffers of the model's state dict. So the worker
processes that receive the weights guess from the same meta, and a
checkpoint contains the meta along with the weights.

**Player**, `PlayerModel.policy(boards, guess)`. It reads the players,
cards and permanents on the board as tokens. To a permanent's token, it
adds a projection of the vector of the object the permanent is attached
to, and of the attacker it blocks. The player also adds a token for each
of these:

- each card of its own deck that has copies it has not yet seen
- the `--guess-cards` cards most likely to be in the opponent's hand, 8 by
  default
- the `--deck-guess-cards` cards with the largest shares of the opponent's
  unseen cards, 16 by default

Illegal actions get no probability. The player has no win chance of its
own.

**Win chance**, `WinChanceModel.win_chance(boards, truth)`. It reads the
board and the acting seat's own deck as the player does. In place of the
guess, it reads one token for each card the opponent truly has unseen,
in hand or in library, with that card's share of those cards and whether
it is in hand. It returns the acting seat's chance to win. In self-play
the engine reports the hidden cards, so this estimate can read them where
the player has to guess. That makes the win chance an asymmetric critic,
one that reads more than the policy it evaluates (Pinto et al.,
[arXiv 1710.06542](https://arxiv.org/abs/1710.06542)). Only training calls
it, so the hidden cards never reach the player.

## The engine's boards

The guess model, the player and the win chance read manabot's encoded boards
directly. The code that reads those arrays is in
[`jace/play/tokens.py`](../../src/jace/play/tokens.py). There,
`board_tokens` turns boards into tokens, and its comment lists what it
assumes about each array. `opponent_cards` counts the opponent's cards for
the guess model's meta guess. The engine pads each kind of row to a fixed
capacity, such as 80 cards per seat and 512 actions, and a real board fills
few of those rows. So `board_tokens` keeps each kind only up to its last
real row in the batch.

The boards of a batch still have different numbers of real tokens. Each
model's transformer runs through `encode_real` in
[`jace/modules.py`](../../src/jace/modules.py), which packs the real tokens
of every board into one matrix. The projections, feed-forward blocks and
norms then work on real tokens alone. Only attention has to keep each
board's tokens apart. On an Apple GPU it runs as Metal kernels over the
packed tokens, in
[`jace/metal_attention.py`](../../src/jace/metal_attention.py). On other
devices it runs as `scaled_dot_product_attention` on each board's tokens,
padded to the longest board's. `encode_real` runs the layers of a standard
`nn.TransformerEncoder`, so the models have that encoder's parameters, and
checkpoints are the same as they would be without the packing.

Most columns of a board are numbers, such as a creature's power. Some name
something instead: a card definition, another object on the board, or a
choice such as an ability's number. `board_layout` reads which columns
those are from the engine's own description of its arrays. A column that
names a card is read as that card's embedding, one that names an object
as that object's vector, and one that names a choice as a row of a small
learned table. The rest, such as an event's raw engine ids, are not read.

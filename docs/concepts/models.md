# Models

jace trains six models. Each reads cards through one shared embedding, and
each is a small transformer.

| Model        | Reads                                         | Returns                                        |
| ------------ | --------------------------------------------- | ---------------------------------------------- |
| Drafter      | Every pack it has seen and every pick so far  | A score for each card in the current pack      |
| Deck builder | A pool, and the deck written so far           | A score for each card, each land total and each basic |
| Deck score   | A deck                                        | One number: how likely the deck is to win      |
| Guess model  | The board, recent events, its earlier guesses | The opponent's deck and hand, as probabilities |
| Player       | The board and the guess                       | A score for each legal action                  |
| Win chance   | The board and the opponent's true hidden cards | The acting seat's chance to win               |

## The card embedding

```python
vocabulary = FIRST_PROGRAM_TOKEN + len(program_vocabulary())
encoder = CardEncoder(inputs, width=64, program_vocabulary=vocabulary, program_layers=1, heads=4)
vectors = encoder(card_ids)  # [..., 64]
table = frozen_table(encoder)  # the same vectors, with nothing to train
```

`inputs` is a `CardInputs`, which holds three things for each card: the
vector a frozen language model gives its printed text, its engine program
as token ids, and its printed facts. The encoder maps the text vector and
the printed facts to 64 numbers each. It reads the program's tokens, each
with a learned position, with a one-layer transformer and averages them. A
small network joins the three vectors and a layer norm ends it. A card's
vector depends only on that card's inputs, so the vectors do not change
when other cards join the pool.

The drafter's pretraining trains the encoder. Deck pretraining and
self-play read `frozen_table(encoder)`, a `CardTable` that holds every
card's vector and has no parameters, so no later loss can move what the
drafter learned. A model takes either one as `CardVectors`: the number of
cards, the width, the device, and a call that turns card ids into vectors.

The program's token ids start at `FIRST_PROGRAM_TOKEN`. The three ids
below it pad a program, stand for a card the engine cannot play, and mark a
(kind, value) pair the vocabulary lacks. `program_vocabulary` numbers every
pair in the engine's catalog, so a pair's id does not depend on the pool.

## Scoring cards by embedding

No model has an output layer with one slot per card. The drafter and the
guess model each produce a query vector and score a card by the dot product
of the query with that card's embedding. The deck builder scores a card of
the pool from that card's own encoding, and a basic land by a query's dot
product with the basic's embedding.

This keeps every weight independent of the card pool. A test builds each
model on a small pool, grows the pool, and checks that the model still runs
and gives old cards the same scores.

## The six models

**Drafter**, `DrafterModel.forward(seen, picks, pack=pack)`. One token for
each earlier pick, made from the card taken and the pack it came from, plus
a token for the current pack. Reading the whole draft lets it see signals:
which cards were still in a pack when it came around.

**Deck builder**, `DeckBuilder.forward(pools, taken, rules=rules)`. One
token for each non-basic card of the pool, with its copies in the pool and
its copies already in the deck. A land head scores each land total and a
basics head scores each basic. `sample_builds` writes many decks from it in
one batch, a few cards a round, and the rules it cannot break are enforced
by masks, not learned: deck size, copies in the pool, and a land count
within a range.

`build_decks` samples many decks for a pool, scores each distinct one once
with the deck score, and returns the best first.
[Deck building](deck-building.md) follows the whole process.

**Deck score**, `DeckScore.forward(decks)`. One token per distinct card,
with its count. It is trained on which decks won, so its score means "in the
hands of the current player".

**Guess model**, `GuessModel.guess(boards, earlier)`. It returns two vectors
over cards: each card's share of the opponent's unseen cards, and the chance
the opponent holds at least one copy. It reads only what a player can know:
the meta, the board with its own deck, the opponent's hand and library
sizes, every card the opponent has shown or revealed, the recent events and
its own earlier guesses. It starts from the meta guess, and its network
learns only corrections to it.

```python
meta = fit_meta(human_decks, fallback=day_zero_meta(booster_copies, deck_size=40))
guess_model.set_meta(meta)  # what decks of the format look like
start = guess_model.meta_guess(boards)  # the guess with no network
guess = guess_model.guess(boards, earlier)  # the meta guess plus the network's corrections
meta = blend_meta(meta, played_decks, decay=0.9)  # after a round, with older decks weighted down
```

The meta is a matrix of card co-occurrence. For each card it holds the mean
copies per deck, and for each ordered pair of cards i and j the mean copies
of j in the decks that hold at least one i. It also holds the effective
number of decks behind it and which cards the format's boosters hold.
Nothing in it names an archetype or a colour: whatever decks share shows
only in which cards they hold together.

| Meta | Fitted from | Used by |
| ---- | ----------- | ------- |
| Day 0 | How often the boosters hold each card, with cards independent: a card whose share of a booster is s has deck size × s copies per deck | A format with no decklists, and any card no deck behind a meta holds |
| Human | The 17Lands decks of the training split, the drafts the drafter trains on | The start of every self-play run |
| Blended | The meta before, each deck weighted by `--meta-decay`, plus an iteration's played decks | Each self-play iteration after the first, and the checkpoint scoring reads |

A self-play iteration blends in its decks only after its games are over,
so no game is guessed with its own decks. Scoring reads the meta saved with
the guess model's checkpoint and never the decks the harness deals.

For the meta guess, a card is known to be in the opponent's deck once it
has shown a copy on the battlefield, in the graveyard, in exile or on the
stack, or the engine reports a copy revealed in its hand. Each known card i
gives every card j a lift, the ratio of j's copies in decks that hold i to
j's mean copies. The meta guess works out, card by card:

| Step | The meta guess's value for a card |
| ---- | --------------------------------- |
| Expected copies | Its mean copies, times the exponential of the known cards' log lifts for it summed and divided by the square root of their number; a known card has at least its mean copies in the decks that hold it |
| Unseen copies | Its expected copies less the copies known, at least a thousandth of a copy for a legal card and none for another, scaled so that all cards' unseen copies fill the opponent's hidden cards that are not known |
| Deck | Its copies known to be in hand plus its unseen copies, as a share of the hidden cards |
| Hand | 1 when a copy is known to be in hand, and otherwise 1 - exp(-p u) |

Here p is the share of the opponent's unknown hidden cards that are in its
hand, and u is the card's unseen copies. The unseen copies are a mean, so
the hand chance takes them as a Poisson number with that mean: the chance
of whole copies, 1 - (1 - p)^u, would put every legal card in hand once the
opponent's library is empty. Summing the log lifts would count
the known cards as independent evidence, which cards of one deck are not,
and averaging them would say no more after twenty cards than after two.
Dividing the sum by the square root of their number sits between. On
held-out human decks with some of their cards shown, it predicted the
hidden cards better than both.

The meta cannot know that the opponent's deck holds no more copies of a
card, so every legal card keeps some share and some chance, and only a card
outside the format gets none. The network reads each card on the board
with that card's row of the meta, as the mean card vector of the decks
that hold it. It adds a correction to the log of each card's share and to
the logit of each hand chance. The corrections start at zero, so an
untrained model's guess is the meta guess, and a chance of 0 or 1 stays
where it is. `guess_loss` sums both guesses over cards, so a decision's
hand guess weighs as much as its deck guess. The meta is held in buffers of
the model's state dict, so the worker processes that receive the weights
guess from the same meta, and the checkpoint holds it.

**Player**, `PlayerModel.policy(boards, guess)`. It reads the players, cards
and permanents on the board as tokens. A permanent's token also holds the
object it is attached to and the attacker it blocks. The player adds one
token for each card of its own deck that has copies it has not yet seen,
and tokens for the likeliest cards from the guess. Illegal actions get no
probability. It has no win chance of its own.

**Win chance**, `WinChanceModel.win_chance(boards, truth)`. It reads the
board and the acting seat's own deck as the player does, and in place of
the guess one token for each card the opponent truly holds unseen, with
the card's share of those cards and whether it is in hand. It returns the
acting seat's chance to win. Self-play knows the hidden cards, so the
estimate need not guess them, as in an asymmetric critic (Pinto et al.,
[arXiv 1710.06542](https://arxiv.org/abs/1710.06542)). Only training calls
it, so the hidden cards never reach the player.

## The engine's boards

The guess model, the player and the win chance read manabot's encoded
boards directly. That knowledge sits in
[`jace/play/tokens.py`](../../src/jace/play/tokens.py): in `board_tokens`,
whose comment lists every array it assumes, and in `opponent_cards`, which
counts the opponent's cards for the guess model's meta guess. The engine pads
each kind of row to a fixed capacity, such as 80 cards and 512 actions,
and a real board fills few of them, so `board_tokens` keeps each kind only
up to its last real row in the batch.

Most columns of a board are numbers, such as a creature's power. Some name
something instead: a card definition, another object on the board, or a
choice such as an ability's number. `board_layout` reads which columns those
are from the engine's own description of its arrays. A column that names a
card becomes that card's embedding, one that names an object becomes that
object's vector, and a choice gets a small learned table. The rest, such as
an event's raw engine ids, are not read.

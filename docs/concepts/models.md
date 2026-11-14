# Models

jace trains five models. Each reads cards through one shared embedding, and
each is a small transformer.

| Model        | Reads                                         | Returns                                        |
| ------------ | --------------------------------------------- | ---------------------------------------------- |
| Drafter      | Every pack it has seen and every pick so far  | A score for each card in the current pack      |
| Deck builder | A pool, and the deck written so far           | A score for each card, each land total and each basic |
| Deck score   | A deck                                        | One number: how likely the deck is to win      |
| Guess model  | The board, recent events, its earlier guesses | The opponent's deck and hand, as probabilities |
| Player       | The board and the guess                       | A score for each legal action, and a win chance|

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

## The five models

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
the opponent holds at least one copy.

**Player**, `PlayerModel.policy(boards, guess)`. It reads the players, cards
and permanents on the board as tokens. A permanent's token also holds the
object it is attached to and the attacker it blocks. The player adds one
token for each card of its own deck that has copies it has not yet seen,
and tokens for the likeliest cards from the guess. Illegal actions get no
probability.

## The engine's boards

The guess model and the player read manabot's encoded boards directly. That
knowledge sits in one function, `board_tokens` in
[`jace/play/tokens.py`](../../src/jace/play/tokens.py), whose comment lists
every array it assumes.

Most columns of a board are numbers, such as a creature's power. Some name
something instead: a card definition, another object on the board, or a
choice such as an ability's number. `board_layout` reads which columns those
are from the engine's own description of its arrays. A column that names a
card becomes that card's embedding, one that names an object becomes that
object's vector, and a choice gets a small learned table. The rest, such as
an event's raw engine ids, are not read.

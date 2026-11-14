# Models

jace trains six models. Each reads cards through one shared embedding, and
each is a small transformer.

| Model        | Reads                                         | Returns                                        |
| ------------ | --------------------------------------------- | ---------------------------------------------- |
| Drafter      | Every pack it has seen and every pick so far  | A score for each card in the current pack      |
| Deck builder | A pool, and the deck written so far           | A score for each card it could add next        |
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

No model has an output layer with one slot per card. The drafter, the deck
builder and the guess model each produce a query vector and score a card by
the dot product of the query with that card's embedding.

This keeps every weight independent of the card pool. A test builds each
model on a small pool, grows the pool, and checks that the model still runs
and gives old cards the same scores.

## The six models

**Drafter**, `DrafterModel.forward(seen, picks, pack=pack)`. One token for
each earlier pick, made from the card taken and the pack it came from, plus
a token for the current pack. Reading the whole draft lets it see signals:
which cards were still in a pack when it came around.

**Deck builder**, `DeckBuilderModel.forward(available, partial)`. It writes a
deck one card at a time. Rules it cannot break are enforced by masks, not
learned: deck size, copies in the pool, and a land count within a range.

`build_decks` samples many decks from it, improves each by swaps, ranks them
with the deck score, and returns the candidates best first.

**Deck score**, `DeckScore.forward(decks)`. One token per distinct card,
with its count. It is trained on which decks won, so its score means "in the
hands of the current player".

**Guess model**, `GuessModel.guess(boards, earlier)`. It returns two vectors
over cards: each card's share of the opponent's unseen cards, and the chance
the opponent holds at least one copy. It starts from a table of what the
acting seat can know, and its network learns only corrections to that
table.

```python
guess_model.set_field(decks)  # the decks the opponent's deck is drawn from
table = guess_model.table(boards)  # the guess with no network
guess = guess_model.guess(boards, earlier)  # the table plus the network's corrections
```

The field is the set of decks the opponent may hold. In self-play it is the
decks of the games being played, and in scoring it is the decks the harness
deals. For each card, the table counts the copies the opponent has shown on
the battlefield, in the graveyard, in exile or on the stack, and the copies
the engine reports were revealed from its hand. Only the field's decks that
hold at least that many copies could be the opponent's, card by card, and
the table averages over those:

| Guess | The table's value for a card                                         |
| ----- | -------------------------------------------------------------------- |
| Deck  | Its copies known to be in hand, plus the mean of the copies those decks hold beyond the ones counted, as a share of all cards |
| Hand  | 1 when a copy is known to be in hand, and otherwise the mean over those decks of 1 - (1 - p)^u |

Here p is the share of the opponent's unknown hidden cards that are in its
hand, and u is the number of copies a deck holds beyond those counted.
Taking the field's mean copies less the copies seen would be simpler, but it
gives a card no chance once the opponent shows more copies than the mean,
which basic lands do within a few turns.

The network adds a correction to the log of each card's share and to the
logit of each hand chance. The corrections start at zero, so an untrained
model's guess is the table. A card the table gives no share keeps none,
and a chance of 0 or 1 stays where it is. `guess_loss` sums both guesses
over cards, so a decision's hand guess weighs as much as its deck guess.
The field is a buffer in the model's state dict, so the worker processes
that receive the weights guess from the same field.

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
counts the opponent's cards for the guess model's table.

Most columns of a board are numbers, such as a creature's power. Some name
something instead: a card definition, another object on the board, or a
choice such as an ability's number. `board_layout` reads which columns those
are from the engine's own description of its arrays. A column that names a
card becomes that card's embedding, one that names an object becomes that
object's vector, and a choice gets a small learned table. The rest, such as
an event's raw engine ids, are not read.

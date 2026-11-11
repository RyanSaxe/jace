# Models

jace trains five models. Each reads cards through one shared embedding, and
each is a small transformer.

| Model        | Reads                                         | Returns                                        |
| ------------ | --------------------------------------------- | ---------------------------------------------- |
| Drafter      | Every pack it has seen and every pick so far  | A score for each card in the current pack      |
| Deck builder | A pool, and the deck written so far           | A score for each card it could add next        |
| Deck score   | A deck                                        | One number: how likely the deck is to win      |
| Guess model  | The board, recent events, its earlier guesses | The opponent's deck and hand, as probabilities |
| Player       | The board and the guess                       | A score for each legal action, and a win chance|

## The card embedding

```python
embedding = CardEmbedding(starting, hidden=64, offset_bound=None)
vectors = embedding(card_ids)  # [..., width]
```

`starting` holds one fixed vector per card. The embedding of a card is its
starting vector plus the output of a small head:

```text
embedding = starting + head(starting)
```

The head is shared by every card and starts at zero, so before training the
embedding equals the starting vector. Training changes the head, never the
starting vectors.

Setting `offset_bound` adds a per-card correction whose length is capped at
that bound. It lets a card with a lot of data move a little away from what
the head gives it. Both versions exist so they can be compared.

## Scoring cards by embedding

No model has an output layer with one slot per card. The drafter, the deck
builder and the guess model each produce a query vector and score a card by
the dot product of the query with that card's embedding.

This keeps every weight independent of the card pool. A test builds each
model on a small pool, grows the pool, and checks that the model still runs
and gives old cards the same scores.

## The five models

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
the opponent holds at least one copy.

**Player**, `PlayerModel.policy(boards, guess)`. It reads the players, cards
and permanents on the board as tokens, plus the likeliest cards from the
guess. Illegal actions get no probability.

## The engine's boards

The guess model and the player read manabot's encoded boards directly. That
knowledge sits in one function, `board_tokens` in
[`jace/play/tokens.py`](../../src/jace/play/tokens.py), whose comment lists
every array it assumes.

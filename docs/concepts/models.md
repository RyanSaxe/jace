# Models

This page explains the card embedding that every model shares, and then the
five models: the drafter, the deck score, the deck builder, the guess model
and the player. For each one it gives what the model takes, what it returns
and the design point that matters most.

Every model here has been built and tested at small sizes. None has been
trained beyond single update steps.

## The card embedding

A card embedding is one vector per card. `CardEmbedding` in `jace.cards` is
the only place a model learns about a card, and all five models read the same
instance.

```python
CardEmbedding(starting: Tensor, *, hidden: int, offset_bound: float | None)
```

`starting` is a float tensor `[cards, width]` with one fixed starting vector
per card, in `CardPool` order. Calling the embedding with int64 card ids of
any shape returns their vectors, with one more axis of size `width`.

The embedding of a card is `e = s + g(s) + o`:

| Part   | What it is                                                                                           | Trained |
| ------ | ---------------------------------------------------------------------------------------------------- | ------- |
| `s`    | The card's starting vector.                                                                          | No      |
| `g(s)` | A small head applied to the starting vector. One head is shared by every card.                       | Yes     |
| `o`    | An offset per card, present only when `offset_bound` is set. Its length is clipped to that bound.    | Yes     |

The head's last layer starts at zero, so the embedding equals the starting
vector before any training. The offset also starts at zero. Its bound lets it
refine a known card without replacing what the head learned.

`pull()` returns the mean squared length of the head's change over all cards.
Added to a loss, it is a penalty that keeps the embedding near its start. The
limited loop does not add it.

### How a new card gets an embedding

The head is a function of the starting vector, and it has no weight per card.
A card from a new set therefore gets a usable embedding from its starting
vector alone: you build a `CardEmbedding` from the longer table of starting
vectors and copy the head's weights into it. The tests grow a pool this way.
No helper does it for you yet.

The design calls for a starting vector built from the card's rules text and
its engine program. That step is not built yet, and the tests use random
starting vectors.

### No weight is sized by the card pool

Each model scores a card by a dot product between a query vector it computes
and a projection of that card's embedding. No model has a parameter with one
slot per card. The same weights therefore score a larger pool once the
embedding covers it, and the tests check that old cards keep their scores
when the pool grows.

### The embedding has its own optimizer

Each model holds the embedding as a submodule, so `model.parameters()`
includes the embedding's parameters. Leave them out of the model's optimizer
and give the embedding its own. `apply` in `jace.train` steps every optimizer
it is given, so the embedding moves together with whichever model is being
updated.

## The drafter

`DrafterModel` in `jace.draft` scores the cards of the pack in front of a
seat, from the packs the seat has seen and the cards it took.

```python
DrafterModel(embedding: CardEmbedding, *, width: int, layers: int, heads: int, picks: int)

def forward(self, seen: Tensor, picks: Tensor, *, pack: Tensor) -> Tensor
```

| Argument | Shape and meaning                                                                    |
| -------- | ------------------------------------------------------------------------------------ |
| `seen`   | `[b, t, cards]`, 0 or 1: the earlier packs. A row of zeros marks a padded step.      |
| `picks`  | int64 `[b, t]`: the card taken from each earlier pack.                               |
| `pack`   | `[b, cards]`, 0 or 1: the current pack.                                              |

It returns logits `[b, cards]`, with cards outside the pack at `-inf`.

Each earlier pick is one token made from the picked card, the mean of the
cards in the pack it came from, and a learned position. The current pack is
one more token, and the transformer's output at that token is the query. The
constructor's `picks` is the largest number of picks in a draft. It sizes the
positions, so a longer draft needs a model built for it.

A seat sees its own packs and picks and never another seat's picks. It can
infer them only from the cards missing when a pack comes back around.

Code that runs drafts does not call the model. It calls a `Drafter`, a
function from seat views to one card per view:

```python
type Drafter = Callable[[Sequence[SeatView]], Sequence[CardId]]

def model_drafter(model: DrafterModel, *, generator: torch.Generator | None) -> Drafter
```

`model_drafter` picks for any number of seats in one forward pass. With a
generator, it samples each pick from the softmax of the logits. With `None`,
it takes the card with the highest logit. It runs without gradients, and
training recomputes the logits from the recorded views with `view_tensors`.

## The deck score

`DeckScore` in `jace.decks` rates a deck by its chance of winning a game.

```python
DeckScore(embedding: CardEmbedding, *, width: int, layers: int, heads: int)

def forward(self, decks: Tensor) -> Tensor
```

`decks` is `[b, cards]`, the copies of each card in each deck. The model
returns `[b]`, the logit of winning a game.

The model reads a deck as a set of distinct cards, each with a copy count.
The order of the cards cannot matter, and a deck of any size fits. Counts
enter through `log1p`, so 17 copies of a basic land stay in the range the
model saw for small counts.

## The deck builder

The deck builder turns a pool into ranked candidate decks. A pool is a plain
tuple of card ids, so the same code builds a limited deck from a draft and a
constructed deck from a collection.

```python
def build_decks(
    pool: Pool,
    *,
    cards: int,
    propose: Proposer,
    score: Scorer,
    rules: BuildRules,
    proposals: int,
    rounds: int,
) -> tuple[Candidate, ...]
```

`build_decks` works in three steps:

1. `propose` writes `proposals` whole decks for the pool.
2. `improve` climbs from each deck by the best one-card swap, for at most
   `rounds` rounds, against `score`.
3. Proposals that ended at the same deck are returned once. Each `Candidate`
   holds a deck and its score, best first.

A search that starts from one deck and swaps one card at a time cannot leave
the colours it started in. The builder therefore starts from many whole
decks, written with no colour rule.

`build_decks` does not choose which candidate to play. The caller does,
because exploring is the training loop's job.

### Deck rules

`BuildRules` says what makes a deck legal: its exact `size`, the `basics`,
which a deck may hold in any number, the `nonbasic_lands`, and the range
`[min_lands, max_lands]` for the total of both. A deck holds no more copies
of any other card than the pool does.

Deck shape is rules, and no model learns it. The proposers and the swap
search only ever make decks that the rules allow. A pool that allows no legal
deck raises `NoLegalDeckError`.

### Proposers

A `Proposer` takes the pool's copy counts and a number of decks, and returns
that many legal decks as rows of copy counts.

| Proposer          | How it writes a deck                                                                         |
| ----------------- | -------------------------------------------------------------------------------------------- |
| `model_proposer`  | One card at a time, sampling each card from `DeckBuilderModel`.                              |
| `random_proposer` | At random with no model: a random land count, random spells and a random mix of lands.       |

```python
DeckBuilderModel(embedding: CardEmbedding, *, width: int, layers: int, heads: int)

def forward(self, available: Tensor, partial: Tensor) -> Tensor
```

`available` is `[b, cards]`, the copies that can still be added, and
`partial` is `[b, cards]`, the copies already in the deck. The model returns
logits `[b, cards]` for the next card, with `-inf` where no copy is
available. It knows nothing about deck rules. The proposer masks the moves
that would break them.

The model learns from the search. `writing_steps` lays finished decks out as
the steps that would have written them, and a cross-entropy loss on those
steps teaches the model to write decks like the ones the search preferred.

The swap search is slow today. [Status](../status.md#known-limits) gives the
measurement.

## The guess model

`GuessModel` in `jace.play` predicts the opponent's hidden cards from one
seat's view. The guess is the player's only view of those cards.

```python
GuessModel(
    embedding: CardEmbedding,
    *,
    engine_to_card: Tensor,
    sample: Boards,
    width: int,
    layers: int,
    heads: int,
)

def guess(self, boards: Boards, earlier: Earlier) -> Guess
```

| Argument         | Meaning                                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `engine_to_card` | int64, the `CardId` of each engine card definition, or -1 for one the pool lacks. `engine_card_ids` builds it. |
| `sample`         | Any batch of boards from the engine that will be played. Only its feature widths are used.                    |
| `boards`         | The engine's named arrays as tensors, one row per pending decision.                                           |
| `earlier`        | The same seat's earlier guesses in this game, newest first, with a mask. It may hold none.                    |

A `Guess` has two tensors, each `[b, cards]`:

| Field  | Meaning                                                             |
| ------ | ------------------------------------------------------------------- |
| `deck` | Each card's share of the opponent's unseen cards. A row sums to 1.  |
| `hand` | The chance the opponent holds at least one copy of each card.       |

The model reads the board, the recent events and its own earlier guesses in
the same game. A guess can therefore build on the last one, and the model
carries no state between calls. It is trained on the true hidden cards that
the engine reports through `hidden_cards`.

## The player

`PlayerModel` in `jace.play` scores every legal action and estimates the
chance of winning.

```python
PlayerModel(
    embedding: CardEmbedding,
    *,
    engine_to_card: Tensor,
    sample: Boards,
    width: int,
    layers: int,
    heads: int,
    guess_cards: int,
)

def policy(self, boards: Boards, guess: Guess) -> tuple[Tensor, Tensor]
```

`policy` returns action logits `[b, max_actions]`, with illegal actions at
`-inf`, and the acting seat's chance to win, `[b]`.

The player reads the board and the guess of the opponent's hand. The guess
enters as `guess_cards` tokens, one for each of the cards the opponent is
likeliest to hold.

One head scores every kind of decision. It scores an action from four
vectors: the action's own features, the two board objects the action names,
and a summary of the whole board. Nothing depends on how many actions a
decision offers.

`random_policy(engine_to_card)` has the same call shape and no weights. It
gives every legal action the same odds, which makes it the opponent for a
first evaluation.

## Policy and Guesser

Code that plays games never names a model class. `collect`, the update
functions and evaluation take two plain functions over batched tensors:

```python
type Policy = Callable[[Boards, Guess], tuple[Tensor, Tensor]]
type Guesser = Callable[[Boards, Earlier], Guess]
```

`PlayerModel.policy` and `GuessModel.guess` satisfy them, and so does a
three-line function. The toy loop passes a small network wrapped in a
function, and the same `collect` plays its games.

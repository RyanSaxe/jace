# The approach

jace learns limited Magic by self-play. It drafts, builds decks and plays
games against itself, and the results of those games are its only reward.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="../../assets/approach-dark.svg">
  <img alt="How jace trains, as a loop of four steps. Draft: the drafter picks a card from each pack. Build: the builder proposes decks and the deck score ranks them. Play: the player acts, with a guess of the opponent's cards. Learn: all six models train on the game results, and the next draft uses the improved models. Below the loop, one card embedding: every model reads cards through it, so a new set needs no new weights." src="../../assets/approach-light.svg">
</picture>

## Why one loop

A pick is good only if it leads to a deck that wins, and a deck wins only in
the hands of a player who can pilot it. So the three skills cannot be judged
apart.

jace plays whole rounds of draft, build and play, and uses the game
results to improve every model at once.

## What each model learns from

| Model        | Signal                                                        | Kind of learning |
| ------------ | ------------------------------------------------------------- | ---------------- |
| Player       | Whether it won the game                                       | Reinforcement    |
| Drafter      | The win rate of the deck built from its picks                 | Reinforcement    |
| Deck score   | Which decks won and lost                                      | Supervised       |
| Deck builder | Each build's win rate minus the mean win rate of its pool's builds | Reinforcement |
| Guess model  | The opponent's true deck and hand, which the engine reports in self-play | Supervised |
| Win chance   | Whether the acting seat won, which it predicts from the board and the opponent's true cards | Supervised |

The player, the drafter and the builder learn from reward, the results of
the choices they made. The deck score and the guess model have exact
answers to learn from, which makes them cheaper to train. The win chance
learns to predict each game's result, and each of the player's actions is
credited by how much it raised that prediction.

The loop has two deck scores. One was trained on human decks and stays
frozen, and a copy of it learns from self-play. The builder proposes many
decks for each pool, and a search ranks them with the human-trained score
until the learning copy predicts the games better. The builder learns only
after that switch, because until then the search ranks its decks by a score
fit to human results, not to the games jace plays.

## One card embedding

A card encoder turns each card into one vector, its embedding, and every
model reads cards through these vectors. The encoder reads three things a
card has before anyone plays it:

- its printed text, as a frozen language model embeds it
- its engine program, the engine's own definition of what the card does
- a few printed facts, such as mana value, colours and rarity

A card's vector depends on that card alone. The encoder reads no 17Lands
statistics.

No model has an output with one slot per card. Instead, a model scores
each candidate card from that card's vector. The drafter, for example, scores
each card in a pack by the dot product of a query vector with a projection
of the card's vector. So a card the model has never seen still has a
vector, and the model can score it. This is how jace is meant to play a
new set without retraining.

Only the drafter's pretraining on human picks trains the encoder. Every
later stage reads its vectors as a fixed table. [Models](models.md)
describes the encoder and [Data](../reference/data.md) its inputs.

## A guess of the opponent's cards

In Magic, a player cannot see the opponent's hand or the rest of the
opponent's deck. A separate guess model predicts both at every decision,
and the player reads the guess as part of its input.

The guess model reads only what a player can know. It starts from the
meta, a matrix of which cards the format's decks contain together. The
meta is fitted from public decklists and then from the decks of earlier
rounds of self-play, never from the decks being played.

From the meta and the cards the opponent has shown or revealed, the guess
model works out how many copies of each card the opponent's deck is
expected to contain, less the copies already seen. This is the meta guess.
The model's network learns only corrections to it, and the corrections
start at zero, so before any training the model's guess is the meta guess.

The guess model also reads its own last few guesses in the same game, so it
does not have to work out again at each decision what it inferred at the
earlier ones.

In self-play the engine reports the opponent's true cards, so every
decision is a labelled example. The loss penalizes a confident wrong
guess more than an unsure one, which teaches the model to give chances that
match how often it is right.

To credit the player's actions, training also needs the chance to win at
every decision. A separate win-chance model estimates it from the board
and the opponent's true cards, so it does not have to guess them. Only
training reads the win chance, so the hidden cards never reach the player.

## Exploring on a schedule

The drafter and the deck builder start from human data, which puts them
near how people already play. Without exploring, they would stay there. So
jace makes its models explore:

- The player, the drafter and the builder sample their choices.
- A term in each of their losses keeps their odds from becoming near-certain early.
- The loop plays several of each pool's top-scored decks, one in each block
  of games.

A schedule sets the weights of those loss terms, and the learning rates,
for every iteration. [Training](training.md#dials-on-a-schedule) shows one.

The results of pretraining on human data and of the first run of the whole
loop are on the [Status](../status.md) page.

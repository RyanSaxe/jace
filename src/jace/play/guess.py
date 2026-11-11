"""The model that guesses the opponent's deck and hand.

The guess is the player's only view of hidden cards, and it is trained on
the true hidden cards the engine reports after the fact. The model reads the
board, the recent events and its own earlier guesses in the same game, so a
guess can build on the last one without any state carried between calls. It
scores each card by a dot product with that card's embedding and has no
parameter sized by the card pool, so the same weights work for a larger pool.
"""

import math

import torch
from torch import Tensor, nn

from jace.cards import CardEmbedding
from jace.modules import run, transformer
from jace.play.tokens import EntityEncoder, board_tokens
from jace.play.types import Boards, Earlier, Guess

__all__ = ("GuessModel",)

# A guess has two parts: the opponent's deck and the opponent's hand.
_GUESS_PARTS = 2
# The base of the wavelengths in the age encoding, the value the original transformer used for positions.
_AGE_ENCODING_BASE = 10000.0


def _age_encoding(age: Tensor, *, width: int) -> Tensor:
    # age: int64 [b, t], how many real rows are newer than each row. Returns float [b, t, width]
    # sinusoids, so the newest real row gets the same encoding however much padding there is.
    # One rate for each sine and cosine pair, so every second position up to the width.
    rates = torch.exp(torch.arange(0, width, 2, device=age.device) * (-math.log(_AGE_ENCODING_BASE) / width))
    angles = age[..., None] * rates  # [b, t, rates]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)[..., :width]


class GuessModel(nn.Module):
    """Predict the opponent's unseen cards and hand from one seat's view.

    `guess` satisfies the `Guesser` type. No
    weight is sized by the card pool, so the model scores a larger pool once
    `embedding` and `engine_to_card` are replaced by ones that cover it. The
    embedding is shared and trained by its own optimizer: leave
    `embedding.parameters()` out of this model's optimizer.
    """

    def __init__(
        self,
        embedding: CardEmbedding,
        *,
        engine_to_card: Tensor,
        sample: Boards,
        width: int,
        layers: int,
        heads: int,
    ) -> None:
        super().__init__()
        card_width = embedding.starting.shape[1]
        self.embedding: CardEmbedding = embedding
        self.width: int = width
        # engine_to_card: int64 [engine definition ids] -> CardId, -1 for a definition the pool lacks.
        self.engine_to_card: Tensor
        self.register_buffer("engine_to_card", engine_to_card.detach().clone(), persistent=False)
        # sample: any batch of boards from the engine that will be played; only its feature widths are used.
        tokens = board_tokens(sample, engine_to_card=engine_to_card)
        self.entities: EntityEncoder = EntityEncoder(tokens, card_width=card_width, width=width)
        self.events: nn.Linear = nn.Linear(tokens.events.shape[-1], width)
        self.earlier: nn.Linear = nn.Linear(_GUESS_PARTS * card_width, width)
        # queries: float [guess parts, width], the tokens whose encodings score the deck and the hand.
        self.queries: nn.Parameter = nn.Parameter(torch.randn(_GUESS_PARTS, width) * width**-0.5)
        self.encoder: nn.TransformerEncoder = transformer(width, layers=layers, heads=heads)
        self.card_keys: nn.Linear = nn.Linear(card_width, width)
        self.hand_bias: nn.Parameter = nn.Parameter(torch.zeros(()))

    def guess(self, boards: Boards, earlier: Earlier) -> Guess:
        # earlier.guesses.deck, .hand: float [b, t, cards], this game's guesses so far, newest first; t may be 0.
        # earlier.mask: bool [b, t]. Returns deck and hand, each float [b, cards].
        tokens = board_tokens(boards, engine_to_card=self.engine_to_card)
        pool = torch.arange(self.embedding.starting.shape[0], device=self.embedding.starting.device)
        card_table = run(self.embedding, pool)  # [cards, card_width]

        entities = self.entities.embed(tokens, card_table)
        # Events are oldest first and earlier guesses newest first, so their ages count in opposite directions.
        event_age = tokens.event_mask.flip(1).cumsum(1).flip(1) - 1
        events = run(self.events, tokens.events) + _age_encoding(event_age, width=self.width)
        # One token per earlier guess: its two probability vectors, each as a weighted sum of card embeddings.
        earlier_cards = torch.cat((earlier.guesses.deck @ card_table, earlier.guesses.hand @ card_table), dim=-1)
        earlier_age = earlier.mask.cumsum(1) - 1
        earlier_tokens = run(self.earlier, earlier_cards) + _age_encoding(earlier_age, width=self.width)

        batch = entities.shape[0]
        queries = self.queries.expand(batch, -1, -1)
        sequence = torch.cat((queries, entities, events, earlier_tokens), dim=1)
        query_mask = torch.ones(batch, _GUESS_PARTS, dtype=torch.bool, device=sequence.device)
        mask = torch.cat((query_mask, tokens.entity_mask, tokens.event_mask, earlier.mask), dim=1)
        encoded = self.encoder.forward(sequence, src_key_padding_mask=~mask)

        keys = run(self.card_keys, card_table)  # [cards, width]
        scores = encoded[:, :_GUESS_PARTS] @ keys.T * self.width**-0.5  # [b, guess parts, cards]
        return Guess(deck=scores[:, 0].softmax(dim=-1), hand=(scores[:, 1] + self.hand_bias).sigmoid())

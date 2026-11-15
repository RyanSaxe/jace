"""Score a deck by its chance of winning a game.

`DeckScore` reads a deck as a set of distinct cards with copy counts, so the
order of the cards cannot matter and a deck of any size fits. It reads
cards only through the shared `CardVectors`, and none of its own parameters
has a slot per card, so it can score decks that contain cards from a set it
was not trained on.

The deck builder reads its pool with a `CardSetEncoder` too, but each model
builds its own encoder, so they share no weights. The deck score reads the
mean of the card encodings, and the builder reads each card's encoding to
score that card.
"""

from collections.abc import Callable
from typing import override

import torch
from torch import Tensor, nn

from jace.cards import CardVectors
from jace.modules import DROPOUT, FEEDFORWARD_EXPANSION

__all__ = ("CardSetEncoder", "DeckScore", "EmptyCardSetError", "Scorer")

# A scorer takes float [b, cards] copy counts and returns float [b], where higher is better.
type Scorer = Callable[[Tensor], Tensor]


class EmptyCardSetError(ValueError):
    def __init__(self, row: int) -> None:
        super().__init__(f"row {row} holds no cards, so there is nothing to encode")


class CardSetEncoder(nn.Module):
    """Encode a set of cards, each with a few amounts, as one vector per row.

    `amounts` is how many amounts each card has. A card is in a row's set
    when any of its amounts is above zero. Each card in the set becomes one
    token, its projected embedding plus a learned encoding of its amounts,
    and a transformer encoder reads the tokens. `encode` returns each card's
    encoding, and calling the module returns their mean.

    Each amount is encoded as its `log1p`, so a count of 17 basic lands becomes
    2.9, close to the 0.7 to 1.6 of one to four copies. Both `encode` and
    calling the module raise `EmptyCardSetError` for a row with no card in its
    set.
    """

    def __init__(self, embedding: CardVectors, *, amounts: int, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.embedding: CardVectors = embedding
        self.project: nn.Linear = nn.Linear(embedding.width, width)
        self.amount: nn.Sequential = nn.Sequential(nn.Linear(amounts, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            width,
            heads,
            dim_feedforward=FEEDFORWARD_EXPANSION * width,
            dropout=DROPOUT,
            batch_first=True,
            norm_first=True,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def encode(self, amounts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        # `amounts` is float [b, cards, amounts]. Returns (encoded, cards, real), with one position for each of a
        # row's present cards in id order, then padding. `encoded` is float [b, longest, width], `cards` is int64
        # [b, longest] card ids, and `real` is bool [b, longest], true at a present card. A padding position has the
        # id of a card that is not present, so no id repeats in a row.
        present = amounts.gt(0).any(dim=-1)  # bool [b, cards]
        sizes = present.sum(dim=1)  # int64 [b], distinct cards in each row
        if not bool(sizes.gt(0).all()):
            raise EmptyCardSetError(int(sizes.argmin()))
        longest = int(sizes.max())
        # A stable sort of the flags puts each row's present cards first, in id order.
        cards = present.int().argsort(dim=1, descending=True, stable=True)[:, :longest]  # int64 [b, longest]
        real = present.gather(1, cards)  # bool [b, longest]
        kept = amounts.gather(1, cards.unsqueeze(-1).expand(-1, -1, amounts.shape[-1]))  # [b, longest, amounts]
        tokens = self.project(self.embedding(cards)) + self.amount(kept.log1p())
        # `forward` is called directly because it is typed and takes the mask by keyword.
        encoded = self.encoder.forward(tokens, src_key_padding_mask=~real)  # [b, longest, width]
        return encoded, cards, real

    @override
    def forward(self, amounts: Tensor) -> Tensor:
        # `amounts` is float [b, cards, amounts]. Returns float [b, width], the mean of the present cards' encodings.
        encoded, _, real = self.encode(amounts)
        weights = real.unsqueeze(-1).to(encoded.dtype)
        return (encoded * weights).sum(dim=1) / weights.sum(dim=1)


class DeckScore(nn.Module):
    def __init__(self, embedding: CardVectors, *, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.cards: CardSetEncoder = CardSetEncoder(embedding, amounts=1, width=width, layers=layers, heads=heads)
        self.head: nn.Linear = nn.Linear(width, 1)

    @override
    def forward(self, decks: Tensor) -> Tensor:
        # `decks` is float [b, cards] copy counts. Returns float [b], the logit of each deck's chance of winning a
        # game.
        encoded = self.cards(decks.unsqueeze(-1).to(torch.float32))
        return self.head(encoded).squeeze(-1)

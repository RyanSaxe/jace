"""Score a deck by its chance of winning a game.

`DeckScore` reads a deck as a set of distinct cards with copy counts, so the
order of the cards cannot matter and a deck of any size fits. Its only
knowledge of the cards comes through the shared `CardEmbedding`, and none of
its own parameters has a slot per card, so it scores decks that hold cards
from a set it was not trained on. `CardSetEncoder` is the part it shares with
the deck builder's model: both turn per-card amounts into one vector.
"""

from collections.abc import Callable
from typing import override

import torch
from torch import Tensor, nn

from jace.cards import CardEmbedding
from jace.modules import run

__all__ = ("CardSetEncoder", "DeckScore", "EmptyCardSetError", "Scorer")

# Takes float [b, cards] copy counts and returns float [b], higher is better.
type Scorer = Callable[[Tensor], Tensor]


class EmptyCardSetError(ValueError):
    def __init__(self, row: int) -> None:
        super().__init__(f"row {row} holds no cards, so there is nothing to encode")


class CardSetEncoder(nn.Module):
    """Encode a set of cards, each with a few amounts, as one vector per row.

    A card is in a row's set when any of its amounts is above zero. Each such
    card becomes one token: its projected embedding plus a learned encoding
    of its amounts. A transformer encoder reads the tokens and the result is
    their mean. Amounts enter through `log1p`, so a count of 17 basic lands
    stays in the range the encoding saw for small counts.
    """

    def __init__(self, embedding: CardEmbedding, *, amounts: int, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.embedding: CardEmbedding = embedding
        self.project: nn.Linear = nn.Linear(embedding.starting.shape[1], width)
        self.amount: nn.Sequential = nn.Sequential(nn.Linear(amounts, width), nn.GELU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=4 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    @override
    def forward(self, amounts: Tensor) -> Tensor:
        # amounts: float [b, cards, amounts]. Returns float [b, width].
        present = amounts.gt(0).any(dim=-1)  # bool [b, cards]
        sizes = present.sum(dim=1)  # int64 [b], distinct cards in each row
        if not bool(sizes.gt(0).all()):
            raise EmptyCardSetError(int(sizes.argmin()))
        longest = int(sizes.max())
        # A stable sort of the flags puts each row's present cards first, in id order.
        cards = present.int().argsort(dim=1, descending=True, stable=True)[:, :longest]  # int64 [b, longest]
        padding = ~present.gather(1, cards)  # bool [b, longest]
        kept = amounts.gather(1, cards.unsqueeze(-1).expand(-1, -1, amounts.shape[-1]))  # [b, longest, amounts]
        tokens = run(self.project, run(self.embedding, cards)) + run(self.amount, kept.log1p())
        # `forward` is called directly because it is typed and takes the mask by keyword.
        encoded = self.encoder.forward(tokens, src_key_padding_mask=padding)  # [b, longest, width]
        real = (~padding).unsqueeze(-1).to(encoded.dtype)
        return (encoded * real).sum(dim=1) / sizes.unsqueeze(-1)


class DeckScore(nn.Module):
    def __init__(self, embedding: CardEmbedding, *, width: int, layers: int, heads: int) -> None:
        super().__init__()
        self.cards: CardSetEncoder = CardSetEncoder(embedding, amounts=1, width=width, layers=layers, heads=heads)
        self.head: nn.Linear = nn.Linear(width, 1)

    @override
    def forward(self, decks: Tensor) -> Tensor:
        # decks: float [b, cards] copy counts. Returns float [b], the logit of winning a game.
        encoded = run(self.cards, decks.unsqueeze(-1).to(torch.float32))
        return run(self.head, encoded).squeeze(-1)

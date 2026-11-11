"""The drafter: a transformer over the packs a seat has seen and the cards it took.

The model reads cards only through the shared `CardEmbedding` and scores a
pack by the dot product of one query vector with each card's projected
embedding. It has no parameter whose size depends on the number of cards, so
the same weights score a larger card pool once `embedding` is replaced by one
that covers it. The embedding is trained by its own optimizer, and
`parameters()` includes it because it is a submodule, so the caller must
leave `embedding.parameters()` out of the drafter's optimizer.
"""

import math
from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor, nn

from jace.cards import CardEmbedding, CardId
from jace.draft.drafting import Drafter, SeatView
from jace.modules import run

__all__ = ("DrafterModel", "model_drafter")


class DrafterModel(nn.Module):
    """Score the cards of the current pack from the draft so far.

    Each earlier pick is one token: the picked card, the mean of the cards in
    the pack it came from, and a learned position. The current pack is one
    more token, and the encoder's output at that token is the query. `picks`
    is the largest number of picks in a draft, which sizes the positions.
    """

    def __init__(self, embedding: CardEmbedding, *, width: int, layers: int, heads: int, picks: int) -> None:
        super().__init__()
        card_width = embedding.starting.shape[1]
        self.embedding: CardEmbedding = embedding
        self.picked_card: nn.Linear = nn.Linear(card_width, width)
        self.pack_mean: nn.Linear = nn.Linear(card_width, width)
        self.candidate: nn.Linear = nn.Linear(card_width, width)
        self.positions: nn.Embedding = nn.Embedding(picks, width)
        # Stands where an earlier token has its picked card.
        self.current: nn.Parameter = nn.Parameter(torch.zeros(width))
        layer = nn.TransformerEncoderLayer(
            width, heads, dim_feedforward=4 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False
        )

    @override
    def forward(self, seen: Tensor, picks: Tensor, *, pack: Tensor) -> Tensor:
        # seen: [b, t, cards] 0/1, the earlier packs. A row of zeros marks a step as padding,
        # which must come after the row's real steps.
        # picks: int64 [b, t], the card taken from each earlier pack. Any card id at a padded step.
        # pack: [b, cards] 0/1, the current pack.
        # Returns float [b, cards]: logits, with cards outside the pack at -inf.
        steps = seen.shape[1]
        if steps >= self.positions.num_embeddings:
            message = f"{steps} earlier picks do not fit a model built for {self.positions.num_embeddings} picks"
            raise ValueError(message)
        cards = run(self.embedding, torch.arange(seen.shape[2], device=seen.device))  # [cards, card_width]
        padding = seen.sum(dim=-1) == 0  # [b, t]
        earlier = (
            run(self.picked_card, cards[picks])
            + run(self.pack_mean, _mean_card(seen, cards))
            + run(self.positions, torch.arange(steps, device=seen.device))
        )  # [b, t, width]
        current = (
            self.current
            + run(self.pack_mean, _mean_card(pack, cards))
            + run(self.positions, steps - padding.sum(dim=-1))
        )  # [b, width]
        tokens = torch.cat([earlier, current.unsqueeze(1)], dim=1)  # [b, t + 1, width]
        ignored = torch.cat([padding, padding.new_zeros(len(padding), 1)], dim=1)  # [b, t + 1]
        query = self.encoder.forward(tokens, src_key_padding_mask=ignored)[:, -1]  # [b, width]
        candidates = run(self.candidate, cards)  # [cards, width]
        scores = query @ candidates.T / math.sqrt(candidates.shape[1])  # [b, cards]
        return scores.masked_fill(pack == 0, float("-inf"))


def _mean_card(packs: Tensor, cards: Tensor) -> Tensor:
    # packs: [..., cards] 0/1. cards: [cards, card_width]. Returns [..., card_width], zeros for an empty pack.
    members = packs.to(cards.dtype)
    return members @ cards / members.sum(dim=-1, keepdim=True).clamp_min(1.0)


def model_drafter(model: DrafterModel, *, generator: torch.Generator | None) -> Drafter:
    """Make a drafter that picks with `model` for any number of seats in one forward pass.

    With a generator, each pick is sampled from the softmax of the model's
    logits over the pack. With `None`, each pick is the card with the highest
    logit. The model runs without gradients, so the picks cannot train it.
    Training recomputes the logits from the recorded views.
    """

    def draft(views: Sequence[SeatView]) -> tuple[CardId, ...]:
        if not views:
            return ()
        starting = model.embedding.starting
        with torch.no_grad():
            seen, picks, pack = _view_tensors(views, cards=starting.shape[0], device=starting.device)
            logits = model.forward(seen, picks, pack=pack)
            if generator is None:
                chosen = logits.argmax(dim=-1)
            else:
                chances = logits.softmax(dim=-1).to(generator.device)
                chosen = torch.multinomial(chances, 1, generator=generator).squeeze(1)
        return tuple(int(card) for card in chosen.tolist())

    return draft


def _view_tensors(views: Sequence[SeatView], *, cards: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    # Returns `seen`, `picks` and `pack` as `DrafterModel.forward` takes them, with shorter
    # histories padded at the end.
    steps = max(view.pick_number for view in views)
    seen = torch.zeros(len(views), steps, cards, device=device)
    pack = torch.zeros(len(views), cards, device=device)
    picks = torch.zeros(len(views), steps, dtype=torch.int64, device=device)
    for row, view in enumerate(views):
        pack[row, list(view.pack)] = 1.0
        for step, (earlier_pack, pick) in enumerate(zip(view.seen, view.picks, strict=True)):
            seen[row, step, list(earlier_pack)] = 1.0
            picks[row, step] = pick
    return seen, picks, pack

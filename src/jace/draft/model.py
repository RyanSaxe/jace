"""The drafter: a transformer over the packs a seat has seen and the cards it took.

The model reads cards only through its `CardVectors`. It scores a pack by
the dot product of one query vector with each card's projected vector. No
parameter's size depends on the number of cards, so the same weights can
score a larger card pool once `embedding` is replaced by one that covers it.

In pretraining, the card vectors are a `CardEncoder` that has its own
optimizer. The encoder is a submodule, so `parameters()` includes its
parameters, and the caller must leave them out of the drafter's optimizer,
for example with `jace.cards.own_parameters`.

An earlier pick's token attends only to itself and to the picks before it.
That is what lets `whole_draft` score every pick of a recorded draft in one
pass and get the same logits that `forward` gives pick by pick.
"""

import math
from collections.abc import Sequence
from typing import override

import torch
from torch import Tensor, nn

from jace.cards import CardId, CardVectors
from jace.draft.drafting import Drafter, SeatView
from jace.modules import DROPOUT, transformer

__all__ = ("DrafterModel", "model_drafter", "view_tensors")


class DrafterModel(nn.Module):
    """Score the cards of the current pack from the draft so far.

    Each earlier pick becomes one token, the sum of a vector for the picked
    card, one for the mean of the cards in the pack it came from, and a
    learned position. The current pack is one more token, and the encoder's
    output at that token is the query. `picks` is the largest number of
    picks in a draft, and the model learns one position for each.
    `dropout` is the encoder's, and acts only in training mode.
    """

    def __init__(
        self, embedding: CardVectors, *, width: int, layers: int, heads: int, picks: int, dropout: float = DROPOUT
    ) -> None:
        super().__init__()
        card_width = embedding.width
        self.embedding: CardVectors = embedding
        self.picked_card: nn.Linear = nn.Linear(card_width, width)
        self.pack_mean: nn.Linear = nn.Linear(card_width, width)
        self.candidate: nn.Linear = nn.Linear(card_width, width)
        self.positions: nn.Embedding = nn.Embedding(picks, width)
        # The current pack's token has no picked card, so this learned vector takes that card's place.
        self.current: nn.Parameter = nn.Parameter(torch.zeros(width))
        self.encoder: nn.TransformerEncoder = transformer(width, layers=layers, heads=heads, dropout=dropout)

    @override
    def forward(self, seen: Tensor, picks: Tensor, *, pack: Tensor) -> Tensor:
        # `seen` is 0/1 [b, t, cards], the earlier packs. A row of zeros marks a step as padding, and padding must
        # come after the real steps of its row. `picks` is int64 [b, t], the card taken from each earlier pack, and
        # may be any card id at a padded step. `pack` is 0/1 [b, cards], the current pack. Returns float [b, cards]
        # logits, with -inf for the cards outside the pack.
        steps = seen.shape[1]
        if steps >= self.positions.num_embeddings:
            message = f"{steps} earlier picks do not fit a model built for {self.positions.num_embeddings} picks"
            raise ValueError(message)
        cards = self.embedding(torch.arange(seen.shape[2], device=seen.device))  # [cards, card_width]
        padding = seen.sum(dim=-1) == 0  # [b, t]
        earlier = (
            self.picked_card(cards[picks])
            + self.pack_mean(_mean_card(seen, cards))
            + self.positions(torch.arange(steps, device=seen.device))
        )  # [b, t, width]
        # The current pack takes the position after the last real earlier step.
        current = (
            self.current + self.pack_mean(_mean_card(pack, cards)) + self.positions(steps - padding.sum(dim=-1))
        )  # [b, width]
        tokens = torch.cat([earlier, current.unsqueeze(1)], dim=1)  # [b, t + 1, width]
        ignored = torch.cat([padding, padding.new_zeros(len(padding), 1)], dim=1)  # [b, t + 1]
        order = torch.arange(steps + 1, device=seen.device)
        later = order[:, None] < order[None, :]  # [t + 1, t + 1], True where a token may not attend
        query = self.encoder.forward(tokens, mask=later, src_key_padding_mask=ignored)[:, -1]  # [b, width]
        candidates = self.candidate(cards)  # [cards, width]
        scores = query @ candidates.T / math.sqrt(candidates.shape[1])  # [b, cards]
        return scores.masked_fill(pack == 0, float("-inf"))

    def whole_draft(self, seen: Tensor, picks: Tensor) -> Tensor:
        """Score every pick of whole drafts in one pass, for training on recorded drafts.

        `seen` is `[b, t, cards]` 0/1 and `picks` is int64 `[b, t]`, as in
        `forward`, but covering every step of the draft. Returns float
        `[b, t, cards]` whose row `i` equals
        `forward(seen[:, :i], picks[:, :i], pack=seen[:, i])`. A padded step,
        a row of zeros in `seen`, gets -inf for every card.

        Raises `ValueError` when the drafts have more steps than the model
        has positions.
        """
        steps = seen.shape[1]
        if steps > self.positions.num_embeddings:
            message = f"{steps} picks do not fit a model built for {self.positions.num_embeddings} picks"
            raise ValueError(message)
        cards = self.embedding(torch.arange(seen.shape[2], device=seen.device))  # [cards, card_width]
        order = torch.arange(steps, device=seen.device)
        packs = self.pack_mean(_mean_card(seen, cards)) + self.positions(order)  # [b, t, width]
        # Each step appears twice. The first t tokens contain each step's pick, and later tokens attend to them. The
        # last t tokens are the queries that score each step's pack.
        tokens = torch.cat([self.picked_card(cards[picks]) + packs, self.current + packs], dim=1)
        later = order[:, None] < order[None, :]  # [t, t]
        other = order[:, None] != order[None, :]  # [t, t]
        # Row i of the mask is the token that attends, and column j a token it may attend to. A picked token attends
        # to the picked tokens up to its own. A query attends to the picked tokens before its step and, of the
        # queries, only to itself.
        blocked = torch.cat(
            [torch.cat([later, torch.ones_like(later)], dim=1), torch.cat([later | ~other, other], dim=1)]
        )  # [2t, 2t], True where a token may not attend
        queries = self.encoder.forward(tokens, mask=blocked)[:, steps:]  # [b, t, width]
        candidates = self.candidate(cards)  # [cards, width]
        scores = queries @ candidates.T / math.sqrt(candidates.shape[1])  # [b, t, cards]
        return scores.masked_fill(seen == 0, float("-inf"))


def _mean_card(packs: Tensor, cards: Tensor) -> Tensor:
    # `packs` is 0/1 [..., cards], and `cards` is [cards, card_width]. Returns [..., card_width], the mean card vector
    # of each pack, or zeros for an empty pack.
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
        with torch.no_grad():
            seen, picks, pack = view_tensors(views, cards=model.embedding.cards, device=model.embedding.device)
            logits = model.forward(seen, picks, pack=pack)
            if generator is None:
                chosen = logits.argmax(dim=-1)
            else:
                chances = logits.softmax(dim=-1).to(generator.device)
                chosen = torch.multinomial(chances, 1, generator=generator).squeeze(1)
        return tuple(int(card) for card in chosen.tolist())

    return draft


def view_tensors(views: Sequence[SeatView], *, cards: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor]:
    # Returns `seen`, `picks` and `pack` as `DrafterModel.forward` takes them, with shorter
    # histories padded at the end. Training uses it to recompute the logits of recorded picks.
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

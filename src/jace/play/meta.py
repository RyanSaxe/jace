"""The meta: what the decks of a format look like, as statistics of which cards they contain together.

A player cannot know the decks it will play against, but it can know what decks
of the format have looked like. The meta describes that with plain statistics
over decks. It contains each card's mean copies per deck and, for each ordered
pair of cards (i, j), the mean copies of j in the decks that contain at least
one i. Nothing in it names an archetype or a colour, so whatever the decks have
in common shows only in which cards they contain together. The guess model
starts its guess from the meta.

`day_zero_meta` builds the meta of a format before any deck is known. It
takes each card's copies from how often the format's boosters contain it,
and it makes every card independent of every other. `fit_meta` fits a meta
from decks, and `blend_meta` blends newer decks into a meta and weights the
older ones down. Both measure which cards the decks contain together.

The decks of a meta are the decks it was fitted or blended from. A card has a
history in a meta once one of the meta's decks contains it. A card with no
history takes its copies from the meta that `fit_meta` falls back on or that
`blend_meta` blends into. In the end that is the day-0 meta, so the card's
copies come from the boosters. The card is also independent of every other
card, because there is no deck to measure its co-occurrence from.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ("Meta", "blend_meta", "day_zero_meta", "fit_meta")


@dataclass(frozen=True, slots=True)
class Meta:
    """Card co-occurrence over the decks of a format, and how many decks it was fitted from.

    `copies` is float `[cards]`, each card's mean copies per deck.

    `conditional` is float `[cards, cards]`. Row i, column j is the mean
    copies of j in the decks that contain at least one i, so the diagonal is
    each card's mean copies in the decks that contain it.

    `holding` is float `[cards]`, the weighted count of decks that contain
    each card. `decks` is a float tensor with no dimensions, the weighted
    count of the meta's decks. A deck's weight is 1 when it is fitted or
    blended in, and each later blend multiplies its weight by that blend's
    decay.

    `legal` is bool `[cards]`, the cards that the format's boosters contain.
    A deck of the format may contain no other card.
    """

    copies: Tensor
    conditional: Tensor
    holding: Tensor
    decks: Tensor
    legal: Tensor


def _independent(copies: Tensor, *, own: Tensor) -> Tensor:
    # copies and own are float [cards]. Returns the `conditional` of independent cards, in which the decks
    # that contain one card have the usual mean copies of every other: each row is `copies`, except that the
    # diagonal is `own`.
    rows = copies.expand(len(copies), -1).clone()
    rows.diagonal().copy_(own)
    return rows


def day_zero_meta(booster_copies: Tensor, *, deck_size: int) -> Meta:
    """Build the meta of a format before any of its decks is known.

    `booster_copies` is float `[cards]`, each card's expected copies in one
    booster, and zero for a card that no booster contains. The legal cards
    are those above zero.

    In the day-0 meta, a deck is `deck_size` cards, each drawn on its own
    with the boosters' odds, so every card is independent of every other.
    For a card that some deck contains, `fit_meta` and `blend_meta` replace
    that independence with co-occurrence measured from the decks.

    The day-0 meta has no decks, so its `holding` and `decks` are zero and
    no card has a history.
    """
    share = booster_copies / booster_copies.sum()
    copies = deck_size * share
    legal = booster_copies > 0
    # A card whose share of a booster's cards is s has Binomial(deck_size, s) copies in a deck. In the decks
    # that contain it, the mean is the overall mean divided by the chance of at least one copy.
    in_holding_decks = torch.where(legal, copies / (1.0 - (1.0 - share) ** deck_size), 0.0)
    return Meta(
        copies=copies,
        conditional=_independent(copies, own=in_holding_decks),
        holding=torch.zeros_like(copies),
        decks=torch.tensor(0.0),
        legal=legal,
    )


def _from_sums(*, decks: Tensor, holding: Tensor, copy_sums: Tensor, pair_sums: Tensor, fallback: Meta) -> Meta:
    # copy_sums is float [cards], each card's weighted copies summed over the decks. pair_sums is float
    # [cards, cards], and row i is the same sum over only the decks that contain card i.
    # A card with no history gets `fallback`'s copies and diagonal entry, and is independent of every other
    # card: its row of `conditional` is every card's mean copies, and its entry in every other row is its own
    # mean copies. Dividing by zero decks gives NaN, but torch.where never picks those entries.
    history = holding > 0
    copies = torch.where(history, copy_sums / decks, fallback.copies)
    both = history[:, None] & history[None, :]
    independent = _independent(copies, own=fallback.conditional.diagonal())
    conditional = torch.where(both, pair_sums / holding[:, None], independent)
    return Meta(copies=copies, conditional=conditional, holding=holding, decks=decks, legal=fallback.legal)


def fit_meta(copies: Tensor, *, fallback: Meta) -> Meta:
    """Fit the meta of `copies`, with each deck weighing one.

    `copies` is float `[decks, cards]`, each deck's copies of each card.

    A card that none of the decks contain takes its mean copies, and its
    mean copies in the decks that contain it, from `fallback`. That is
    usually the day-0 meta. The card is independent of every other card,
    because there is no deck to measure its co-occurrence from. When there
    are no decks, this holds for every card. The meta's legal cards are
    `fallback`'s.
    """
    held = (copies > 0).float()
    return _from_sums(
        decks=torch.tensor(float(len(copies))),
        holding=held.sum(dim=0),
        copy_sums=copies.sum(dim=0),
        pair_sums=held.T @ copies,
        fallback=fallback,
    )


def blend_meta(meta: Meta, copies: Tensor, *, decay: float) -> Meta:
    """Blend the decks of `copies` into `meta`, after multiplying the weight of its decks by `decay`.

    `copies` is float `[decks, cards]`, and each new deck's weight is 1. With
    `decay` 1, the result is the meta that fitting all the decks at once
    would give. With a smaller `decay`, the older decks count for less, so
    the meta changes as the decks change over time.

    For a card that neither the new decks nor `meta`'s decks contain, the
    mean copies and the mean copies in the decks that contain it are those
    of `meta`. The card is independent of every other card.
    """
    history = meta.holding > 0
    # A card with no history appears in none of the meta's decks, so its sums are zero, whatever fallback
    # values the meta has for it.
    copy_sums = torch.where(history, meta.copies, 0.0) * meta.decks
    pair_sums = torch.where(history[:, None] & history[None, :], meta.conditional, 0.0) * meta.holding[:, None]
    held = (copies > 0).float()
    return _from_sums(
        decks=decay * meta.decks + len(copies),
        holding=decay * meta.holding + held.sum(dim=0),
        copy_sums=decay * copy_sums + copies.sum(dim=0),
        pair_sums=decay * pair_sums + held.T @ copies,
        fallback=meta,
    )

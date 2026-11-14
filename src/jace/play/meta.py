"""The meta: what the decks of a format look like, as statistics of which cards they hold together.

A player cannot know the decks it will meet, but it can know what decks of
the format have looked like. The meta holds that as plain statistics over
decks: each card's mean copies per deck and, for each ordered pair of cards
(i, j), the mean copies of j in the decks that hold at least one i. Nothing
in it names an archetype or a colour, so whatever the decks have in common
shows only in which cards they hold together. The guess model starts its
guess from it.

`fit_meta` fits a meta from decks, `blend_meta` blends newer decks into a
meta with the older ones weighted down, and `day_zero_meta` is the meta
before any deck: each card's copies from how often the format's boosters
hold it, with cards independent. A card that no deck behind a meta holds
has no history and keeps what the meta it was fitted onto says of it,
which in the end is the day-0 meta's: its copies from the boosters, and
independence from every other card.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = ("Meta", "blend_meta", "day_zero_meta", "fit_meta")


@dataclass(frozen=True, slots=True)
class Meta:
    """Card co-occurrence over the decks of a format, and how many decks it rests on.

    `copies` is float `[cards]`, each card's mean copies per deck.
    `conditional` is float `[cards, cards]`: row i, column j is the mean
    copies of j in the decks that hold at least one i, so its diagonal is
    each card's mean copies in the decks that hold it. `holding` is float
    `[cards]`, the effective number of decks that hold each card, and
    `decks` the effective number behind the meta, a float tensor of no
    dimensions: a deck blended in with decay d counts 1, then d after the
    next blend, d**2 after the one after. `legal` is bool `[cards]`, the
    cards the format's boosters hold, which are the only ones a deck of the
    format may hold.
    """

    copies: Tensor
    conditional: Tensor
    holding: Tensor
    decks: Tensor
    legal: Tensor


def _independent(copies: Tensor, *, own: Tensor) -> Tensor:
    # copies, own: float [cards]. Returns the conditional copies of cards that tell nothing about each
    # other: every row is `copies`, except that each card's own column holds `own`.
    rows = copies.expand(len(copies), -1).clone()
    rows.diagonal().copy_(own)
    return rows


def day_zero_meta(booster_copies: Tensor, *, deck_size: int) -> Meta:
    """Build the meta of a format before any of its decks is known.

    `booster_copies` is float `[cards]`, each card's expected copies in one
    booster, zero for a card no booster holds. With cards independent, a
    deck is `deck_size` cards each drawn with the boosters' odds, so a card
    whose share of a booster's cards is s holds Binomial(deck_size, s)
    copies: deck_size * s per deck on average, and
    deck_size * s / (1 - (1 - s) ** deck_size) on average in the decks that
    hold it. No deck is behind it, so every card has no history.
    """
    share = booster_copies / booster_copies.sum()
    copies = deck_size * share
    legal = booster_copies > 0
    in_holding_decks = torch.where(legal, copies / (1.0 - (1.0 - share) ** deck_size), 0.0)
    return Meta(
        copies=copies,
        conditional=_independent(copies, own=in_holding_decks),
        holding=torch.zeros_like(copies),
        decks=torch.tensor(0.0),
        legal=legal,
    )


def _from_sums(*, decks: Tensor, holding: Tensor, copy_sums: Tensor, pair_sums: Tensor, fallback: Meta) -> Meta:
    # copy_sums: float [cards], each card's copies summed over the decks, each deck times its weight.
    # pair_sums: float [cards, cards], row i the same over the decks that hold i. A card no deck holds keeps
    # `fallback`'s copies and its own diagonal, and is independent of every other card: the decks that would
    # hold it hold every other card's mean copies, and the decks that hold another card hold its mean copies.
    # Where a sum is divided by no deck, its NaN is never taken.
    history = holding > 0
    copies = torch.where(history, copy_sums / decks, fallback.copies)
    both = history[:, None] & history[None, :]
    independent = _independent(copies, own=fallback.conditional.diagonal())
    conditional = torch.where(both, pair_sums / holding[:, None], independent)
    return Meta(copies=copies, conditional=conditional, holding=holding, decks=decks, legal=fallback.legal)


def fit_meta(copies: Tensor, *, fallback: Meta) -> Meta:
    """Fit the meta of `copies`, each deck weighing one.

    `copies` is float `[decks, cards]`, each deck's copies of each card. A
    card none of the decks holds keeps what `fallback` says of it, which is
    usually the day-0 meta, and so does every card when there are no decks.
    The meta's legal cards are `fallback`'s.
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
    """Blend the decks of `copies` into `meta`, every deck already behind it weighted by `decay`.

    `copies` is float `[decks, cards]`, and each new deck weighs one. With
    `decay` 1 the result is the meta of every deck at once, and with a
    smaller one the older decks fade, so the meta follows decks that change
    over time. A card that no deck behind either holds keeps what `meta`
    says of it.
    """
    history = meta.holding > 0
    # A card with no history has no copies in any deck behind the meta, whatever its fallback values say.
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

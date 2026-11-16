"""Check the meta that `fit_meta` and `blend_meta` compute from a hand-made set of decks.

A meta contains each card's mean copies per deck and, for each pair of cards,
the mean copies of one card in the decks that contain the other. The three
decks here contain only cards 0 to 3. Card 4 is legal but in none of them, and
card 5 is not legal, because the format's boosters never contain it. A card
that no deck contains gets its copies from the day-0 meta, which is the meta
before any deck is known, and is independent of every other card. Here the
day-0 meta is that of 10-card decks drawn from boosters that contain cards 0 to
4 equally often.
"""

import pytest
import torch

from jace.play import blend_meta, day_zero_meta, fit_meta

__all__ = ()

DECK_SIZE = 10
# Each deck's copies of cards 0 to 5.
DECKS = torch.tensor(
    [
        [2.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 3.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
    ]
)
NEWER_DECK = torch.tensor([[0.0, 2.0, 0.0, 1.0, 0.0, 0.0]])


def test_the_meta_holds_mean_and_conditional_copies_and_blending_weighs_older_decks_down() -> None:
    day_zero = day_zero_meta(torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 0.0]), deck_size=DECK_SIZE)

    meta = fit_meta(DECKS, fallback=day_zero)
    blended_whole = blend_meta(meta, NEWER_DECK, decay=1.0)
    blended_half = blend_meta(meta, NEWER_DECK, decay=0.5)

    # No deck contains card 4, so its mean copies come from the day-0 meta. Each of a deck's 10 cards is card 4
    # with chance 1/5, which gives 2 copies on average.
    assert torch.allclose(meta.copies, torch.tensor([1.0, 1 / 3, 1.0, 4 / 3, 2.0, 0.0]))
    # The two decks that contain card 0 have, on average, 1.5 copies of it, half a copy of card 1 and 1.5
    # copies of card 2.
    assert torch.allclose(meta.conditional[0, :4], torch.tensor([1.5, 0.5, 1.5, 0.0]))
    assert torch.allclose(meta.conditional[2, :4], torch.tensor([1.0, 0.0, 3.0, 0.0]))
    # Card 4 is independent of the other cards in the meta. The mean copies of card 4 in the decks that contain
    # another card are its overall mean of 2, and the mean copies of another card in the decks that contain
    # card 4 are that card's overall mean. A deck that contains card 4 has at least one copy, so its mean
    # copies of card 4 are those of a Binomial(10, 1/5) count of at least one, 2 / (1 - 0.8**10).
    assert torch.allclose(meta.conditional[:4, 4], torch.full((4,), 2.0))
    assert torch.allclose(meta.conditional[4, :4], meta.copies[:4])
    assert meta.conditional[4, 4].item() == pytest.approx(2.0 / (1.0 - 0.8**DECK_SIZE))
    assert meta.decks.item() == 3.0
    assert torch.equal(meta.legal, torch.tensor([True, True, True, True, True, False]))
    # A decay of 1 blends the newer deck in as if all four decks were fitted at once. With a decay of one half,
    # each older deck counts as half a deck, so the blend's decks count as 2.5 decks and card 1's mean copies are
    # (0.5 * 1 + 2) / (0.5 * 3 + 1) = 1.
    all_at_once = fit_meta(torch.cat((DECKS, NEWER_DECK)), fallback=day_zero)
    assert torch.allclose(blended_whole.copies, all_at_once.copies)
    assert torch.allclose(blended_whole.conditional, all_at_once.conditional)
    assert blended_half.copies[1].item() == pytest.approx(1.0)
    assert blended_half.decks.item() == 2.5

"""Check that the meta holds the right copies for a hand-made set of decks.

Cards 0 to 3 are held by the decks, card 4 is legal but held by none, and
card 5 is not legal: the format's boosters never hold it. The day-0 meta is
fitted onto a format of 10-card decks whose boosters hold cards 0 to 4
equally often.
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

    # Card 4 has no history, so it keeps its booster odds: 10 cards drawn with a share of 1/5 each.
    assert torch.allclose(meta.copies, torch.tensor([1.0, 1 / 3, 1.0, 4 / 3, 2.0, 0.0]))
    # The decks holding card 0 hold 1.5 copies of it, half a copy of card 1, and 1.5 of card 2.
    assert torch.allclose(meta.conditional[0, :4], torch.tensor([1.5, 0.5, 1.5, 0.0]))
    assert torch.allclose(meta.conditional[2, :4], torch.tensor([1.0, 0.0, 3.0, 0.0]))
    # Card 4 tells nothing about the others, and the decks that hold it hold Binomial(10, 1/5) copies of it.
    assert torch.allclose(meta.conditional[:4, 4], torch.full((4,), 2.0))
    assert torch.allclose(meta.conditional[4, :4], meta.copies[:4])
    assert meta.conditional[4, 4].item() == pytest.approx(2.0 / (1.0 - 0.8**DECK_SIZE))
    assert meta.decks.item() == 3.0
    assert torch.equal(meta.legal, torch.tensor([True, True, True, True, True, False]))
    # Blending with no decay is fitting every deck at once, and with a decay of one half each older deck
    # counts half: card 1's mean copies are (0.5 * 1 + 2) / (0.5 * 3 + 1).
    all_at_once = fit_meta(torch.cat((DECKS, NEWER_DECK)), fallback=day_zero)
    assert torch.allclose(blended_whole.copies, all_at_once.copies)
    assert torch.allclose(blended_whole.conditional, all_at_once.conditional)
    assert blended_half.copies[1].item() == pytest.approx(1.0)
    assert blended_half.decks.item() == 2.5

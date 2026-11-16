"""Check that the evaluation numbers mean what a reader takes them to mean.

Both tests use results whose answer is known. The win-rate test scores 20
games and then 2,000 games, half of them won each time, so only the number
of games differs. The guess test states each card's true chance, one of
0.05, 0.35 and 0.85, and draws whether the card is in hand at that chance.
Each of the three chances falls in its own range of stated chance, and in
each of those ranges the report's stated and observed chances must match.
"""

import torch

from jace.eval import check_guess, win_rate
from jace.play import Guess

__all__ = ()


def test_the_range_around_a_win_rate_narrows_as_games_are_added() -> None:
    generator = torch.Generator().manual_seed(0)
    few = torch.tensor([1.0, 0.0] * 10)
    many = torch.tensor([1.0, 0.0] * 1000)

    small = win_rate(few, resamples=2000, generator=generator)
    large = win_rate(many, resamples=2000, generator=generator)

    assert small.rate == large.rate == 0.5
    assert small.low < 0.5 < small.high
    assert (large.high - large.low) < (small.high - small.low) / 4


def test_a_guess_that_states_true_frequencies_reads_as_honest_and_beats_the_base_rate() -> None:
    generator = torch.Generator().manual_seed(0)
    # Each card's chance changes from one decision to the next, as it does when the guess model reads the board. The
    # base rate gives each card its frequency over all the decisions, and it would guess as well as the true
    # chances if those never changed.
    chance = torch.tensor([[0.05, 0.35, 0.85], [0.85, 0.05, 0.35]]).repeat(10000, 1)
    truth = (torch.rand(chance.shape, generator=generator) < chance).float()

    report = check_guess(Guess(deck=chance, hand=chance), hand_truth=truth)

    assert report.loss < report.base_rate_loss
    for bucket, expected in ((0, 0.05), (3, 0.35), (8, 0.85)):
        assert abs(report.observed[bucket] - expected) < 0.02
        assert abs(report.stated[bucket] - expected) < 0.001

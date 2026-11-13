"""Check that the evaluation numbers mean what a reader takes them to mean.

A win rate's range must tighten as games are added, and an honest guess must
read as honest. Both are checked on results with a known answer.
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
    # Each card's chance changes from one decision to the next, as it does when a guess reads the board.
    chance = torch.tensor([[0.05, 0.35, 0.85], [0.85, 0.05, 0.35]]).repeat(10000, 1)
    truth = (torch.rand(chance.shape, generator=generator) < chance).float()

    report = check_guess(Guess(deck=chance, hand=chance), hand_truth=truth)

    assert report.loss < report.base_rate_loss
    for bucket, expected in ((0, 0.05), (3, 0.35), (8, 0.85)):
        assert abs(report.observed[bucket] - expected) < 0.02
        assert abs(report.stated[bucket] - expected) < 0.001

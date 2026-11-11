"""Check that the evaluation numbers mean what a reader takes them to mean.

A win rate's range must tighten as games are added, and an honest guess must
read as honest. Both are checked on results with a known answer.
"""

import torch

from jace.eval import check_guess, learner_results, win_rate
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


def test_each_game_counts_once_for_the_learner_however_many_decisions_it_had() -> None:
    # Game 7: the learner (seat 0) decides twice and wins. Game 9: it decides once and loses.
    games = torch.tensor([7, 7, 7, 9, 9])
    learner = torch.tensor([True, False, True, False, True])
    results = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])

    assert learner_results(games, learner=learner, results=results).tolist() == [1.0, 0.0]


def test_a_guess_that_states_true_frequencies_reads_as_honest_and_beats_the_base_rate() -> None:
    generator = torch.Generator().manual_seed(0)
    chance = torch.tensor([0.05, 0.35, 0.85]).repeat(20000, 1)
    truth = (torch.rand(chance.shape, generator=generator) < chance).float()

    report = check_guess(Guess(deck=chance, hand=chance), hand_truth=truth)

    assert report.loss < report.base_rate_loss
    for bucket, expected in ((0, 0.05), (3, 0.35), (8, 0.85)):
        assert abs(report.observed[bucket] - expected) < 0.02
        assert abs(report.stated[bucket] - expected) < 0.001

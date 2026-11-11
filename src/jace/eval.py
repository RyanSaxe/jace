"""Measure a player and a guess model from the decisions of finished games.

Evaluation plays games with `jace.train.collect`, the same code that
training uses, so there is one way to play a game. This module turns the
resulting rollout into the numbers a person reads: a win rate with a range
that says how much it could move by chance, and whether the guess model's
stated chances match how often they come true.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from jace.play import Guess

__all__ = ("GuessReport", "WinRate", "check_guess", "learner_results", "win_rate")


@dataclass(frozen=True, slots=True)
class WinRate:
    rate: float
    # low, high: the 2.5th and 97.5th percentiles of the rate when the games are resampled.
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class GuessReport:
    """How good and how honest the hand guess is.

    `loss` and `base_rate_loss` are the average log loss of the guess and of
    a guess that gives every card its overall frequency; the model is useful
    only when `loss` is lower. `stated` and `observed` hold, for each of ten
    equal ranges of stated chance, the mean chance the model stated and how
    often the card was really in hand; they match when the model is honest.
    A range with no guesses in it is NaN in both.
    """

    loss: float
    base_rate_loss: float
    stated: tuple[float, ...]
    observed: tuple[float, ...]


def learner_results(games: Tensor, *, learner: Tensor, results: Tensor) -> Tensor:
    # games: int64 [n]; learner: bool [n]; results: [n]. Returns the learner's result in each game, [games].
    mine = games[learner]
    order = torch.argsort(mine, stable=True)
    sorted_games = mine[order]
    first = torch.ones_like(sorted_games, dtype=torch.bool)
    first[1:] = sorted_games[1:] != sorted_games[:-1]
    return results[learner][order][first]


def win_rate(results: Tensor, *, resamples: int, generator: torch.Generator) -> WinRate:
    """Summarize per-game results (1 win, 0.5 draw, 0 loss) as a rate with a range.

    The range comes from resampling the games with replacement `resamples`
    times, so it reflects how many games were played. It treats games as
    independent, which overstates certainty when many games share a deck or
    a deal.
    """
    picks = torch.randint(0, len(results), (resamples, len(results)), generator=generator)
    rates = results[picks].mean(dim=1)
    low, high = rates.quantile(torch.tensor([0.025, 0.975])).tolist()
    return WinRate(rate=float(results.mean()), low=low, high=high)


def _log_loss(chance: Tensor, truth: Tensor) -> float:
    safe = chance.clamp(1e-6, 1.0 - 1e-6)
    return float(-(truth * safe.log() + (1.0 - truth) * (1.0 - safe).log()).mean())


def check_guess(guess: Guess, *, hand_truth: Tensor) -> GuessReport:
    # guess.hand, hand_truth: [n, cards]. Scores every (decision, card) pair.
    stated = guess.hand.flatten()
    truth = hand_truth.flatten()
    bucket = (stated * 10).long().clamp_max(9)
    counts = torch.zeros(10).scatter_add(0, bucket, torch.ones_like(stated))
    stated_means = torch.zeros(10).scatter_add(0, bucket, stated) / counts
    observed_means = torch.zeros(10).scatter_add(0, bucket, truth) / counts
    return GuessReport(
        loss=_log_loss(stated, truth),
        base_rate_loss=_log_loss(truth.mean().expand_as(truth), truth),
        stated=tuple(stated_means.tolist()),
        observed=tuple(observed_means.tolist()),
    )

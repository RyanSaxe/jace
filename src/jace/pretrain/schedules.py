"""Dials that change over training, and small functions to schedule them.

A dial is a value that training reads and that can change from one iteration to
the next, such as a learning rate or the weight of a loss term. A loop calls a
schedule with each iteration's number and gets that iteration's dials back, so
how training changes over time is one function a user can read and replace. A
phase is a stretch of iterations in which that function behaves one way, such
as the drafter's rate staying at zero until the player has had time to learn.

The limited schedule follows Ataraxos, a Stratego agent trained by self-play.
Like Ataraxos, it lowers the player's learning rate and the player's pull
toward even odds as a power of the iteration number. Pretraining on human data
has one dial, a factor on the learning rate, which `warm_up_then_decay`
computes and `scale_rates` applies.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torch.optim import Optimizer

__all__ = (
    "Dials",
    "RateScale",
    "Schedule",
    "current_rates",
    "phased_schedule",
    "power_decay",
    "ramp",
    "scale_rates",
    "warm_up_then_decay",
)

# The player's learning rate and its pull toward even odds follow Ataraxos's schedules, from Table S7 of its supplement.
_RATE_SCALE = 0.5
_RATE_POWER = 1.1
_RATE_FLOOR = 5e-6
_RATE_CAP = 1e-4
_SPREAD_START = 0.05
_SPREAD_POWER = 0.3

# First guesses with no source, to be replaced by measured values. The builder's rate and entropy weight equal the
# drafter's, because both models take a policy-gradient step on their decks' win rates.
_DRAFTER_RATE = 1e-5
_SCORE_RATE = 1e-4
_BUILDER_RATE = 1e-5
_DRAFTER_ENTROPY = 0.01
_BUILDER_ENTROPY = 0.01
# The drafter's rate ramps up over this fraction of the iterations it was frozen for.
_DRAFTER_WARM_UP = 0.1

# In pretraining on human data, the rate rises from zero over the first _WARM_UP_SHARE of the steps and then falls
# in a straight line to _LAST_RATE_SHARE of its peak.
_WARM_UP_SHARE = 0.05
_LAST_RATE_SHARE = 0.1


@dataclass(frozen=True, slots=True)
class Dials:
    """The values a limited iteration reads, for one iteration.

    `player_rate`, `score_rate`, `builder_rate` and `drafter_rate` are learning
    rates, and a rate of zero freezes that model. `score_rate` is the rate of
    the learning deck score. The builder learns at `builder_rate` only once the
    search reads that score. `spread` is the player's pull toward even odds
    over its legal actions. `human_pull` is the weight of the drafter's pull
    toward the human-trained drafter. `drafter_entropy` and `builder_entropy`
    are the weights of the drafter's and the builder's entropy bonuses, which
    reward odds that stay spread out.
    """

    player_rate: float
    spread: float
    score_rate: float
    builder_rate: float
    drafter_rate: float
    human_pull: float
    drafter_entropy: float
    builder_entropy: float


# Takes an iteration, counted from 1, and returns that iteration's dials.
type Schedule = Callable[[int], Dials]
# Takes a step, counted from 1, and returns the factor that multiplies each optimizer's starting learning rate.
type RateScale = Callable[[int], float]


def power_decay(iteration: int, *, scale: float, power: float, low: float, high: float) -> float:
    # `iteration` counts from 1.
    value = scale / iteration**power
    return min(high, max(low, value))


def ramp(iteration: int, *, start: int, end: int, first: float, last: float) -> float:
    if iteration <= start:
        return first
    if iteration >= end:
        return last
    return first + (last - first) * (iteration - start) / (end - start)


def phased_schedule(*, drafter_starts: int, pull_ends: int) -> Schedule:
    """Build a schedule with soft phases, as a starting point to edit.

    The player's rate and spread follow Ataraxos's published curves from the
    first iteration. The deck models' rates are fixed. The builder starts
    learning when the loop's check switches the search to the learning deck
    score. That switch is the loop's decision, not a phase of the schedule.

    The drafter stays frozen through iteration `drafter_starts`, so that it
    learns only from the results of a player that has had time to learn. After
    that its dials change in straight lines instead of jumping, which is what
    makes the phases soft. Its rate ramps up over a tenth of the iterations it
    was frozen for, at least one iteration. Its pull toward the human drafter
    falls from one at `drafter_starts` to zero at `pull_ends`.

    The deck models' and the drafter's numbers are first guesses with no
    source.
    """

    def schedule(iteration: int) -> Dials:
        drafter_warm_end = drafter_starts + max(1, round(drafter_starts * _DRAFTER_WARM_UP))
        return Dials(
            player_rate=power_decay(iteration, scale=_RATE_SCALE, power=_RATE_POWER, low=_RATE_FLOOR, high=_RATE_CAP),
            spread=power_decay(iteration, scale=_SPREAD_START, power=_SPREAD_POWER, low=0.0, high=_SPREAD_START),
            score_rate=_SCORE_RATE,
            builder_rate=_BUILDER_RATE,
            drafter_rate=ramp(iteration, start=drafter_starts, end=drafter_warm_end, first=0.0, last=_DRAFTER_RATE),
            human_pull=ramp(iteration, start=drafter_starts, end=pull_ends, first=1.0, last=0.0),
            drafter_entropy=_DRAFTER_ENTROPY,
            builder_entropy=_BUILDER_ENTROPY,
        )

    return schedule


def warm_up_then_decay(steps: int) -> RateScale:
    """Build the rate scale the pretraining entry points use for a run of `steps` steps.

    The factor rises in a straight line from zero to one over the first
    twentieth of the steps, at least one step, and then falls in a straight
    line to a tenth at the last step. Multiply an optimizer's peak learning
    rate by it with `scale_rates`.
    """
    warm_up_steps = max(1, round(steps * _WARM_UP_SHARE))

    def rate_scale(step: int) -> float:
        if step <= warm_up_steps:
            return step / warm_up_steps
        return ramp(step, start=warm_up_steps, end=steps, first=1.0, last=_LAST_RATE_SHARE)

    return rate_scale


def current_rates(optimizers: Sequence[Optimizer]) -> tuple[float, ...]:
    # Every parameter group's learning rate, in the order that `scale_rates` expects its `starting_rates`.
    return tuple(float(group["lr"]) for optimizer in optimizers for group in optimizer.param_groups)


def scale_rates(optimizers: Sequence[Optimizer], starting_rates: Sequence[float], *, factor: float) -> None:
    groups = [group for optimizer in optimizers for group in optimizer.param_groups]
    for group, rate in zip(groups, starting_rates, strict=True):
        group["lr"] = rate * factor

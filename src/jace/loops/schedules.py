"""Dials that change over training, and small functions to schedule them.

A loop asks a schedule for the dials of each iteration, so how training
changes over time is one function a user can read and replace. A phase is a
stretch of iterations in which that function behaves one way, such as the
drafter's rate staying at zero until the player has had time to learn. The
shapes of the limited schedule follow Ataraxos, which lowers its learning
rate and its pull toward even odds as a power of the iteration number.
Pretraining on human data has one dial, a factor on the learning rate, and
`warm_up_then_decay` and `scale_rates` set it.
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

# Ataraxos's schedules (its supplement, Table S7). The learning rate is
# RATE_SCALE / iteration ** RATE_POWER kept within [RATE_FLOOR, RATE_CAP], and the pull
# toward even odds is SPREAD_START / iteration ** SPREAD_POWER.
_RATE_SCALE = 0.5
_RATE_POWER = 1.1
_RATE_FLOOR = 5e-6
_RATE_CAP = 1e-4
_SPREAD_START = 0.05
_SPREAD_POWER = 0.3

# First guesses with no source, to be replaced by measured values.
_DRAFTER_RATE = 1e-5
_DECK_RATE = 1e-4
_DRAFTER_ENTROPY = 0.01
_EXPLORE_SHARE = 0.1
# The drafter's rate ramps up over this fraction of the iterations it was frozen for.
_DRAFTER_WARM_UP = 0.1

# Pretraining on human data: the rate rises from zero over this share of the steps, then falls
# in a straight line to this share of its peak.
_WARM_UP_SHARE = 0.05
_LAST_RATE_SHARE = 0.1


@dataclass(frozen=True, slots=True)
class Dials:
    """The values a limited iteration reads, for one iteration.

    `player_rate`, `deck_rate` and `drafter_rate` are learning rates; a rate
    of zero freezes that model. `deck_rate` is the rate of both deck models.
    `spread` is the player's pull toward even odds over legal actions. `human_pull` weighs the drafter's pull toward the
    human-trained drafter, and `drafter_entropy` its bonus for keeping pick
    odds spread out. `explore_share` is the share of pools for which a deck
    other than the top-scored one is played.
    """

    player_rate: float
    spread: float
    deck_rate: float
    drafter_rate: float
    human_pull: float
    drafter_entropy: float
    explore_share: float


# iteration, counted from 1 -> that iteration's dials.
type Schedule = Callable[[int], Dials]
# step, counted from 1 -> the factor each optimizer's starting learning rate is multiplied by.
type RateScale = Callable[[int], float]


def power_decay(iteration: int, *, scale: float, power: float, low: float, high: float) -> float:
    # scale / iteration ** power, kept within [low, high]. iteration counts from 1.
    value = scale / iteration**power
    return min(high, max(low, value))


def ramp(iteration: int, *, start: int, end: int, first: float, last: float) -> float:
    # `first` up to `start`, `last` from `end` on, and a straight line between them.
    if iteration <= start:
        return first
    if iteration >= end:
        return last
    return first + (last - first) * (iteration - start) / (end - start)


def phased_schedule(*, decks_start: int, drafter_starts: int, pull_ends: int) -> Schedule:
    """Build a schedule with soft phases, as a starting point to edit.

    The player's rate and spread follow Ataraxos's published curves from the
    first iteration. The deck models stay frozen until `decks_start` and
    the drafter until `drafter_starts`, so the player's results mean
    something before they rank decks or steer picks. The deck models then
    learn at a fixed rate. Its rate then
    ramps up over a tenth of that span, and its pull toward the human
    drafter falls in a straight line to zero at `pull_ends`. The drafter's
    numbers and the explore share are first guesses with no source.
    """

    def schedule(iteration: int) -> Dials:
        drafter_warm_end = drafter_starts + max(1, round(drafter_starts * _DRAFTER_WARM_UP))
        return Dials(
            player_rate=power_decay(iteration, scale=_RATE_SCALE, power=_RATE_POWER, low=_RATE_FLOOR, high=_RATE_CAP),
            spread=power_decay(iteration, scale=_SPREAD_START, power=_SPREAD_POWER, low=0.0, high=_SPREAD_START),
            deck_rate=_DECK_RATE if iteration >= decks_start else 0.0,
            drafter_rate=ramp(iteration, start=drafter_starts, end=drafter_warm_end, first=0.0, last=_DRAFTER_RATE),
            human_pull=ramp(iteration, start=drafter_starts, end=pull_ends, first=1.0, last=0.0),
            drafter_entropy=_DRAFTER_ENTROPY,
            explore_share=_EXPLORE_SHARE,
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
    # The learning rate of every parameter group, in the order `scale_rates` takes them back.
    return tuple(float(group["lr"]) for optimizer in optimizers for group in optimizer.param_groups)


def scale_rates(optimizers: Sequence[Optimizer], starting_rates: Sequence[float], *, factor: float) -> None:
    # Sets every parameter group's learning rate to its entry of `starting_rates` times `factor`.
    groups = [group for optimizer in optimizers for group in optimizer.param_groups]
    for group, rate in zip(groups, starting_rates, strict=True):
        group["lr"] = rate * factor

"""Dials that change over training, and small functions to schedule them.

A loop asks a schedule for the dials of each iteration, so how training
changes over time is one function a user can read and replace. A phase is a
stretch of iterations in which that function behaves one way, such as the
drafter's rate staying at zero until the player has had time to learn. The
shapes here follow Ataraxos, which lowers its learning rate and its pull
toward even odds as a power of the iteration number.
"""

from collections.abc import Callable
from dataclasses import dataclass

__all__ = ("Dials", "Schedule", "phased_schedule", "power_decay", "ramp")

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
_DRAFTER_ENTROPY = 0.01
_EXPLORE_SHARE = 0.1
# The drafter's rate ramps up over this fraction of the iterations it was frozen for.
_DRAFTER_WARM_UP = 0.1


@dataclass(frozen=True, slots=True)
class Dials:
    """The values a limited iteration reads, for one iteration.

    `player_rate` and `drafter_rate` are learning rates; a rate of zero
    freezes that model. `spread` is the player's pull toward even odds over
    legal actions. `human_pull` weighs the drafter's pull toward the
    human-trained drafter, and `drafter_entropy` its bonus for keeping pick
    odds spread out. `explore_share` is the share of pools for which a deck
    other than the top-scored one is played.
    """

    player_rate: float
    spread: float
    drafter_rate: float
    human_pull: float
    drafter_entropy: float
    explore_share: float


# iteration, counted from 1 -> that iteration's dials.
type Schedule = Callable[[int], Dials]


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


def phased_schedule(*, drafter_starts: int, pull_ends: int) -> Schedule:
    """Build a schedule with soft phases, as a starting point to edit.

    The player's rate and spread follow Ataraxos's published curves from the
    first iteration. The drafter stays frozen until `drafter_starts`, so the
    player's results mean something before they steer picks. Its rate then
    ramps up over a tenth of that span, and its pull toward the human
    drafter falls in a straight line to zero at `pull_ends`. The drafter's
    numbers and the explore share are first guesses with no source.
    """

    def schedule(iteration: int) -> Dials:
        warm = drafter_starts + max(1, round(drafter_starts * _DRAFTER_WARM_UP))
        return Dials(
            player_rate=power_decay(iteration, scale=_RATE_SCALE, power=_RATE_POWER, low=_RATE_FLOOR, high=_RATE_CAP),
            spread=power_decay(iteration, scale=_SPREAD_START, power=_SPREAD_POWER, low=0.0, high=_SPREAD_START),
            drafter_rate=ramp(iteration, start=drafter_starts, end=warm, first=0.0, last=_DRAFTER_RATE),
            human_pull=ramp(iteration, start=drafter_starts, end=pull_ends, first=1.0, last=0.0),
            drafter_entropy=_DRAFTER_ENTROPY,
            explore_share=_EXPLORE_SHARE,
        )

    return schedule

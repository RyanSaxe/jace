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
        warm = drafter_starts + max(1, drafter_starts // 10)
        return Dials(
            player_rate=power_decay(iteration, scale=0.5, power=1.1, low=5e-6, high=1e-4),
            spread=power_decay(iteration, scale=0.05, power=0.3, low=0.0, high=0.05),
            drafter_rate=ramp(iteration, start=drafter_starts, end=warm, first=0.0, last=1e-5),
            human_pull=ramp(iteration, start=drafter_starts, end=pull_ends, first=1.0, last=0.0),
            drafter_entropy=0.01,
            explore_share=0.1,
        )

    return schedule

"""Example training loops and the schedules that drive them.

A loop here is a short function assembled from the pieces in the other
modules, and this is the only package that may import `jace.draft`. The
limited loop shows the whole system working together and how dials are
scheduled; it is meant to be read, copied and changed.
"""

from jace.loops.limited import Learners, LimitedSetup, StartBatch, limited_iteration
from jace.loops.schedules import (
    Dials,
    RateScale,
    Schedule,
    current_rates,
    phased_schedule,
    power_decay,
    ramp,
    scale_rates,
    warm_up_then_decay,
)

__all__ = (
    "Dials",
    "Learners",
    "LimitedSetup",
    "RateScale",
    "Schedule",
    "StartBatch",
    "current_rates",
    "limited_iteration",
    "phased_schedule",
    "power_decay",
    "ramp",
    "scale_rates",
    "warm_up_then_decay",
)

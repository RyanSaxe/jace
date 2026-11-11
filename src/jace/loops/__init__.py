"""Example training loops and the schedules that drive them.

A loop here is a short function assembled from the pieces in the other
modules, and this is the only package that may import `jace.draft`. The
limited loop shows the whole system working together and how dials are
scheduled; it is meant to be read, copied and changed.
"""

from jace.loops.limited import Learners, LimitedSetup, StartBatch, limited_iteration
from jace.loops.schedules import Dials, Schedule, phased_schedule, power_decay, ramp

__all__ = (
    "Dials",
    "Learners",
    "LimitedSetup",
    "Schedule",
    "StartBatch",
    "limited_iteration",
    "phased_schedule",
    "power_decay",
    "ramp",
)

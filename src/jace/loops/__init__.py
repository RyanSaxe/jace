"""Example training loops and the schedules that drive them.

A loop here is a short function assembled from the pieces in the other
modules, and this is the only package that may import `jace.draft`. The
limited loop shows the whole system working together and how dials are
scheduled; it is meant to be read, copied and changed. Each limited
format's sizes are one `LimitedFormat` record, which every entry point
looks up by set code.
"""

from jace.loops.formats import TLA_LIMITED, LimitedFormat, UnknownFormatError, deck_rules, limited_format
from jace.loops.limited import Learners, LimitedSetup, UnevenDecksError, balanced_matchups, limited_iteration
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
    "TLA_LIMITED",
    "Dials",
    "Learners",
    "LimitedFormat",
    "LimitedSetup",
    "RateScale",
    "Schedule",
    "UnevenDecksError",
    "UnknownFormatError",
    "balanced_matchups",
    "current_rates",
    "deck_rules",
    "limited_format",
    "limited_iteration",
    "phased_schedule",
    "power_decay",
    "ramp",
    "scale_rates",
    "warm_up_then_decay",
)

"""Top-level training and pretraining loops wired from jace's modular packages.

Each loop connects data loading, Limited format rules, model pretraining, or
RL self-play into a runnable script. Lower-level packages never import
`jace.loops`, so reusable RL and Limited code stays decoupled from CLI wiring.
"""

from jace.loops.formats import TLA_LIMITED, LimitedFormat, UnknownFormatError, deck_rules, limited_format
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
    "LimitedFormat",
    "RateScale",
    "Schedule",
    "UnknownFormatError",
    "current_rates",
    "deck_rules",
    "limited_format",
    "phased_schedule",
    "power_decay",
    "ramp",
    "scale_rates",
    "warm_up_then_decay",
)

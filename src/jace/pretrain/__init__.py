"""Supervised pretraining for Limited Magic models on 17Lands human data.

Provides format geometry (`formats`), learning-rate schedules (`schedules`), supervised
loss helpers (`supervised`), deck evaluation measures (`deck_measures`), shared CLI
helpers (`common`), and the two runnable entry points (`draft` and `decks`). Keeps
supervised pretraining decoupled from `jace.rl`, `jace.eval`, and `manabot`.
"""

from jace.pretrain.formats import TLA_LIMITED, LimitedFormat, UnknownFormatError, deck_rules, limited_format
from jace.pretrain.schedules import (
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

"""Example training loops and the schedules that drive them.

A loop here is a short function built from the pieces in jace's other
packages. This is the only package that may import `jace.draft`, so that
constructed play needs no drafting code. The limited loop shows the whole
system working together. It also shows how a schedule sets the dials, the
values such as learning rates that change from one iteration to the next.
The loop is meant to be read, copied and changed.

The sizes of each limited format are one `LimitedFormat` record, which
every training entry point looks up by set code. The limited loop passes a
`ScoreCheck` from one iteration to the next. The check contains the recent
test losses of the two deck scores, and whether the deck search has
switched from the human-trained score to the one that learns from
self-play.
"""

from jace.loops.checked_score import (
    FIRST_CHECK,
    PlayedDecks,
    ScoreCheck,
    ScoreRule,
    check_scores,
    train_learning_score,
)
from jace.loops.formats import TLA_LIMITED, LimitedFormat, UnknownFormatError, deck_rules, limited_format
from jace.loops.limited import (
    Learners,
    LimitedSetup,
    UnevenDecksError,
    balanced_matchups,
    block_builds,
    limited_iteration,
)
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
    "FIRST_CHECK",
    "TLA_LIMITED",
    "Dials",
    "Learners",
    "LimitedFormat",
    "LimitedSetup",
    "PlayedDecks",
    "RateScale",
    "Schedule",
    "ScoreCheck",
    "ScoreRule",
    "UnevenDecksError",
    "UnknownFormatError",
    "balanced_matchups",
    "block_builds",
    "check_scores",
    "current_rates",
    "deck_rules",
    "limited_format",
    "limited_iteration",
    "phased_schedule",
    "power_decay",
    "ramp",
    "scale_rates",
    "train_learning_score",
    "warm_up_then_decay",
)

"""Simulate booster drafts and pick cards in them.

The `pod` module defines the rules of a draft table, and nothing about
which card a seat should take. Three other modules open boosters, run many
pods at once and define the drafter model. They use a pod only through its
states and its one transition, `advance`.
"""

from jace.draft.boosters import (
    TLA_PLAY_BOOSTER,
    BoosterSpec,
    BoosterSpecError,
    SlotOdds,
    copies_per_booster,
    open_booster,
)
from jace.draft.drafting import Drafter, DraftRecord, SeatView, run_drafts
from jace.draft.model import DrafterModel, model_drafter, view_tensors
from jace.draft.pod import (
    Complete,
    DraftFormat,
    IllegalPickError,
    InvalidDraftError,
    Pack,
    Picking,
    PodState,
    advance,
    open_pod,
)

__all__ = (
    "TLA_PLAY_BOOSTER",
    "BoosterSpec",
    "BoosterSpecError",
    "Complete",
    "DraftFormat",
    "DraftRecord",
    "Drafter",
    "DrafterModel",
    "IllegalPickError",
    "InvalidDraftError",
    "Pack",
    "Picking",
    "PodState",
    "SeatView",
    "SlotOdds",
    "advance",
    "copies_per_booster",
    "model_drafter",
    "open_booster",
    "open_pod",
    "run_drafts",
    "view_tensors",
)

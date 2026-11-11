"""Simulate booster drafts and pick cards in them.

The pod owns the rules of a draft table and nothing about which card a seat
should take. Boosters, the loop that runs many pods at once, and the drafter
model are separate modules that talk to a pod only through its states and
its one transition.
"""

from jace.draft.boosters import TLA_PLAY_BOOSTER, BoosterSpec, BoosterSpecError, SlotOdds, open_booster
from jace.draft.drafting import Drafter, DraftRecord, SeatView, run_drafts
from jace.draft.model import DrafterModel, model_drafter
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
    "model_drafter",
    "open_booster",
    "open_pod",
    "run_drafts",
)

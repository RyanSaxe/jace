"""Pure-tensor opponent belief inference, format deck priors, and pre-game `V_0` predictors.

This package operates exclusively on PyTorch tensors and `jace.cards.CardVectors`
with no engine (`managym`/`manabot`) dependencies.
"""

from jace.belief.model import (
    GAMEPLAY_SIGNAL_DIM,
    BeliefNet,
    belief_loss,
    evaluate_belief_log_loss,
    infer_belief_state,
    load_belief_net,
    project_belief_overrides,
    save_belief_net,
    train_belief_net,
)
from jace.belief.prior import (
    BayesianDeckPoolPrior,
    PriorBackend,
    SetTransformerPrior,
    fit_format_prior,
    load_format_prior,
    save_format_prior,
)
from jace.belief.types import BeliefOverride, BeliefState, FormatPrior, PriorBackendName
from jace.belief.v0 import MatchupPredictor, evaluate_opening_hand_vs_v0, score_deck_vs_field

__all__ = (
    "GAMEPLAY_SIGNAL_DIM",
    "BayesianDeckPoolPrior",
    "BeliefNet",
    "BeliefOverride",
    "BeliefState",
    "FormatPrior",
    "MatchupPredictor",
    "PriorBackend",
    "PriorBackendName",
    "SetTransformerPrior",
    "belief_loss",
    "evaluate_belief_log_loss",
    "evaluate_opening_hand_vs_v0",
    "fit_format_prior",
    "infer_belief_state",
    "load_belief_net",
    "load_format_prior",
    "project_belief_overrides",
    "save_belief_net",
    "save_format_prior",
    "score_deck_vs_field",
    "train_belief_net",
)

"""Self-play reinforcement learning algorithms and training loop for Jace v2.

Separates pure tensor RL math (`credit`, `select`, `losses`) from the `manabot`
collection bridge (`loop`) so credit assignment, row filtering, and policy/value losses
can be composed, swapped, and unit-tested without starting native Magic games. Upstream
`manabot` learner updates and recipes remain strictly banned by pre-commit AST rules.
"""

from jace.rl.credit import CreditFn, CreditTargets, Trajectory, gae_wdl, monte_carlo, standardize_advantages
from jace.rl.loop import (
    StepRecord,
    TrainConfig,
    UpdateCallback,
    UpdateConfig,
    UpdateStats,
    train_self_play,
    update_step,
)
from jace.rl.losses import (
    LossBreakdown,
    LossFn,
    PolicyBatch,
    categorical_value_loss,
    damped_policy_loss,
    policy_log_probs,
    ppo_loss,
    reverse_kl,
)
from jace.rl.select import SelectFn, all_rows, top_fraction

__all__ = (
    "CreditFn",
    "CreditTargets",
    "LossBreakdown",
    "LossFn",
    "PolicyBatch",
    "SelectFn",
    "StepRecord",
    "TrainConfig",
    "Trajectory",
    "UpdateCallback",
    "UpdateConfig",
    "UpdateStats",
    "all_rows",
    "categorical_value_loss",
    "damped_policy_loss",
    "gae_wdl",
    "monte_carlo",
    "policy_log_probs",
    "ppo_loss",
    "reverse_kl",
    "standardize_advantages",
    "top_fraction",
    "train_self_play",
    "update_step",
)

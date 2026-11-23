"""Self-play reinforcement learning algorithms and training loop for Jace v2.

Separates pure tensor RL math (`credit`, `select`, `losses`, `pool`) from the `manabot`
collection bridge (`loop`) so credit assignment, row filtering, schedules, snapshot pools,
and policy/value losses can be composed, swapped, and unit-tested without starting native
Magic games. Upstream `manabot` learner updates and recipes remain banned by AST rules.
"""

from jace.rl.credit import CreditFn, CreditTargets, Trajectory, gae_wdl, monte_carlo, standardize_advantages
from jace.rl.diagnostics import GradientAgreementTracker, snapshot_win_rate
from jace.rl.loop import (
    CollectFn,
    LossScheduleFn,
    StepRecord,
    TrainConfig,
    UpdateCallback,
    UpdateConfig,
    UpdateStats,
    train_self_play,
    update_step,
)
from jace.rl.losses import (
    ForkLossBreakdown,
    LossBreakdown,
    LossFn,
    PolicyBatch,
    categorical_value_loss,
    damped_policy_loss,
    fork_pairwise_loss,
    fork_targets_from_returns,
    policy_log_probs,
    ppo_loss,
    reverse_kl,
)
from jace.rl.pool import DampedSchedule, DecaySchedule, SnapshotEntry, SnapshotPool, load_snapshot
from jace.rl.select import SelectFn, all_rows, bucketed_minibatch_order, top_fraction

__all__ = (
    "CollectFn",
    "CreditFn",
    "CreditTargets",
    "DampedSchedule",
    "DecaySchedule",
    "ForkLossBreakdown",
    "GradientAgreementTracker",
    "LossBreakdown",
    "LossFn",
    "LossScheduleFn",
    "PolicyBatch",
    "SelectFn",
    "SnapshotEntry",
    "SnapshotPool",
    "StepRecord",
    "TrainConfig",
    "Trajectory",
    "UpdateCallback",
    "UpdateConfig",
    "UpdateStats",
    "all_rows",
    "bucketed_minibatch_order",
    "categorical_value_loss",
    "damped_policy_loss",
    "fork_pairwise_loss",
    "fork_targets_from_returns",
    "gae_wdl",
    "load_snapshot",
    "monte_carlo",
    "policy_log_probs",
    "ppo_loss",
    "reverse_kl",
    "snapshot_win_rate",
    "standardize_advantages",
    "top_fraction",
    "train_self_play",
    "update_step",
)

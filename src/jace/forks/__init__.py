"""Parallel counterfactual fork collection, batched playouts, and horizon auditing.

Clones live `managym.Env` game states across spawned CPU workers to compare the
policy's top action against its runner-up under shared random streams (`collect`
and `batch`). `audit` stages pairwise fork and continuation WDL auxiliary losses
for `train_self_play` alongside paired horizon-correlation audits so truncated
rollouts are validated against full-game outcomes before training.
"""

from jace.forks.audit import (
    BatchedForkSpec,
    HorizonAuditSummary,
    add_fork_cli_args,
    audit_horizon_correlation,
    build_batched_fork_aux_loss,
    make_batched_fork_prepare_aux,
    open_batched_fork_hook,
    parse_fork_kinds,
)
from jace.forks.batch import (
    BatchedForkCollectorPool,
    BatchedForkConfig,
    BatchedForkResult,
    ContinuationValueSample,
)
from jace.forks.collect import ForkCollectorPool, ForkConfig, ForkExample, ForkStats, build_fork_aux_loss

__all__ = (
    "BatchedForkCollectorPool",
    "BatchedForkConfig",
    "BatchedForkResult",
    "BatchedForkSpec",
    "ContinuationValueSample",
    "ForkCollectorPool",
    "ForkConfig",
    "ForkExample",
    "ForkStats",
    "HorizonAuditSummary",
    "add_fork_cli_args",
    "audit_horizon_correlation",
    "build_batched_fork_aux_loss",
    "build_fork_aux_loss",
    "make_batched_fork_prepare_aux",
    "open_batched_fork_hook",
    "parse_fork_kinds",
)

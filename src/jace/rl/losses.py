"""Policy and critic loss functions over plain tensors for self-play RL.

Implements both the damped policy objective (clipped surrogate plus reverse KL to the
frozen behavior policy and reverse KL to a full-support magnet) and standard clipped
PPO with an entropy bonus. All functions accept a plain `PolicyBatch` of tensors with
explicit legal masks so they run identically on CPU and MPS without importing `manabot`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = (
    "ForkLossBreakdown",
    "LossBreakdown",
    "LossFn",
    "PolicyBatch",
    "categorical_value_loss",
    "damped_policy_loss",
    "fork_pairwise_loss",
    "fork_targets_from_returns",
    "policy_log_probs",
    "ppo_loss",
    "reverse_kl",
)

_MATRIX_NDIM = 2
_PAIR_WIDTH = 2
_WDL_CLASSES = 3


@dataclass(frozen=True, slots=True)
class PolicyBatch:
    logits: Tensor
    value_output: Tensor
    actions: Tensor
    advantages: Tensor
    value_targets: Tensor
    behavior_logs: Tensor
    reference_probs: Tensor
    legal_mask: Tensor
    policy_mask: Tensor | None = None


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    policy: float
    value: float
    collection_kl: float
    magnet_kl: float
    entropy: float
    clip_fraction: float


@dataclass(frozen=True, slots=True)
class ForkLossBreakdown:
    total: Tensor
    cross_entropy: float
    anchor_kl: float
    active_count: int
    mean_gap: float


type LossFn = Callable[[PolicyBatch], LossBreakdown]


def _host_double(tensor: Tensor) -> Tensor:
    # MPS lacks float64; moving the small scalar/per-row surrogate operand to CPU preserves autograd.
    return tensor.cpu().double() if tensor.device.type == "mps" else tensor.double()


def policy_log_probs(logits: Tensor, legal_mask: Tensor) -> Tensor:
    valid = legal_mask.bool()
    if not bool(valid.any(dim=-1).all()):
        msg = "every row must have at least one legal action"
        raise ValueError(msg)
    return logits.masked_fill(~valid, -torch.inf).log_softmax(dim=-1)


def reverse_kl(current_logs: Tensor, target_logs: Tensor, *, legal_mask: Tensor) -> Tensor:
    valid = legal_mask.bool()
    diff = current_logs.masked_fill(~valid, 0.0) - target_logs.detach().masked_fill(~valid, 0.0)
    return (current_logs.exp() * diff).sum(dim=-1)


def _policy_entropy(logs: Tensor, legal_mask: Tensor) -> Tensor:
    valid = legal_mask.bool()
    return -(logs.exp() * logs.masked_fill(~valid, 0.0)).sum(dim=-1)


def categorical_value_loss(value_logits: Tensor, targets: Tensor) -> Tensor:
    if value_logits.ndim != _MATRIX_NDIM or value_logits.shape[-1] != _WDL_CLASSES:
        msg = f"expected [B, 3] WDL value logits, got {tuple(value_logits.shape)}"
        raise ValueError(msg)
    return -(targets.detach() * value_logits.log_softmax(dim=-1)).sum(dim=-1)


def _value_loss_rows(value_output: Tensor, value_targets: Tensor) -> Tensor:
    if value_targets.ndim == _MATRIX_NDIM and value_targets.shape[-1] == _WDL_CLASSES:
        return categorical_value_loss(value_output, value_targets)
    predicted = value_output.squeeze(-1) if value_output.ndim == _MATRIX_NDIM else value_output
    return (predicted - value_targets.detach()).square()


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    if mask is None:
        return values.mean()
    keep = mask.bool().to(values.device)
    denom = keep.sum().clamp_min(1)
    return values[keep].sum() / denom


def _surrogate_and_clip(
    current_logs: Tensor,
    batch: PolicyBatch,
    *,
    clip: float,
) -> tuple[Tensor, Tensor]:
    chosen = batch.actions[:, None]
    new_action_logs = current_logs.gather(-1, chosen).squeeze(-1)
    old_action_logs = batch.behavior_logs.detach().gather(-1, chosen).squeeze(-1)
    log_ratio = _host_double(new_action_logs - old_action_logs)
    ratio = log_ratio.exp()
    adv = _host_double(batch.advantages.detach())
    surrogate = -torch.minimum(ratio * adv, ratio.clamp(1.0 - clip, 1.0 + clip) * adv)
    clipped = ((ratio.detach() - 1.0).abs() > clip).float()
    return surrogate, clipped


def damped_policy_loss(
    *,
    clip: float = 0.2,
    collection_kl: float = 0.1,
    magnet_kl: float = 0.05,
    value_coef: float = 1.0,
) -> LossFn:
    """Build the damped policy-gradient loss (clipped surrogate + dual reverse KL + critic).

    Combines the float64-ratio clipped surrogate with a reverse KL anchor to the
    collection behavior distribution (`collection_kl`) and a reverse KL pull toward
    `batch.reference_probs` (`magnet_kl`), plus `value_coef` times categorical WDL
    cross-entropy (or scalar MSE). When `batch.policy_mask` is set, policy terms
    average over selected rows while the critic still trains on every row.
    """
    if not (0.0 < clip < 1.0) or collection_kl < 0.0 or magnet_kl < 0.0 or value_coef < 0.0:
        msg = "clip must lie in (0, 1) and loss coefficients must be non-negative"
        raise ValueError(msg)

    def compute(batch: PolicyBatch) -> LossBreakdown:
        valid = batch.legal_mask.bool()
        logs = policy_log_probs(batch.logits, valid)
        surrogate, clipped = _surrogate_and_clip(logs, batch, clip=clip)
        beh_kl = reverse_kl(logs, batch.behavior_logs, legal_mask=valid)
        ref_logs = batch.reference_probs.detach().masked_fill(~valid, 1.0).log()
        mag_kl = reverse_kl(logs, ref_logs, legal_mask=valid)
        ent = _policy_entropy(logs, valid)

        policy_rows = surrogate + collection_kl * _host_double(beh_kl) + magnet_kl * _host_double(mag_kl)
        policy_term = _masked_mean(policy_rows, batch.policy_mask)
        value_term = _value_loss_rows(batch.value_output, batch.value_targets).mean()
        total = policy_term + value_coef * _host_double(value_term)
        return LossBreakdown(
            total=total,
            policy=float(_masked_mean(surrogate.detach(), batch.policy_mask).item()),
            value=float(value_term.detach().item()),
            collection_kl=float(_masked_mean(beh_kl.detach(), batch.policy_mask).item()),
            magnet_kl=float(_masked_mean(mag_kl.detach(), batch.policy_mask).item()),
            entropy=float(_masked_mean(ent.detach(), batch.policy_mask).item()),
            clip_fraction=float(_masked_mean(clipped, batch.policy_mask).item()),
        )

    return compute


def ppo_loss(
    *,
    clip: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
) -> LossFn:
    if not (0.0 < clip < 1.0) or entropy_coef < 0.0 or value_coef < 0.0:
        msg = "clip must lie in (0, 1) and loss coefficients must be non-negative"
        raise ValueError(msg)

    def compute(batch: PolicyBatch) -> LossBreakdown:
        valid = batch.legal_mask.bool()
        logs = policy_log_probs(batch.logits, valid)
        surrogate, clipped = _surrogate_and_clip(logs, batch, clip=clip)
        beh_kl = reverse_kl(logs, batch.behavior_logs, legal_mask=valid)
        ent = _policy_entropy(logs, valid)

        policy_rows = surrogate - entropy_coef * _host_double(ent)
        policy_term = _masked_mean(policy_rows, batch.policy_mask)
        value_term = _value_loss_rows(batch.value_output, batch.value_targets).mean()
        total = policy_term + value_coef * _host_double(value_term)
        return LossBreakdown(
            total=total,
            policy=float(_masked_mean(surrogate.detach(), batch.policy_mask).item()),
            value=float(value_term.detach().item()),
            collection_kl=float(_masked_mean(beh_kl.detach(), batch.policy_mask).item()),
            magnet_kl=0.0,
            entropy=float(_masked_mean(ent.detach(), batch.policy_mask).item()),
            clip_fraction=float(_masked_mean(clipped, batch.policy_mask).item()),
        )

    return compute


def fork_targets_from_returns(
    pair_returns: Tensor,
    *,
    temperature: float = 0.25,
    min_gap: float = 0.25,
) -> tuple[Tensor, Tensor]:
    """Turn paired counterfactual rollout returns `[B, 2]` into regularized targets and weights.

    Follows regularized policy optimization for small simulation budgets (Grill et al.,
    2020): applies `softmax(pair_returns / temperature)` over the two compared actions
    and assigns weight equal to `|q_0 - q_1|` when the gap reaches `min_gap`, zeroing out
    tied or noisy forks so they contribute zero gradient.
    """
    if pair_returns.ndim != _MATRIX_NDIM or pair_returns.shape[-1] != _PAIR_WIDTH:
        msg = f"expected [B, 2] pair_returns, got {tuple(pair_returns.shape)}"
        raise ValueError(msg)
    if temperature <= 0.0 or min_gap < 0.0:
        msg = "temperature must be positive and min_gap must be non-negative"
        raise ValueError(msg)
    gap = (pair_returns[:, 0] - pair_returns[:, 1]).abs()
    targets = (pair_returns.detach() / temperature).softmax(dim=-1)
    weights = torch.where(gap >= min_gap, gap, torch.zeros_like(gap))
    return targets, weights


def fork_pairwise_loss(
    logits: Tensor,
    *,
    action_pairs: Tensor,
    pair_returns: Tensor,
    legal_mask: Tensor,
    reference_logs: Tensor | None = None,
    temperature: float = 0.25,
    min_gap: float = 0.25,
    anchor_kl: float = 0.1,
) -> ForkLossBreakdown:
    """Compute bounded pairwise cross-entropy loss over counterfactual fork comparisons.

    Restricts the softmax to the two compared actions `action_pairs` (`[B, 2]`), which
    equals the full action space for binary decisions like `declare_attacker` and avoids
    distorting un-forked actions on wider spaces. Because cross-entropy has gradient
    `p - target`, it stays full-strength even when a collapsed policy assigns `p = 0.999`
    to the losing action. Optionally adds `anchor_kl` times full-action reverse KL to
    `reference_logs` to protect unrelated decisions sharing trunk features.
    """
    if action_pairs.ndim != _MATRIX_NDIM or action_pairs.shape[-1] != _PAIR_WIDTH:
        msg = f"expected [B, 2] action_pairs, got {tuple(action_pairs.shape)}"
        raise ValueError(msg)
    targets, weights = fork_targets_from_returns(pair_returns, temperature=temperature, min_gap=min_gap)
    pair_logits = logits.gather(-1, action_pairs.long())
    pair_logs = pair_logits.log_softmax(dim=-1)
    ce_rows = -(targets.detach() * pair_logs).sum(dim=-1)
    active_mask = weights > 0.0
    active_count = int(active_mask.sum().item())
    if active_count == 0:
        zero = logits.sum() * 0.0
        return ForkLossBreakdown(total=zero, cross_entropy=0.0, anchor_kl=0.0, active_count=0, mean_gap=0.0)
    denom = weights.sum().clamp_min(1e-6)
    ce_loss = (ce_rows * weights.detach()).sum() / denom
    kl_loss = logits.sum() * 0.0
    if reference_logs is not None and anchor_kl > 0.0:
        full_logs = policy_log_probs(logits, legal_mask)
        kl_rows = reverse_kl(full_logs, reference_logs, legal_mask=legal_mask)
        kl_loss = kl_rows[active_mask].mean()
    total = ce_loss + anchor_kl * kl_loss
    return ForkLossBreakdown(
        total=total,
        cross_entropy=float(ce_loss.detach().item()),
        anchor_kl=float(kl_loss.detach().item()),
        active_count=active_count,
        mean_gap=float(weights[active_mask].mean().item()),
    )

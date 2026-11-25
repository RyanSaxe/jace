"""Fixed held-out probe-state decision diffs across checkpoints.

Captures a deterministic bank of real Magic decision states (priority, combat, targeting)
from heuristic games and compares two `ActorCritic` checkpoints on those exact states.
Reports total-variation distance, KL divergence, argmax flip rate, and the top-K largest
policy shifts so changes in play style are visible even when headline win rates are close.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import managym
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from managym.enums import ActionEnum, ActionSpaceEnum

from jace.eval.deals import Deal, PolicyPlayer

__all__ = (
    "DecisionShift",
    "DiffSummary",
    "ProbeState",
    "capture_probes",
    "diff_checkpoints",
)

_ACTION_SPACE_NAMES = {int(member): member.name.lower() for member in ActionSpaceEnum}
_MIN_PROB = 1e-12
_PROBE_KINDS = frozenset(
    {
        int(ActionSpaceEnum.PRIORITY),
        int(ActionSpaceEnum.DECLARE_ATTACKER),
        int(ActionSpaceEnum.DECLARE_BLOCKER),
        int(ActionSpaceEnum.CHOOSE_TARGET),
    }
)


@dataclass(frozen=True, slots=True)
class ProbeState:
    deal: Deal
    prefix_actions: tuple[int, ...]
    action_space_kind: str
    turn: int
    action_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecisionShift:
    probe_index: int
    action_space_kind: str
    turn: int
    tv_distance: float
    kl_divergence: float
    argmax_changed: bool
    before_top_action: str
    before_top_prob: float
    after_top_action: str
    after_top_prob: float
    value_delta: float

    def to_dict(self) -> dict[str, object]:
        return {
            "probe_index": self.probe_index,
            "action_space_kind": self.action_space_kind,
            "turn": self.turn,
            "tv_distance": round(self.tv_distance, 4),
            "kl_divergence": round(self.kl_divergence, 4),
            "argmax_changed": self.argmax_changed,
            "before_top_action": self.before_top_action,
            "before_top_prob": round(self.before_top_prob, 4),
            "after_top_action": self.after_top_action,
            "after_top_prob": round(self.after_top_prob, 4),
            "value_delta": round(self.value_delta, 4),
        }


@dataclass(frozen=True, slots=True)
class DiffSummary:
    probes: int
    mean_tv: float
    mean_kl: float
    argmax_flip_rate: float
    mean_abs_value_delta: float
    top_shifts: tuple[DecisionShift, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "probes": self.probes,
            "mean_tv": round(self.mean_tv, 4),
            "mean_kl": round(self.mean_kl, 4),
            "argmax_flip_rate": round(self.argmax_flip_rate, 4),
            "mean_abs_value_delta": round(self.mean_abs_value_delta, 4),
            "top_shifts": [shift.to_dict() for shift in self.top_shifts],
        }


def _action_label(action: managym.Action) -> str:
    if action.description is not None and action.description != "":
        return str(action.description)
    kind = ActionEnum(int(action.action_type)).name.lower()
    if int(action.action_type) in (int(ActionEnum.DECLARE_ATTACKER), int(ActionEnum.DECLARE_BLOCKER)):
        return f"{kind}:{'yes' if action.declared is True else 'no'}"
    return kind


def _replay_probe(probe: ProbeState, *, decision_memory: bool = False) -> managym.Observation:
    env = managym.Env(seed=probe.deal.seed, skip_trivial=True, decision_memory=decision_memory)
    obs, _ = env.reset(probe.deal.game_setup(swapped=False))
    for action in probe.prefix_actions:
        obs, _, _, _, _ = env.step(action)
    return obs


def capture_probes(
    deals: Sequence[Deal],
    *,
    max_probes: int = 64,
    per_deal_cap: int = 8,
    max_actions: int = 400,
) -> list[ProbeState]:
    heuristic = managym.Heuristic()
    probes: list[ProbeState] = []
    for deal in deals:
        if len(probes) >= max_probes:
            break
        env = managym.Env(seed=deal.seed, skip_trivial=True)
        obs, _ = env.reset(deal.game_setup(swapped=False))
        prefix: list[int] = []
        captured = 0
        terminal = False
        truncated = False
        while not terminal and not truncated and len(prefix) < max_actions and captured < per_deal_cap:
            space_type = int(obs.action_space.action_space_type)
            actions = obs.action_space.actions
            if space_type in _PROBE_KINDS and len(actions) > 1 and (len(prefix) % 3 == 0 or space_type != 1):
                probes.append(
                    ProbeState(
                        deal=deal,
                        prefix_actions=tuple(prefix),
                        action_space_kind=_ACTION_SPACE_NAMES.get(space_type, f"kind_{space_type}"),
                        turn=int(obs.turn.turn_number),
                        action_labels=tuple(_action_label(a) for a in actions),
                    )
                )
                captured += 1
                if len(probes) >= max_probes:
                    break
            chosen = int(heuristic.select_action(obs))
            prefix.append(chosen)
            obs, _, terminal, truncated, _ = env.step(chosen)
    return probes


def _compare_distributions(
    probe_index: int,
    probe: ProbeState,
    *,
    before_probs: Sequence[float],
    before_value: float,
    after_probs: Sequence[float],
    after_value: float,
) -> DecisionShift:
    tv = 0.5 * sum(abs(p - q) for p, q in zip(before_probs, after_probs, strict=True))
    kl = sum(
        q * math.log(max(q, _MIN_PROB) / max(p, _MIN_PROB))
        for p, q in zip(before_probs, after_probs, strict=True)
        if q > 0.0
    )
    before_idx = max(range(len(before_probs)), key=before_probs.__getitem__)
    after_idx = max(range(len(after_probs)), key=after_probs.__getitem__)
    return DecisionShift(
        probe_index=probe_index,
        action_space_kind=probe.action_space_kind,
        turn=probe.turn,
        tv_distance=tv,
        kl_divergence=kl,
        argmax_changed=before_idx != after_idx,
        before_top_action=probe.action_labels[before_idx],
        before_top_prob=before_probs[before_idx],
        after_top_action=probe.action_labels[after_idx],
        after_top_prob=after_probs[after_idx],
        value_delta=after_value - before_value,
    )


def diff_checkpoints(
    before: ActorCritic,
    after: ActorCritic,
    *,
    obs_space: ObservationSpace,
    probes: Sequence[ProbeState],
    top_k: int = 10,
) -> DiffSummary:
    """Compare two `ActorCritic` checkpoints on the exact same bank of probe states.

    Replays each probe deterministically once, evaluates both policies under cached
    inference scopes, and returns aggregate policy/value shift statistics plus the
    `top_k` largest total-variation shifts.
    """
    shifts: list[DecisionShift] = []
    use_memory = obs_space.config.max_decision_records > 0
    with (
        PolicyPlayer(before, obs_space, deterministic=True) as player_before,
        PolicyPlayer(after, obs_space, deterministic=True) as player_after,
    ):
        for index, probe in enumerate(probes):
            obs = _replay_probe(probe, decision_memory=use_memory)
            _, before_probs, before_val = player_before.evaluate(obs)
            _, after_probs, after_val = player_after.evaluate(obs)
            shifts.append(
                _compare_distributions(
                    index,
                    probe,
                    before_probs=before_probs,
                    before_value=before_val,
                    after_probs=after_probs,
                    after_value=after_val,
                )
            )
    count = len(shifts)
    if count == 0:
        return DiffSummary(0, 0.0, 0.0, 0.0, 0.0, ())
    ranked = sorted(shifts, key=lambda item: item.tv_distance, reverse=True)
    return DiffSummary(
        probes=count,
        mean_tv=sum(s.tv_distance for s in shifts) / count,
        mean_kl=sum(s.kl_divergence for s in shifts) / count,
        argmax_flip_rate=sum(int(s.argmax_changed) for s in shifts) / count,
        mean_abs_value_delta=sum(abs(s.value_delta) for s in shifts) / count,
        top_shifts=tuple(ranked[:top_k]),
    )

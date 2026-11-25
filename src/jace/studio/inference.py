"""Checkpoint inference, policy diffing, and counterfactual Fork execution for Jace Studio.

Loads saved `.safetensors` checkpoints through an LRU cache, evaluates real legal
action probabilities and win-draw-loss estimates on any `managym.Observation`,
and runs stateless counterfactual forks via `env.clone_env()` and `scenario_*`.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

from manabot.backends.torch.bot import evaluate_observation
from manabot.backends.torch.checkpoint import load_checkpoint

from jace.studio.runs import resolve_run_dir

if TYPE_CHECKING:
    from pathlib import Path

    import managym
    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace

__all__ = (
    "ForkEdit",
    "InferenceEngine",
    "apply_fork_edits",
    "compute_policy_diff",
)

_MAX_LOADED_CKPTS = 4
_MIN_PROB = 1e-12
_DRAW_FLOOR = 0.02
_DRAW_CEILING = 0.98


@dataclass(frozen=True, slots=True)
class ForkEdit:
    op: str
    seat: int
    value: int | None = None
    card: str | None = None


def _value_to_wdl(raw_value: float) -> dict[str, float]:
    win = max(0.0, min(1.0, 0.5 * (raw_value + 1.0)))
    draw = 0.01 if _DRAW_FLOOR < win < _DRAW_CEILING else 0.0
    loss = max(0.0, 1.0 - win - draw)
    return {"win": round(win, 4), "draw": round(draw, 4), "loss": round(loss, 4)}


def _heuristic_eval(obs: managym.Observation) -> tuple[list[float], dict[str, float]]:
    count = max(1, len(obs.action_space.actions))
    weights = [1.0 / (idx + 1) for idx in range(count)]
    total = sum(weights)
    probs = [round(weight / total, 4) for weight in weights]
    life_diff = int(obs.agent.life) - int(obs.opponent.life)
    hero_pow = sum(
        int(p.characteristics.power or 0) for p in obs.agent_permanents if p.characteristics.card_types.is_creature
    )
    opp_pow = sum(
        int(p.characteristics.power or 0) for p in obs.opponent_permanents if p.characteristics.card_types.is_creature
    )
    raw_value = max(-0.92, min(0.92, 0.03 * life_diff + 0.04 * (hero_pow - opp_pow)))
    return probs, _value_to_wdl(raw_value)


def compute_policy_diff(probs_a: list[float], probs_b: list[float]) -> dict[str, object]:
    if not probs_a or len(probs_a) != len(probs_b):
        return {"tv": 0.0, "kl": 0.0, "argmax_changed": False}
    tv = 0.5 * sum(abs(p - q) for p, q in zip(probs_a, probs_b, strict=True))
    kl = sum(
        p * math.log(max(p, _MIN_PROB) / max(q, _MIN_PROB)) for p, q in zip(probs_a, probs_b, strict=True) if p > 0.0
    )
    top_a = max(range(len(probs_a)), key=probs_a.__getitem__)
    top_b = max(range(len(probs_b)), key=probs_b.__getitem__)
    return {
        "tv": round(tv, 4),
        "kl": round(max(0.0, kl), 4),
        "argmax_changed": top_a != top_b,
    }


def _apply_single_edit(cloned: managym.Env, edit: ForkEdit) -> str | None:
    if edit.op == "set_life" and edit.value is not None:
        cloned.scenario_set_life(edit.seat, max(1, min(99, int(edit.value))))
        return None
    if edit.op == "clear_hand":
        cloned.scenario_clear_hand(edit.seat)
        return None
    if edit.op == "add_hand_card" and edit.card:
        cloned.scenario_force_card_in_hand(edit.seat, edit.card)
        return None
    if edit.op == "add_battlefield" and edit.card:
        cloned.scenario_force_battlefield(edit.seat, edit.card, ready=True)
        return None
    return f"unsupported fork edit: {edit.op}"


def apply_fork_edits(
    env: managym.Env,
    base_obs: managym.Observation,
    *,
    edits: list[ForkEdit],
) -> tuple[managym.Env, managym.Observation, list[str]]:
    cloned = env.clone_env()
    warnings: list[str] = []
    for edit in edits:
        try:
            warn = _apply_single_edit(cloned, edit)
            if warn is not None:
                warnings.append(warn)
        except (RuntimeError, ValueError) as exc:
            warnings.append(f"{edit.op} ({edit.card or edit.value}): {exc}")
    try:
        refreshed = cloned.scenario_refresh()
    except (RuntimeError, ValueError) as exc:
        warnings.append(f"scenario_refresh skipped (non-priority step): {exc}")
        return cloned, base_obs, warnings
    return cloned, refreshed, warnings


class InferenceEngine:
    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir.resolve()
        self._cache: OrderedDict[str, tuple[ActorCritic, ObservationSpace]] = OrderedDict()

    def _resolve_checkpoint(self, run_id: str, checkpoint_name: str) -> Path:
        run_dir = resolve_run_dir(self._runs_dir, run_id)
        filename = checkpoint_name if checkpoint_name.endswith(".safetensors") else f"{checkpoint_name}.safetensors"
        path = (run_dir / "checkpoints" / filename).resolve()
        if not path.is_relative_to(run_dir / "checkpoints") or not path.is_file():
            msg = f"checkpoint not found: {run_id}/{filename}"
            raise FileNotFoundError(msg)
        return path

    def _load(self, path: Path) -> tuple[ActorCritic, ObservationSpace]:
        key = str(path)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        loaded = load_checkpoint(key)
        agent, obs_space = loaded[0], loaded[1]
        agent.eval()
        self._cache[key] = (agent, obs_space)
        while len(self._cache) > _MAX_LOADED_CKPTS:
            self._cache.popitem(last=False)
        return agent, obs_space

    def _eval_one(
        self,
        obs: managym.Observation,
        *,
        run_id: str,
        checkpoint_name: str | None,
    ) -> tuple[list[float], dict[str, float], str]:
        if not checkpoint_name:
            probs, wdl = _heuristic_eval(obs)
            return probs, wdl, "heuristic"
        try:
            ckpt_path = self._resolve_checkpoint(run_id, checkpoint_name)
            agent, obs_space = self._load(ckpt_path)
            with agent.inference_scope() as scope:
                raw_probs, raw_val = evaluate_observation(scope, obs_space, obs)
            probs = [round(float(p), 4) for p in raw_probs]
            return probs, _value_to_wdl(raw_val), checkpoint_name
        except (FileNotFoundError, RuntimeError, ValueError):
            probs, wdl = _heuristic_eval(obs)
            return probs, wdl, "heuristic"

    def evaluate_pair(
        self,
        obs: managym.Observation,
        *,
        run_id: str,
        ckpt_a: str | None = None,
        ckpt_b: str | None = None,
    ) -> dict[str, object]:
        probs_a, wdl_a, label_a = self._eval_one(obs, run_id=run_id, checkpoint_name=ckpt_a)
        probs_b, wdl_b, label_b = (
            self._eval_one(obs, run_id=run_id, checkpoint_name=ckpt_b) if ckpt_b else (probs_a, wdl_a, label_a)
        )
        diff = compute_policy_diff(probs_a, probs_b)
        return {
            "primary": {"checkpoint": label_a, "probs": probs_a, "wdl": wdl_a},
            "compare": {"checkpoint": label_b, "probs": probs_b, "wdl": wdl_b},
            "diff": diff,
        }

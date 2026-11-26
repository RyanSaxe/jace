"""Checkpoint inference, policy diffing, batched trajectory scoring, and Fork/Branch execution.

Loads saved `.safetensors` checkpoints through a thread-safe LRU cache with a
persistent `PolicyInference` scope, evaluates legal-action probabilities and
3-class WDL distributions normalized to Hero's perspective, scores sampled game
trajectories in batches for mistake/surprise indexing, and runs stateless
counterfactual board edits plus any-step action branching via `env.clone_env()`.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import managym
import torch
from manabot.backends.torch.checkpoint import load_checkpoint
from manabot.env.observation import ObservationSpace
from managym.enums import ActionEnum

from jace.studio.replay import engine_step_errors, non_permanent_card_names
from jace.studio.runs import resolve_run_dir

if TYPE_CHECKING:
    from pathlib import Path

    from manabot.backends.torch.policy import ActorCritic, PolicyInference

__all__ = (
    "BranchStepEvent",
    "ForkEdit",
    "InferenceEngine",
    "apply_fork_and_branch",
    "apply_fork_edits",
    "compute_policy_diff",
)

_MAX_LOADED_CKPTS = 4
_MIN_PROB = 1e-12
_WDL_CLASSES = 3


@dataclass(frozen=True, slots=True)
class ForkEdit:
    op: str
    seat: int
    value: int | None = None
    card: str | None = None
    after_branch: int = 0


@dataclass(frozen=True, slots=True)
class BranchStepEvent:
    branch_index: int
    turn: int
    actor_side: str
    action_index: int
    label: str


@dataclass(slots=True)
class _CachedModel:
    agent: ActorCritic
    obs_space: ObservationSpace
    scope: PolicyInference


def _scalar_to_wdl(raw_value: float) -> dict[str, float]:
    win = max(0.0, min(1.0, 0.5 * (raw_value + 1.0)))
    loss = max(0.0, 1.0 - win)
    return {"win": round(win, 4), "draw": 0.0, "loss": round(loss, 4)}


def _normalize_wdl_for_hero(
    wdl: dict[str, float],
    *,
    acting_seat: int,
    hero_seat: int,
) -> dict[str, float]:
    if acting_seat == hero_seat:
        return wdl
    return {"win": wdl["loss"], "draw": wdl["draw"], "loss": wdl["win"]}


def _heuristic_eval(obs: managym.Observation, *, hero_seat: int = 0) -> tuple[list[float], dict[str, float]]:
    count = max(1, len(obs.action_space.actions))
    weights = [1.0 / (idx + 1) for idx in range(count)]
    total = sum(weights)
    probs = [round(weight / total, 4) for weight in weights]
    life_diff = int(obs.agent.life) - int(obs.opponent.life)
    agent_pow = sum(
        int(p.characteristics.power or 0) for p in obs.agent_permanents if p.characteristics.card_types.is_creature
    )
    opp_pow = sum(
        int(p.characteristics.power or 0) for p in obs.opponent_permanents if p.characteristics.card_types.is_creature
    )
    raw_value = max(-0.92, min(0.92, 0.03 * life_diff + 0.04 * (agent_pow - opp_pow)))
    wdl = _normalize_wdl_for_hero(
        _scalar_to_wdl(raw_value),
        acting_seat=int(obs.agent.player_index),
        hero_seat=hero_seat,
    )
    return probs, wdl


def _forward_model(
    cached: _CachedModel,
    obs: managym.Observation,
    *,
    hero_seat: int,
) -> tuple[list[float], dict[str, float]]:
    count = len(obs.action_space.actions)
    space = cached.obs_space
    active_space = (
        ObservationSpace(space.config.model_copy(update={"max_actions": count}))
        if count > space.config.max_actions
        else space
    )
    obs_batch = managym.encode_observation_batch([obs], active_space.layout, active_space.config.action_overflow)
    with torch.no_grad():
        prepared = cached.scope.prepare(obs_batch)
        logits, value_out = cached.agent.forward_distribution(prepared)
        raw_probs = torch.softmax(logits, dim=-1)[0, :count].tolist()
        if value_out.shape[-1] == _WDL_CLASSES:
            loss_p, draw_p, win_p = torch.softmax(value_out, dim=-1)[0].tolist()
            wdl = {"win": round(float(win_p), 4), "draw": round(float(draw_p), 4), "loss": round(float(loss_p), 4)}
        else:
            wdl = _scalar_to_wdl(float(value_out.item()))
    probs = [round(float(p), 4) for p in raw_probs]
    hero_wdl = _normalize_wdl_for_hero(wdl, acting_seat=int(obs.agent.player_index), hero_seat=hero_seat)
    return probs, hero_wdl


def _forward_batch_hero_win(
    cached: _CachedModel,
    obs_list: list[managym.Observation],
    *,
    hero_seat: int,
) -> list[float]:
    if not obs_list:
        return []
    max_actions = max((len(obs.action_space.actions) for obs in obs_list), default=1)
    space = cached.obs_space
    active_space = (
        ObservationSpace(space.config.model_copy(update={"max_actions": max_actions}))
        if max_actions > space.config.max_actions
        else space
    )
    obs_batch = managym.encode_observation_batch(obs_list, active_space.layout, active_space.config.action_overflow)
    with torch.no_grad():
        prepared = cached.scope.prepare(obs_batch)
        _, value_out = cached.agent.forward_distribution(prepared)
        if value_out.shape[-1] == _WDL_CLASSES:
            wdl_probs = torch.softmax(value_out, dim=-1).tolist()
            results: list[float] = []
            for obs, row in zip(obs_list, wdl_probs, strict=True):
                loss_p, _, win_p = float(row[0]), float(row[1]), float(row[2])
                hero_win = win_p if int(obs.agent.player_index) == hero_seat else loss_p
                results.append(round(hero_win, 4))
            return results
        scalars = value_out.reshape(-1).tolist()
        results = []
        for obs, raw_val in zip(obs_list, scalars, strict=True):
            wdl = _normalize_wdl_for_hero(
                _scalar_to_wdl(float(raw_val)),
                acting_seat=int(obs.agent.player_index),
                hero_seat=hero_seat,
            )
            results.append(wdl["win"])
        return results


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
        if edit.card in non_permanent_card_names():
            return f"Cannot put Instant/Sorcery '{edit.card}' onto the battlefield (CR 304.4) - add to Hand instead."
        cloned.scenario_force_battlefield(edit.seat, edit.card, ready=True)
        return None
    return f"unsupported fork edit: {edit.op}"


def _describe_branch_action(obs: managym.Observation, action_idx: int) -> str:
    actions = obs.action_space.actions
    if action_idx < 0 or action_idx >= len(actions):
        return f"Action #{action_idx}"
    action = actions[action_idx]
    if action.description:
        return str(action.description)
    return ActionEnum(int(action.action_type)).name.replace("_", " ").title()


def apply_fork_edits(
    env: managym.Env,
    base_obs: managym.Observation,
    *,
    edits: list[ForkEdit],
) -> tuple[managym.Env, managym.Observation, list[str]]:
    if not edits:
        return env.clone_env(), base_obs, []
    step_errors = engine_step_errors()
    cloned = env.clone_env()
    warnings: list[str] = []
    for edit in edits:
        try:
            warn = _apply_single_edit(cloned, edit)
            if warn is not None:
                warnings.append(warn)
        except step_errors as exc:
            warnings.append(f"{edit.op} ({edit.card or edit.value}): {exc}")
    try:
        refreshed = cloned.scenario_refresh()
    except step_errors as exc:
        warnings.append(f"Board edits require a Priority step ({exc})")
        return env.clone_env(), base_obs, warnings
    return cloned, refreshed, warnings


def _group_edits_by_stage(edits: list[ForkEdit], max_stage: int) -> dict[int, list[ForkEdit]]:
    grouped: dict[int, list[ForkEdit]] = {}
    for edit in edits:
        stage = max(0, min(int(edit.after_branch), max_stage))
        grouped.setdefault(stage, []).append(edit)
    return grouped


def _step_branch_once(
    cloned: managym.Env,
    current_obs: managym.Observation,
    *,
    idx: int,
    action_idx: int,
    hero_seat: int,
    step_errors: tuple[type[BaseException], ...],
) -> tuple[managym.Observation, BranchStepEvent | None, bool, str | None]:
    actions = current_obs.action_space.actions
    if action_idx < 0 or action_idx >= len(actions):
        err = f"Branch step {idx + 1}: action index {action_idx} out of bounds ({len(actions)} legal)"
        return current_obs, None, False, err
    label = _describe_branch_action(current_obs, action_idx)
    actor_side = "hero" if int(current_obs.agent.player_index) == hero_seat else "opp"
    turn_num = int(current_obs.turn.turn_number)
    try:
        next_obs, _, done, truncated, _ = cloned.step(action_idx)
    except step_errors as exc:
        return current_obs, None, False, f"Branch step {idx + 1} failed: {exc}"
    event = BranchStepEvent(
        branch_index=idx, turn=turn_num, actor_side=actor_side, action_index=action_idx, label=label
    )
    return next_obs, event, bool(done or truncated), None


def apply_fork_and_branch(
    env: managym.Env,
    base_obs: managym.Observation,
    *,
    edits: list[ForkEdit],
    branch_actions: list[int],
    hero_seat: int = 0,
) -> tuple[managym.Env, managym.Observation, list[str], list[BranchStepEvent], bool]:
    step_errors = engine_step_errors()
    edits_by_stage = _group_edits_by_stage(edits, len(branch_actions))
    cloned, current_obs, warnings = apply_fork_edits(env, base_obs, edits=edits_by_stage.get(0, []))
    branch_events: list[BranchStepEvent] = []
    branch_done = False
    for idx, action_idx in enumerate(branch_actions):
        next_obs, event, is_done, err = _step_branch_once(
            cloned, current_obs, idx=idx, action_idx=action_idx, hero_seat=hero_seat, step_errors=step_errors
        )
        if err is not None or event is None:
            if err is not None:
                warnings.append(err)
            break
        branch_events.append(event)
        current_obs = next_obs
        if is_done:
            branch_done = True
            break
        stage_edits = edits_by_stage.get(idx + 1, [])
        if stage_edits:
            cloned, current_obs, stage_warns = apply_fork_edits(cloned, current_obs, edits=stage_edits)
            warnings.extend(stage_warns)
    return cloned, current_obs, warnings, branch_events, branch_done


class InferenceEngine:
    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir.resolve()
        self._cache: OrderedDict[str, _CachedModel] = OrderedDict()
        self._lock = threading.Lock()

    def _resolve_checkpoint(self, run_id: str, checkpoint_name: str) -> Path:
        run_dir = resolve_run_dir(self._runs_dir, run_id)
        filename = checkpoint_name if checkpoint_name.endswith(".safetensors") else f"{checkpoint_name}.safetensors"
        path = (run_dir / "checkpoints" / filename).resolve()
        if not path.is_relative_to(run_dir / "checkpoints") or not path.is_file():
            msg = f"checkpoint not found: {run_id}/{filename}"
            raise FileNotFoundError(msg)
        return path

    def find_checkpoint_for_iteration(self, run_id: str, iteration: int) -> str | None:
        try:
            run_dir = resolve_run_dir(self._runs_dir, run_id)
        except (FileNotFoundError, ValueError):
            return None
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            return None
        exact = ckpt_dir / f"iter-{iteration:04d}.safetensors"
        if exact.is_file():
            return exact.stem
        exact_ema = ckpt_dir / f"iter-{iteration:04d}-ema.safetensors"
        if exact_ema.is_file():
            return exact_ema.stem
        candidates = sorted(p for p in ckpt_dir.iterdir() if p.is_file() and p.suffix == ".safetensors")
        return candidates[-1].stem if candidates else None

    def _get_or_load_locked(self, path: Path) -> _CachedModel:
        key = str(path)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        loaded = load_checkpoint(key)
        agent, obs_space = loaded[0], loaded[1]
        agent.eval()
        scope = agent.inference_scope()
        scope.__enter__()
        model = _CachedModel(agent=agent, obs_space=obs_space, scope=scope)
        self._cache[key] = model
        while len(self._cache) > _MAX_LOADED_CKPTS:
            _, evicted = self._cache.popitem(last=False)
            evicted.scope.__exit__(None, None, None)
        return model

    def evaluate_batch_win_probs(
        self,
        obs_list: list[managym.Observation],
        *,
        run_id: str,
        checkpoint_name: str | None,
        hero_seat: int,
    ) -> tuple[list[float], str]:
        if not obs_list:
            return [], "heuristic"
        if checkpoint_name and checkpoint_name != "heuristic":
            try:
                ckpt_path = self._resolve_checkpoint(run_id, checkpoint_name)
                with self._lock:
                    cached = self._get_or_load_locked(ckpt_path)
                    wins = _forward_batch_hero_win(cached, obs_list, hero_seat=hero_seat)
            except (FileNotFoundError, RuntimeError, ValueError):
                pass
            else:
                return wins, "model"
        return [_heuristic_eval(obs, hero_seat=hero_seat)[1]["win"] for obs in obs_list], "heuristic"

    def _eval_one(
        self,
        obs: managym.Observation,
        *,
        run_id: str,
        checkpoint_name: str | None,
        hero_seat: int,
    ) -> tuple[list[float], dict[str, float], str, str | None]:
        if not checkpoint_name or checkpoint_name == "heuristic":
            probs, wdl = _heuristic_eval(obs, hero_seat=hero_seat)
            return probs, wdl, "heuristic", None
        try:
            ckpt_path = self._resolve_checkpoint(run_id, checkpoint_name)
            with self._lock:
                cached = self._get_or_load_locked(ckpt_path)
                probs, wdl = _forward_model(cached, obs, hero_seat=hero_seat)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            probs, wdl = _heuristic_eval(obs, hero_seat=hero_seat)
            return probs, wdl, "heuristic", f"Checkpoint '{checkpoint_name}' fallback to heuristic: {exc}"
        else:
            return probs, wdl, checkpoint_name, None

    def evaluate_pair(
        self,
        obs: managym.Observation,
        *,
        run_id: str,
        hero_seat: int = 0,
        ckpt_a: str | None = None,
        ckpt_b: str | None = None,
        hide_actions: bool = False,
    ) -> dict[str, object]:
        probs_a, wdl_a, label_a, warn_a = self._eval_one(
            obs, run_id=run_id, checkpoint_name=ckpt_a, hero_seat=hero_seat
        )
        if ckpt_b and ckpt_b != "same":
            probs_b, wdl_b, label_b, warn_b = self._eval_one(
                obs, run_id=run_id, checkpoint_name=ckpt_b, hero_seat=hero_seat
            )
        else:
            probs_b, wdl_b, label_b, warn_b = probs_a, wdl_a, label_a, None
        diff = (
            {"tv": 0.0, "kl": 0.0, "argmax_changed": False} if hide_actions else compute_policy_diff(probs_a, probs_b)
        )
        warnings = [w for w in (warn_a, warn_b) if w]
        return {
            "primary": {"checkpoint": label_a, "probs": [] if hide_actions else probs_a, "wdl": wdl_a},
            "compare": {"checkpoint": label_b, "probs": [] if hide_actions else probs_b, "wdl": wdl_b},
            "diff": diff,
            "warnings": warnings,
        }

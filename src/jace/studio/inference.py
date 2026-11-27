"""Checkpoint inference, policy diffing, and N-model policy matrix evaluation.

Loads saved `.safetensors` checkpoints (including cross-run `run_id::ckpt`
references) through a thread-safe LRU cache with device acceleration (`cuda` /
`mps` / `cpu` with safe CPU fallback), evaluates legal-action probabilities and
3-class WDL distributions normalized to Hero's perspective, re-exports stateless
fork/branch helpers from `jace.studio.forking`, and delegates batched playouts
and cross-checkpoint comparisons to `jace.studio.analysis`.
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

from jace.modules import default_device_name
from jace.studio.forking import BranchStepEvent, ForkEdit, apply_fork_and_branch, apply_fork_edits
from jace.studio.replay import serialize_legal_actions
from jace.studio.runs import resolve_run_dir

if TYPE_CHECKING:
    from collections.abc import Callable
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

_MAX_LOADED_CKPTS = 6
_MIN_PROB = 1e-12
_WDL_CLASSES = 3
_BATCH_CHUNK_SIZE = 64


@dataclass(slots=True)
class _CachedModel:
    agent: ActorCritic
    obs_space: ObservationSpace
    scope: PolicyInference
    device: str


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


def _format_checkpoint_label(ref: str) -> str:
    if not ref or ref == "heuristic":
        return "Heuristic"
    if "::" in ref:
        run_part, ckpt_part = ref.split("::", 1)
        run_tail = run_part.split("/")[-1]
        return f"{run_tail} · {ckpt_part}"
    return ref


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


def _active_obs_space(space: ObservationSpace, max_actions: int) -> ObservationSpace:
    if max_actions > space.config.max_actions:
        return ObservationSpace(space.config.model_copy(update={"max_actions": max_actions}))
    return space


def _move_cached_to_cpu(cached: _CachedModel) -> None:
    cached.scope.__exit__(None, None, None)
    cached.agent.to("cpu")
    new_scope = cached.agent.inference_scope()
    new_scope.__enter__()
    cached.scope = new_scope
    cached.device = "cpu"


def _run_forward_distribution(
    cached: _CachedModel,
    obs_batch: managym.ObservationBatch,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        with torch.no_grad():
            prepared = cached.scope.prepare(obs_batch)
            return cached.agent.forward_distribution(prepared)
    except RuntimeError:
        if cached.device == "cpu":
            raise
        _move_cached_to_cpu(cached)
        with torch.no_grad():
            prepared = cached.scope.prepare(obs_batch)
            return cached.agent.forward_distribution(prepared)


def _decode_wdl_rows(
    value_out: torch.Tensor,
    obs_list: list[managym.Observation],
    *,
    hero_seats: list[int],
) -> list[dict[str, float]]:
    if value_out.shape[-1] == _WDL_CLASSES:
        rows = torch.softmax(value_out, dim=-1).cpu().tolist()
        return [
            _normalize_wdl_for_hero(
                {"win": round(float(row[2]), 4), "draw": round(float(row[1]), 4), "loss": round(float(row[0]), 4)},
                acting_seat=int(obs.agent.player_index),
                hero_seat=seat,
            )
            for obs, row, seat in zip(obs_list, rows, hero_seats, strict=True)
        ]
    scalars = value_out.reshape(-1).cpu().tolist()
    return [
        _normalize_wdl_for_hero(
            _scalar_to_wdl(float(raw_val)),
            acting_seat=int(obs.agent.player_index),
            hero_seat=seat,
        )
        for obs, raw_val, seat in zip(obs_list, scalars, hero_seats, strict=True)
    ]


def _forward_batch_full(
    cached: _CachedModel,
    obs_list: list[managym.Observation],
    *,
    hero_seats: list[int],
) -> tuple[list[list[float]], list[dict[str, float]]]:
    if not obs_list:
        return [], []
    probs_out: list[list[float]] = []
    wdls_out: list[dict[str, float]] = []
    for start in range(0, len(obs_list), _BATCH_CHUNK_SIZE):
        chunk = obs_list[start : start + _BATCH_CHUNK_SIZE]
        chunk_seats = hero_seats[start : start + _BATCH_CHUNK_SIZE]
        max_actions = max((len(obs.action_space.actions) for obs in chunk), default=1)
        active_space = _active_obs_space(cached.obs_space, max_actions)
        obs_batch = managym.encode_observation_batch(chunk, active_space.layout, active_space.config.action_overflow)
        logits, value_out = _run_forward_distribution(cached, obs_batch)
        probs_tensor = torch.softmax(logits, dim=-1).cpu()
        for row_idx, obs in enumerate(chunk):
            count = len(obs.action_space.actions)
            raw_probs = probs_tensor[row_idx, :count].tolist()
            probs_out.append([round(float(p), 4) for p in raw_probs])
        wdls_out.extend(_decode_wdl_rows(value_out, chunk, hero_seats=chunk_seats))
    return probs_out, wdls_out


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


class InferenceEngine:
    def __init__(self, runs_dir: Path) -> None:
        self._runs_dir = runs_dir.resolve()
        self._cache: OrderedDict[str, _CachedModel] = OrderedDict()
        self._lock = threading.Lock()
        self._preferred_device = default_device_name()

    @property
    def preferred_device(self) -> str:
        return self._preferred_device

    @staticmethod
    def diff_policies(probs_a: list[float], probs_b: list[float]) -> dict[str, object]:
        return compute_policy_diff(probs_a, probs_b)

    def _resolve_checkpoint(self, run_id: str, checkpoint_name: str) -> Path:
        if "::" in checkpoint_name:
            target_run_id, ckpt_name = checkpoint_name.split("::", 1)
        else:
            target_run_id, ckpt_name = run_id, checkpoint_name
        run_dir = resolve_run_dir(self._runs_dir, target_run_id)
        filename = ckpt_name if ckpt_name.endswith(".safetensors") else f"{ckpt_name}.safetensors"
        path = (run_dir / "checkpoints" / filename).resolve()
        if not path.is_relative_to(run_dir / "checkpoints") or not path.is_file():
            msg = f"checkpoint not found: {target_run_id}/{filename}"
            raise FileNotFoundError(msg)
        return path

    def can_load_checkpoint(self, run_id: str, checkpoint_name: str) -> bool:
        try:
            self._resolve_checkpoint(run_id, checkpoint_name)
        except (FileNotFoundError, ValueError):
            return False
        return True

    def find_checkpoint_for_iteration(self, run_id: str, iteration: int) -> str | None:
        try:
            run_dir = resolve_run_dir(self._runs_dir, run_id)
        except (FileNotFoundError, ValueError):
            return None
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.is_dir():
            return None
        for candidate_name in (f"iter-{iteration:04d}.safetensors", f"iter-{iteration:04d}-ema.safetensors"):
            exact = ckpt_dir / candidate_name
            if exact.is_file():
                return exact.stem
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
        device = self._preferred_device
        try:
            agent.to(device)
            scope = agent.inference_scope()
            scope.__enter__()
        except RuntimeError:
            device = "cpu"
            agent.to(device)
            scope = agent.inference_scope()
            scope.__enter__()
        model = _CachedModel(agent=agent, obs_space=obs_space, scope=scope, device=device)
        self._cache[key] = model
        while len(self._cache) > _MAX_LOADED_CKPTS:
            _, evicted = self._cache.popitem(last=False)
            evicted.scope.__exit__(None, None, None)
        return model

    def run_with_checkpoint[T](
        self,
        *,
        run_id: str,
        checkpoint_name: str,
        fn: Callable[[PolicyInference, ObservationSpace], T],
    ) -> T:
        ckpt_path = self._resolve_checkpoint(run_id, checkpoint_name)
        with self._lock:
            cached = self._get_or_load_locked(ckpt_path)
            return fn(cached.scope, cached.obs_space)

    def evaluate_batch_full(
        self,
        obs_list: list[managym.Observation],
        *,
        run_id: str,
        checkpoint_name: str | None,
        hero_seats: list[int],
    ) -> tuple[list[list[float]], list[dict[str, float]], str, str | None]:
        if not obs_list:
            return [], [], "cpu", None
        if checkpoint_name and checkpoint_name != "heuristic":
            try:
                ckpt_path = self._resolve_checkpoint(run_id, checkpoint_name)
                with self._lock:
                    cached = self._get_or_load_locked(ckpt_path)
                    probs, wdls = _forward_batch_full(cached, obs_list, hero_seats=hero_seats)
                    device = cached.device
            except (FileNotFoundError, RuntimeError, ValueError) as exc:
                pairs = [_heuristic_eval(obs, hero_seat=seat) for obs, seat in zip(obs_list, hero_seats, strict=True)]
                return (
                    [p for p, _ in pairs],
                    [w for _, w in pairs],
                    "cpu",
                    f"Checkpoint '{checkpoint_name}' fallback to heuristic: {exc}",
                )
            else:
                return probs, wdls, device, None
        pairs = [_heuristic_eval(obs, hero_seat=seat) for obs, seat in zip(obs_list, hero_seats, strict=True)]
        return [p for p, _ in pairs], [w for _, w in pairs], "cpu", None

    def evaluate_batch_action_probs(
        self,
        obs_list: list[managym.Observation],
        *,
        run_id: str,
        checkpoint_name: str,
    ) -> tuple[list[list[float]], str, str | None]:
        probs, _, device, warn = self.evaluate_batch_full(
            obs_list, run_id=run_id, checkpoint_name=checkpoint_name, hero_seats=[0] * len(obs_list)
        )
        return probs, device, warn

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
        _, wdls, _, warn = self.evaluate_batch_full(
            obs_list,
            run_id=run_id,
            checkpoint_name=checkpoint_name,
            hero_seats=[hero_seat] * len(obs_list),
        )
        used_model = bool(checkpoint_name) and checkpoint_name != "heuristic" and warn is None
        return [float(w["win"]) for w in wdls], ("model" if used_model else "heuristic")

    def _eval_one(
        self,
        obs: managym.Observation,
        *,
        run_id: str,
        checkpoint_name: str | None,
        hero_seat: int,
    ) -> tuple[list[float], dict[str, float], str, str | None]:
        probs_list, wdls_list, _, warn = self.evaluate_batch_full(
            [obs], run_id=run_id, checkpoint_name=checkpoint_name, hero_seats=[hero_seat]
        )
        is_model = bool(checkpoint_name) and checkpoint_name != "heuristic" and warn is None
        label = checkpoint_name if (is_model and checkpoint_name is not None) else "heuristic"
        return probs_list[0], wdls_list[0], label, warn

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

    def evaluate_matrix(
        self,
        obs: managym.Observation,
        *,
        run_id: str,
        hero_seat: int = 0,
        models: list[str],
        hide_actions: bool = False,
    ) -> dict[str, object]:
        legal_actions = [] if hide_actions else serialize_legal_actions(obs, hero_seat=hero_seat)
        columns: list[dict[str, object]] = []
        for ref in models:
            probs, wdl, _, warn = self._eval_one(obs, run_id=run_id, checkpoint_name=ref, hero_seat=hero_seat)
            visible_probs = [] if hide_actions else probs
            top_idx = max(range(len(visible_probs)), key=visible_probs.__getitem__) if visible_probs else None
            columns.append(
                {
                    "ref": ref,
                    "label": _format_checkpoint_label(ref),
                    "probs": visible_probs,
                    "wdl": wdl,
                    "top_action_index": top_idx,
                    "warning": warn,
                }
            )
        return {"legal_actions": legal_actions, "columns": columns}

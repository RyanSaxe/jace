"""Cross-fork batched counterfactual rollout engine with on-policy value-head reuse.

Eliminates the small-batch bottleneck of per-fork rollouts by harvesting cloned decision
roots across each worker's deals first and stepping every active branch across all forks
in one wide batched forward pass per ply. Optionally captures on-policy terminal WDL
value targets from continuation states (Cobbe et al., 2020) and supports truncated
value-bootstrapped horizons (Sokota et al., 2025; Schrittwieser et al., 2019).
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, Self

import managym
import numpy as np
import torch
from manabot.backends.torch.bot import evaluate_observation
from managym.enums import ActionSpaceEnum

from jace.collect.agent import DecisionMemoryKind, build_agent
from jace.collect.decks import elemental_decks, load_human_deck_catalog
from jace.eval.deals import Deal, build_deals, build_human_deals
from jace.forks.collect import ForkExample

if TYPE_CHECKING:
    from collections.abc import Sequence
    from multiprocessing.pool import Pool as ProcessPool
    from pathlib import Path

    from manabot.backends.torch.policy import ActorCritic, PolicyInference
    from manabot.env.observation import ObservationSpace

__all__ = (
    "BatchedForkCollectorPool",
    "BatchedForkConfig",
    "BatchedForkResult",
    "ContinuationValueSample",
)

_ACTION_SPACE_NAMES = {int(member): member.name.lower() for member in ActionSpaceEnum}
_MIN_BRANCH_ACTIONS = 2


@dataclass(slots=True)
class _BatchWorkerState:
    agent: ActorCritic | None = None
    obs_space: ObservationSpace | None = None
    decks: list[tuple[str, managym.Deck]] | None = None
    human_decks_path: Path | None = None


_BATCH_WORKER = _BatchWorkerState()


@dataclass(frozen=True, slots=True)
class BatchedForkConfig:
    forks_per_update: int = 64
    playouts_per_action: int = 4
    decision_kinds: tuple[str, ...] | None = ("declare_attacker",)
    sample_prob: float = 0.35
    num_workers: int = 8
    max_steps: int = 1200
    rollout_horizon: int | None = None
    value_samples_per_fork: int = 8
    temperature: float = 0.25
    min_gap: float = 0.25
    fork_weight: float = 0.03
    value_reuse_weight: float = 0.25
    anchor_kl: float = 0.2


@dataclass(frozen=True, slots=True)
class ContinuationValueSample:
    encoded_obs: dict[str, np.ndarray]
    outcome_class: int


@dataclass(frozen=True, slots=True)
class BatchedForkResult:
    examples: list[ForkExample]
    value_samples: list[ContinuationValueSample]


@dataclass(slots=True)
class _HarvestedRoot:
    env: managym.Env
    encoded_obs: dict[str, np.ndarray]
    hero_idx: int
    action_pair: tuple[int, int]
    policy_pair_probs: tuple[float, float]
    decision_kind: str
    turn: int
    num_actions: int
    base_seed: int


@dataclass(slots=True)
class _ActiveBranch:
    fork_idx: int
    pair_idx: int
    arm_idx: int
    hero_idx: int
    env: managym.Env
    obs: managym.Observation | None
    rng: random.Random
    done: bool = False
    failed: bool = False
    steps: int = 0
    leaf_value: float | None = None
    sampled_obs: list[tuple[dict[str, np.ndarray], int]] | None = None


def _init_batch_worker(init_spec: tuple[int, int, DecisionMemoryKind, Path | None]) -> None:
    hidden_dim, attention_layers, decision_memory, human_decks_path = init_spec
    logging.getLogger("manabot").setLevel(logging.WARNING)
    torch.set_num_threads(1)
    agent, obs_space = build_agent(
        hidden_dim=hidden_dim,
        attention_layers=attention_layers,
        decision_memory=decision_memory,
        bucketed_shapes=True,
        live_tokens=False,
    )
    _BATCH_WORKER.agent, _BATCH_WORKER.obs_space = agent, obs_space
    _BATCH_WORKER.human_decks_path = human_decks_path
    _BATCH_WORKER.decks = None if human_decks_path is not None else elemental_decks()


def _engine_error_types() -> tuple[type[BaseException], ...]:
    panic_types = tuple(cls for cls in BaseException.__subclasses__() if cls.__name__ == "PanicException")
    return (*panic_types, RuntimeError, ValueError, KeyError, IndexError)


class _ForkEnv(Protocol):
    def step(self, action: int) -> tuple[managym.Observation, float, bool, bool, dict[str, object]]: ...
    def is_game_over(self) -> bool: ...
    def clone_env(self) -> managym.Env: ...


def _safe_step(env: _ForkEnv, action: int) -> tuple[managym.Observation | None, bool, bool]:
    try:
        obs, _, terminated, truncated, _ = env.step(action)
    except _engine_error_types():
        return None, True, True
    return obs, bool(terminated or truncated or env.is_game_over()), False


def _safe_clone(env: _ForkEnv) -> managym.Env | None:
    try:
        return env.clone_env()
    except _engine_error_types():
        return None


def _batched_eval(
    scope: PolicyInference,
    *,
    obs_space: ObservationSpace,
    observations: Sequence[managym.Observation],
) -> tuple[list[tuple[float, ...]], list[float], list[dict[str, np.ndarray] | None]]:
    max_actions = obs_space.encoder.max_actions
    batch_indices: list[int] = []
    encoded_list: list[dict[str, np.ndarray]] = []
    counts: list[int] = []
    probs_out: list[tuple[float, ...] | None] = [None] * len(observations)
    vals_out: list[float] = [0.0] * len(observations)
    enc_out: list[dict[str, np.ndarray] | None] = [None] * len(observations)

    for idx, obs in enumerate(observations):
        count = len(obs.action_space.actions)
        if count <= max_actions:
            encoded = {key: np.asarray(val, dtype=np.float32) for key, val in obs_space.encode(obs).items()}
            batch_indices.append(idx)
            encoded_list.append(encoded)
            enc_out[idx] = encoded
            counts.append(count)
        else:
            probs, val = evaluate_observation(scope, obs_space, obs)
            probs_out[idx], vals_out[idx] = probs, float(val)

    if encoded_list:
        stacked = {key: np.stack([item[key] for item in encoded_list], axis=0) for key in encoded_list[0]}
        with torch.no_grad():
            logits, raw_val = scope.agent(scope.prepare(stacked))
            probs_t = torch.softmax(logits, dim=-1).cpu()
            scalars = raw_val.float().reshape(-1).cpu().tolist()
        for row_idx, out_idx in enumerate(batch_indices):
            probs_out[out_idx] = tuple(probs_t[row_idx, : counts[row_idx]].tolist())
            vals_out[out_idx] = float(scalars[row_idx])

    return [item for item in probs_out if item is not None], vals_out, enc_out


def _hero_win_prob(env: managym.Env, *, hero_idx: int, active_value: float) -> float:
    seat = env.current_agent_index()
    signed = active_value if seat is None or int(seat) == hero_idx else -active_value
    return float(min(1.0, max(0.0, 0.5 * (signed + 1.0))))


def _terminal_score(env: managym.Env, *, hero_idx: int, leaf_value: float | None) -> float:
    if leaf_value is not None and not env.is_game_over():
        return leaf_value
    winner = env.winner_index()
    if winner is None:
        return 0.5
    return 1.0 if int(winner) == hero_idx else 0.0


def _maybe_harvest_root(
    env: managym.Env,
    obs: managym.Observation,
    *,
    obs_space: ObservationSpace,
    probs: Sequence[float],
    allowed: frozenset[str] | None,
    sample_prob: float,
    rng: random.Random,
    base_seed: int,
) -> _HarvestedRoot | None:
    kind = _ACTION_SPACE_NAMES.get(int(obs.action_space.action_space_type), "unknown")
    eligible = _MIN_BRANCH_ACTIONS <= len(probs) <= obs_space.encoder.max_actions and (
        allowed is None or kind in allowed
    )
    if not eligible or rng.random() >= sample_prob:
        return None
    hero_idx = env.current_agent_index()
    cloned = _safe_clone(env) if hero_idx is not None else None
    if hero_idx is None or cloned is None:
        return None
    ranked = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
    a0, a1 = ranked[0], ranked[1]
    encoded = {key: np.asarray(val, dtype=np.float32) for key, val in obs_space.encode(obs).items()}
    return _HarvestedRoot(
        cloned,
        encoded,
        int(hero_idx),
        (a0, a1),
        (float(probs[a0]), float(probs[a1])),
        kind,
        int(obs.turn.turn_number),
        len(probs),
        base_seed,
    )


def _harvest_roots_from_deal(
    deal: Deal,
    *,
    scope: PolicyInference,
    obs_space: ObservationSpace,
    remaining: int,
    allowed: frozenset[str] | None,
    config: BatchedForkConfig,
    rng: random.Random,
    base_seed: int,
) -> list[_HarvestedRoot]:
    use_memory = obs_space.encoder.config.max_decision_records > 0
    env = managym.Env(seed=deal.seed, skip_trivial=True, decision_memory=use_memory)
    try:
        obs, _ = env.reset(deal.game_setup())
    except _engine_error_types():
        return []
    roots: list[_HarvestedRoot] = []
    for step_idx in range(config.max_steps):
        if env.is_game_over() or len(roots) >= remaining:
            break
        probs, _ = evaluate_observation(scope, obs_space, obs)
        root = _maybe_harvest_root(
            env,
            obs,
            obs_space=obs_space,
            probs=probs,
            allowed=allowed,
            sample_prob=config.sample_prob,
            rng=rng,
            base_seed=base_seed + step_idx * 31,
        )
        if root is not None:
            roots.append(root)
        action = rng.choices(range(len(probs)), weights=probs, k=1)[0]
        next_obs, is_done, is_failed = _safe_step(env, action)
        if is_done or is_failed or next_obs is None:
            break
        obs = next_obs
    return roots


def _spawn_all_branches(roots: Sequence[_HarvestedRoot], *, config: BatchedForkConfig) -> list[_ActiveBranch]:
    branches: list[_ActiveBranch] = []
    want_samples = config.value_samples_per_fork > 0 and config.rollout_horizon is None
    for fork_idx, root in enumerate(roots):
        for pair_idx in range(config.playouts_per_action):
            pair_seed = root.base_seed + pair_idx * 9973
            for arm_idx, root_action in enumerate(root.action_pair):
                clone = _safe_clone(root.env)
                if clone is None:
                    next_obs, is_done, is_failed = None, True, True
                    clone = root.env
                else:
                    next_obs, is_done, is_failed = _safe_step(clone, root_action)
                branches.append(
                    _ActiveBranch(
                        fork_idx,
                        pair_idx,
                        arm_idx,
                        root.hero_idx,
                        clone,
                        next_obs,
                        random.Random(pair_seed),
                        is_done,
                        is_failed,
                        sampled_obs=[] if want_samples else None,
                    )
                )
    return branches


def _maybe_record_value_sample(
    branch: _ActiveBranch, *, encoded: dict[str, np.ndarray] | None, cap_per_branch: int
) -> None:
    if branch.sampled_obs is None or encoded is None or cap_per_branch <= 0:
        return
    seat = branch.env.current_agent_index()
    if seat is None:
        return
    if len(branch.sampled_obs) < cap_per_branch:
        branch.sampled_obs.append((encoded, int(seat)))
    elif branch.rng.randrange(branch.steps + 1) < cap_per_branch:
        branch.sampled_obs[branch.rng.randrange(cap_per_branch)] = (encoded, int(seat))


def _step_all_branches(
    branches: Sequence[_ActiveBranch],
    *,
    scope: PolicyInference,
    obs_space: ObservationSpace,
    config: BatchedForkConfig,
) -> None:
    horizon = config.rollout_horizon
    limit = config.max_steps if horizon is None else min(config.max_steps, horizon)
    cap_per_branch = max(0, config.value_samples_per_fork // max(1, 2 * config.playouts_per_action))
    for _ in range(limit + 1):
        active = [branch for branch in branches if not branch.done and branch.obs is not None]
        if not active:
            break
        probs_batch, vals_batch, enc_batch = _batched_eval(
            scope, obs_space=obs_space, observations=[b.obs for b in active if b.obs is not None]
        )
        for branch, probs, val, encoded in zip(active, probs_batch, vals_batch, enc_batch, strict=True):
            if horizon is not None and branch.steps >= horizon:
                branch.leaf_value = _hero_win_prob(branch.env, hero_idx=branch.hero_idx, active_value=val)
                branch.done = True
                continue
            _maybe_record_value_sample(branch, encoded=encoded, cap_per_branch=cap_per_branch)
            action = branch.rng.choices(range(len(probs)), weights=probs, k=1)[0]
            next_obs, is_done, is_failed = _safe_step(branch.env, action)
            branch.obs, branch.done, branch.failed = next_obs, is_done, is_failed
            branch.steps += 1


def _collect_value_samples(branches: Sequence[_ActiveBranch]) -> list[ContinuationValueSample]:
    samples: list[ContinuationValueSample] = []
    for branch in branches:
        if branch.failed or branch.sampled_obs is None or not branch.env.is_game_over():
            continue
        winner = branch.env.winner_index()
        for encoded, seat_idx in branch.sampled_obs:
            outcome = 1 if winner is None else (2 if int(winner) == seat_idx else 0)
            samples.append(ContinuationValueSample(encoded_obs=encoded, outcome_class=outcome))
    return samples


def _assemble_results(
    roots: Sequence[_HarvestedRoot], branches: Sequence[_ActiveBranch], *, playouts_per_action: int
) -> BatchedForkResult:
    grouped = {(branch.fork_idx, branch.pair_idx, branch.arm_idx): branch for branch in branches}
    examples: list[ForkExample] = []
    for fork_idx, root in enumerate(roots):
        valid = [
            (
                _terminal_score(b0.env, hero_idx=root.hero_idx, leaf_value=b0.leaf_value),
                _terminal_score(b1.env, hero_idx=root.hero_idx, leaf_value=b1.leaf_value),
            )
            for pair_idx in range(playouts_per_action)
            for b0, b1 in ((grouped[(fork_idx, pair_idx, 0)], grouped[(fork_idx, pair_idx, 1)]),)
            if not b0.failed and not b1.failed
        ]
        if not valid:
            continue
        q0 = sum(pair[0] for pair in valid) / len(valid)
        q1 = sum(pair[1] for pair in valid) / len(valid)
        examples.append(
            ForkExample(
                encoded_obs=root.encoded_obs,
                action_pair=root.action_pair,
                pair_returns=(q0, q1),
                policy_pair_probs=root.policy_pair_probs,
                decision_kind=root.decision_kind,
                turn=root.turn,
                num_actions=root.num_actions,
                valid_pairs=len(valid),
            )
        )
    return BatchedForkResult(examples=examples, value_samples=_collect_value_samples(branches))


def _batch_worker_task(args: tuple[dict[str, np.ndarray], int, BatchedForkConfig, int]) -> BatchedForkResult:
    state_np, target_forks, config, seed = args
    if _BATCH_WORKER.agent is None or _BATCH_WORKER.obs_space is None:
        msg = "batched fork worker was not initialized"
        raise RuntimeError(msg)
    agent, obs_space = _BATCH_WORKER.agent, _BATCH_WORKER.obs_space
    agent.load_state_dict({key: torch.from_numpy(arr) for key, arr in state_np.items()})
    agent.eval()

    deal_count = max(target_forks * 6, 24)
    if _BATCH_WORKER.human_decks_path is not None:
        catalog = load_human_deck_catalog(_BATCH_WORKER.human_decks_path)
        deals = build_human_deals(catalog, split="train", count=deal_count, seed=seed, decklists="closed")
    else:
        deals = build_deals(_BATCH_WORKER.decks or elemental_decks(), count=deal_count, seed=seed, decklists="closed")
    rng, allowed = random.Random(seed), (None if config.decision_kinds is None else frozenset(config.decision_kinds))
    roots: list[_HarvestedRoot] = []
    with agent.inference_scope() as scope:
        for deal_idx, deal in enumerate(deals):
            if len(roots) >= target_forks:
                break
            roots.extend(
                _harvest_roots_from_deal(
                    deal,
                    scope=scope,
                    obs_space=obs_space,
                    remaining=target_forks - len(roots),
                    allowed=allowed,
                    config=config,
                    rng=rng,
                    base_seed=seed * 100_003 + deal_idx * 1009,
                )
            )
        branches = _spawn_all_branches(roots, config=config)
        _step_all_branches(branches, scope=scope, obs_space=obs_space, config=config)
    return _assemble_results(roots, branches, playouts_per_action=config.playouts_per_action)


class BatchedForkCollectorPool:
    _hidden_dim: int
    _attention_layers: int
    _decision_memory: DecisionMemoryKind
    _num_workers: int
    _human_decks_path: Path | None
    _pool: ProcessPool | None

    def __init__(
        self,
        *,
        hidden_dim: int,
        attention_layers: int,
        decision_memory: DecisionMemoryKind = "none",
        num_workers: int = 8,
        human_decks_path: Path | None = None,
    ) -> None:
        self._hidden_dim, self._attention_layers = hidden_dim, attention_layers
        self._decision_memory, self._num_workers = decision_memory, max(1, num_workers)
        self._human_decks_path, self._pool = human_decks_path, None

    def __enter__(self) -> Self:
        initargs = ((self._hidden_dim, self._attention_layers, self._decision_memory, self._human_decks_path),)
        self._pool = mp.get_context("spawn").Pool(self._num_workers, initializer=_init_batch_worker, initargs=initargs)
        return self

    def __exit__(self, *_: object) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def collect(self, agent: ActorCritic, *, config: BatchedForkConfig, seed: int) -> BatchedForkResult:
        if self._pool is None:
            msg = "BatchedForkCollectorPool must be entered as a context manager before collect()"
            raise RuntimeError(msg)
        if config.forks_per_update <= 0:
            return BatchedForkResult(examples=[], value_samples=[])
        state_np = {key: val.detach().cpu().numpy() for key, val in agent.state_dict().items()}
        base, rem = divmod(config.forks_per_update, self._num_workers)
        tasks = [
            (state_np, base + (1 if rank < rem else 0), config, seed + rank * 7919)
            for rank in range(self._num_workers)
            if base + (1 if rank < rem else 0) > 0
        ]
        parts = self._pool.map(_batch_worker_task, tasks)
        return BatchedForkResult(
            examples=[ex for part in parts for ex in part.examples],
            value_samples=[sample for part in parts for sample in part.value_samples],
        )

"""Parallel counterfactual fork collection and regularized policy updates for Jace v2.

Clones live closed-decklist `managym.Env` game states across spawned CPU workers, forces
the policy's top action against its runner-up alternative across `R` paired playouts
sharing RNG streams (common random numbers, Schulman et al. 2015; Surio et al. 2026),
steps all active branches in one batched policy forward pass per ply to true game
termination (zero learned-judge bias), and updates the policy with bounded pairwise
regularized cross-entropy (Grill et al. 2020; Anthony et al. 2017) plus a reverse-KL
anchor to protect unrelated decisions.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Self

import managym
import numpy as np
import torch
from manabot.backends.torch.arrays import on_device
from manabot.backends.torch.bot import evaluate_observation
from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ModelConfig, ObservationConfig
from managym.enums import ActionSpaceEnum

from jace.eval.deals import Deal, build_deals, elemental_decks
from jace.rl.losses import fork_pairwise_loss, policy_log_probs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from multiprocessing.pool import Pool as ProcessPool

    from torch.optim import Optimizer

__all__ = (
    "ForkCollectorPool",
    "ForkConfig",
    "ForkExample",
    "ForkStats",
    "apply_fork_update",
)

_ACTION_SPACE_NAMES = {int(member): member.name.lower() for member in ActionSpaceEnum}
_MIN_BRANCH_ACTIONS = 2


@dataclass(slots=True)
class _WorkerState:
    agent: ActorCritic | None = None
    obs_space: ObservationSpace | None = None
    decks: list[tuple[str, managym.Deck]] | None = None


_WORKER = _WorkerState()


@dataclass(frozen=True, slots=True)
class ForkConfig:
    forks_per_update: int = 32
    playouts_per_action: int = 4
    decision_kinds: tuple[str, ...] | None = ("declare_attacker",)
    sample_prob: float = 0.35
    num_workers: int = 8
    max_steps: int = 1200
    temperature: float = 0.25
    min_gap: float = 0.25
    fork_weight: float = 0.25
    anchor_kl: float = 0.2
    max_grad_norm: float = 0.5


@dataclass(frozen=True, slots=True)
class ForkExample:
    encoded_obs: dict[str, np.ndarray]
    action_pair: tuple[int, int]
    pair_returns: tuple[float, float]
    policy_pair_probs: tuple[float, float]
    decision_kind: str
    turn: int
    num_actions: int
    valid_pairs: int


@dataclass(frozen=True, slots=True)
class ForkStats:
    total_forks: int
    decisive_forks: int
    alt_better_forks: int
    mean_gap: float
    mean_top_prob: float
    mean_alt_prob: float
    fork_loss: float
    anchor_kl: float
    grad_norm: float
    collect_s: float
    learn_s: float


def _init_fork_worker(hidden_dim: int, attention_layers: int) -> None:
    logging.getLogger("manabot").setLevel(logging.WARNING)
    torch.set_num_threads(1)
    obs_space = ObservationSpace(
        ObservationConfig(policy_input_version=11, max_cards_per_player=80, max_permanents_per_player=64)
    )
    model_cfg = ModelConfig(
        world="world",
        hidden_dim=hidden_dim,
        attention_layers=attention_layers,
        num_attention_heads=4,
        policy_film=True,
        value_kind="categorical_wdl",
        value_aggregation="masked_mean",
    )
    _WORKER.agent = ActorCritic(obs_space, model_cfg, bucketed_shapes=True, live_tokens=False)
    _WORKER.obs_space = obs_space
    _WORKER.decks = elemental_decks()


def _top_two_actions(probs: Sequence[float]) -> tuple[int, int]:
    ranked = sorted(range(len(probs)), key=probs.__getitem__, reverse=True)
    return ranked[0], ranked[1]


def _sample_action(probs: Sequence[float], rng: random.Random) -> int:
    return rng.choices(range(len(probs)), weights=probs, k=1)[0]


def _batched_probs(
    scope: PolicyInference,
    *,
    obs_space: ObservationSpace,
    observations: Sequence[managym.Observation],
) -> list[tuple[float, ...]]:
    max_actions = obs_space.encoder.max_actions
    batch_indices: list[int] = []
    encoded_list: list[Mapping[str, np.ndarray]] = []
    counts: list[int] = []
    results: list[tuple[float, ...] | None] = [None] * len(observations)

    for idx, obs in enumerate(observations):
        count = len(obs.action_space.actions)
        if count <= max_actions:
            batch_indices.append(idx)
            encoded_list.append(obs_space.encode(obs))
            counts.append(count)
        else:
            probs, _ = evaluate_observation(scope, obs_space, obs)
            results[idx] = probs

    if encoded_list:
        stacked = {
            key: np.stack([np.asarray(item[key], dtype=np.float32) for item in encoded_list], axis=0)
            for key in encoded_list[0]
        }
        with torch.no_grad():
            logits, _ = scope.agent(scope.prepare(stacked))
            probs_t = torch.softmax(logits, dim=-1).cpu()
        for row_idx, out_idx in enumerate(batch_indices):
            results[out_idx] = tuple(probs_t[row_idx, : counts[row_idx]].tolist())

    return [item for item in results if item is not None]


def _safe_step(env: managym.Env, action: int) -> tuple[managym.Observation | None, bool, bool]:
    try:
        obs, _, terminated, truncated, _ = env.step(action)
    except (RuntimeError, ValueError, KeyError, IndexError):
        return None, True, True
    return obs, bool(terminated or truncated or env.is_game_over()), False


def _score_env(env: managym.Env, hero_idx: int) -> float:
    winner = env.winner_index()
    if winner is None:
        return 0.5
    return 1.0 if int(winner) == hero_idx else 0.0


def _spawn_paired_branches(
    env: managym.Env,
    *,
    action_pair: tuple[int, int],
    playouts_per_action: int,
    base_seed: int,
) -> tuple[list[managym.Env], list[managym.Observation | None], list[bool], list[bool], list[random.Random]]:
    envs: list[managym.Env] = []
    obs_list: list[managym.Observation | None] = []
    done: list[bool] = []
    failed: list[bool] = []
    rngs: list[random.Random] = []
    for pair_idx in range(playouts_per_action):
        pair_seed = base_seed + pair_idx * 9973
        for root_action in action_pair:
            clone = env.clone_env()
            next_obs, is_done, is_failed = _safe_step(clone, root_action)
            envs.append(clone)
            obs_list.append(next_obs)
            done.append(is_done)
            failed.append(is_failed)
            rngs.append(random.Random(pair_seed))
    return envs, obs_list, done, failed, rngs


def _rollout_paired_branches(
    env: managym.Env,
    *,
    hero_idx: int,
    action_pair: tuple[int, int],
    scope: PolicyInference,
    obs_space: ObservationSpace,
    playouts_per_action: int,
    base_seed: int,
    max_steps: int,
) -> tuple[float, float, int]:
    envs, obs_list, done, failed, rngs = _spawn_paired_branches(
        env,
        action_pair=action_pair,
        playouts_per_action=playouts_per_action,
        base_seed=base_seed,
    )
    for _ in range(max_steps):
        active_pairs = [(idx, ob) for idx, ob in enumerate(obs_list) if not done[idx] and ob is not None]
        if not active_pairs:
            break
        probs_batch = _batched_probs(scope, obs_space=obs_space, observations=[ob for _, ob in active_pairs])
        for (idx, _), probs in zip(active_pairs, probs_batch, strict=True):
            next_obs, is_done, is_failed = _safe_step(envs[idx], _sample_action(probs, rngs[idx]))
            obs_list[idx] = next_obs
            done[idx] = is_done
            failed[idx] = is_failed

    valid = [
        (_score_env(envs[pair_idx * 2], hero_idx), _score_env(envs[pair_idx * 2 + 1], hero_idx))
        for pair_idx in range(playouts_per_action)
        if not failed[pair_idx * 2] and not failed[pair_idx * 2 + 1]
    ]
    if not valid:
        return 0.5, 0.5, 0
    return sum(p[0] for p in valid) / len(valid), sum(p[1] for p in valid) / len(valid), len(valid)


def _fork_decision(
    env: managym.Env,
    obs: managym.Observation,
    *,
    probs: Sequence[float],
    kind: str,
    scope: PolicyInference,
    obs_space: ObservationSpace,
    config: ForkConfig,
    base_seed: int,
) -> ForkExample | None:
    hero_idx = env.current_agent_index()
    if hero_idx is None:
        return None
    a0, a1 = _top_two_actions(probs)
    encoded = {key: np.asarray(val, dtype=np.float32) for key, val in obs_space.encode(obs).items()}
    q0, q1, valid_pairs = _rollout_paired_branches(
        env,
        hero_idx=int(hero_idx),
        action_pair=(a0, a1),
        scope=scope,
        obs_space=obs_space,
        playouts_per_action=config.playouts_per_action,
        base_seed=base_seed,
        max_steps=config.max_steps,
    )
    if valid_pairs == 0:
        return None
    return ForkExample(
        encoded_obs=encoded,
        action_pair=(a0, a1),
        pair_returns=(q0, q1),
        policy_pair_probs=(float(probs[a0]), float(probs[a1])),
        decision_kind=kind,
        turn=int(obs.turn.turn_number),
        num_actions=len(probs),
        valid_pairs=valid_pairs,
    )


def _collect_from_deal(
    deal: Deal,
    *,
    scope: PolicyInference,
    obs_space: ObservationSpace,
    remaining: int,
    allowed: frozenset[str] | None,
    config: ForkConfig,
    rng: random.Random,
    base_seed: int,
) -> list[ForkExample]:
    env = managym.Env(seed=deal.seed, skip_trivial=True)
    obs, _ = env.reset(deal.game_setup())
    collected: list[ForkExample] = []
    for step_idx in range(config.max_steps):
        if env.is_game_over() or len(collected) >= remaining:
            break
        probs, _ = evaluate_observation(scope, obs_space, obs)
        kind = _ACTION_SPACE_NAMES.get(int(obs.action_space.action_space_type), "unknown")
        eligible = _MIN_BRANCH_ACTIONS <= len(probs) <= obs_space.encoder.max_actions and (
            allowed is None or kind in allowed
        )
        if eligible and rng.random() < config.sample_prob:
            example = _fork_decision(
                env,
                obs,
                probs=probs,
                kind=kind,
                scope=scope,
                obs_space=obs_space,
                config=config,
                base_seed=base_seed + step_idx * 31,
            )
            if example is not None:
                collected.append(example)
        next_obs, is_done, is_failed = _safe_step(env, _sample_action(probs, rng))
        if is_done or is_failed or next_obs is None:
            break
        obs = next_obs
    return collected


def _fork_worker_task(args: tuple[dict[str, np.ndarray], int, ForkConfig, int]) -> list[ForkExample]:
    state_np, target_forks, config, seed = args
    if _WORKER.agent is None or _WORKER.obs_space is None or _WORKER.decks is None:
        msg = "fork worker was not initialized"
        raise RuntimeError(msg)
    agent, obs_space, decks = _WORKER.agent, _WORKER.obs_space, _WORKER.decks
    agent.load_state_dict({key: torch.from_numpy(arr) for key, arr in state_np.items()})
    agent.eval()

    rng = random.Random(seed)
    deals = build_deals(decks, count=max(target_forks * 6, 24), seed=seed, decklists="closed")
    allowed = None if config.decision_kinds is None else frozenset(config.decision_kinds)
    examples: list[ForkExample] = []
    with agent.inference_scope() as scope:
        for deal_idx, deal in enumerate(deals):
            if len(examples) >= target_forks:
                break
            examples.extend(
                _collect_from_deal(
                    deal,
                    scope=scope,
                    obs_space=obs_space,
                    remaining=target_forks - len(examples),
                    allowed=allowed,
                    config=config,
                    rng=rng,
                    base_seed=seed * 100_003 + deal_idx * 1009,
                )
            )
    return examples


class ForkCollectorPool:
    """Persistent spawned CPU worker pool for closed-decklist counterfactual forks.

    Initializes one CPU `ActorCritic` and `ObservationSpace` per worker once on entry so
    each update only transfers NumPy weights and gathers paired common-random-number fork
    outcomes in parallel without worker-respawn overhead.
    """

    def __init__(self, *, hidden_dim: int, attention_layers: int, num_workers: int = 8) -> None:
        self._hidden_dim = hidden_dim
        self._attention_layers = attention_layers
        self._num_workers = max(1, num_workers)
        self._pool: ProcessPool | None = None

    def __enter__(self) -> Self:
        ctx = mp.get_context("spawn")
        self._pool = ctx.Pool(
            self._num_workers,
            initializer=_init_fork_worker,
            initargs=(self._hidden_dim, self._attention_layers),
        )
        return self

    def __exit__(self, *_: object) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def collect(self, agent: ActorCritic, *, config: ForkConfig, seed: int) -> list[ForkExample]:
        if self._pool is None:
            msg = "ForkCollectorPool must be entered as a context manager before collect()"
            raise RuntimeError(msg)
        if config.forks_per_update <= 0:
            return []
        state_np = {key: val.detach().cpu().numpy() for key, val in agent.state_dict().items()}
        base, rem = divmod(config.forks_per_update, self._num_workers)
        tasks = [
            (state_np, base + (1 if rank < rem else 0), config, seed + rank * 7919)
            for rank in range(self._num_workers)
            if base + (1 if rank < rem else 0) > 0
        ]
        parts = self._pool.map(_fork_worker_task, tasks)
        return [item for part in parts for item in part]

    def collect_and_update(
        self, agent: ActorCritic, optimizer: Optimizer, *, config: ForkConfig, device: torch.device, seed: int
    ) -> ForkStats:
        t0 = time.perf_counter()
        examples = self.collect(agent, config=config, seed=seed)
        return apply_fork_update(
            agent, optimizer, examples=examples, config=config, device=device, collect_s=time.perf_counter() - t0
        )


def _legal_mask_from_counts(counts: Sequence[int], *, width: int, device: torch.device) -> torch.Tensor:
    arange = torch.arange(width, device=device)[None, :]
    limit = torch.tensor(counts, dtype=torch.long, device=device)[:, None]
    return arange < limit


def apply_fork_update(
    agent: ActorCritic,
    optimizer: Optimizer,
    *,
    examples: Sequence[ForkExample],
    config: ForkConfig,
    device: torch.device,
    collect_s: float = 0.0,
) -> ForkStats:
    """Apply one bounded pairwise counterfactual cross-entropy update on decisive forks.

    Filters `examples` to decisive comparisons (`|q_0 - q_1| >= config.min_gap`), anchors
    full-action logits to the pre-update policy with `config.anchor_kl`, runs one clipped
    optimizer step scaled by `config.fork_weight`, and returns summary fork diagnostics.
    """
    t0 = time.perf_counter()
    total = len(examples)
    if total == 0:
        return ForkStats(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, collect_s, 0.0)

    mean_top_p = sum(item.policy_pair_probs[0] for item in examples) / total
    mean_alt_p = sum(item.policy_pair_probs[1] for item in examples) / total
    decisive = [item for item in examples if abs(item.pair_returns[0] - item.pair_returns[1]) >= config.min_gap]
    alt_better = sum(1 for item in decisive if item.pair_returns[1] > item.pair_returns[0])
    if not decisive:
        return ForkStats(total, 0, 0, 0.0, mean_top_p, mean_alt_p, 0.0, 0.0, 0.0, collect_s, time.perf_counter() - t0)

    stacked = {key: np.stack([item.encoded_obs[key] for item in decisive], axis=0) for key in decisive[0].encoded_obs}
    action_pairs = torch.tensor([item.action_pair for item in decisive], dtype=torch.long, device=device)
    pair_returns = torch.tensor([item.pair_returns for item in decisive], dtype=torch.float32, device=device)
    counts = [item.num_actions for item in decisive]

    agent.to(device)
    prepared = on_device(agent.prepare(stacked), device)
    agent.eval()
    with torch.no_grad():
        old_logits, _ = agent.forward_distribution(prepared)
        legal_mask = _legal_mask_from_counts(counts, width=old_logits.shape[-1], device=device)
        ref_logs = policy_log_probs(old_logits, legal_mask).detach()

    agent.train()
    logits, _ = agent.forward_distribution(prepared)
    breakdown = fork_pairwise_loss(
        logits,
        action_pairs=action_pairs,
        pair_returns=pair_returns,
        legal_mask=legal_mask,
        reference_logs=ref_logs,
        temperature=config.temperature,
        min_gap=config.min_gap,
        anchor_kl=config.anchor_kl,
    )
    optimizer.zero_grad(set_to_none=True)
    (config.fork_weight * breakdown.total).backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm).item())
    optimizer.step()
    if device.type == "mps":
        torch.mps.empty_cache()

    return ForkStats(
        total_forks=total,
        decisive_forks=breakdown.active_count,
        alt_better_forks=alt_better,
        mean_gap=breakdown.mean_gap,
        mean_top_prob=mean_top_p,
        mean_alt_prob=mean_alt_p,
        fork_loss=breakdown.cross_entropy,
        anchor_kl=breakdown.anchor_kl,
        grad_norm=grad_norm,
        collect_s=collect_s,
        learn_s=time.perf_counter() - t0,
    )

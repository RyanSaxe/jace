"""Search-aware Arena player, parallel search match runner, and world-budget sweep CLI.

Lives in `jace.forks` above `jace.eval` so all imports flow strictly downward:
wraps `jace.forks.search.run_local_search` into `SearchPlayer`, runs timed
paired-deal matches across spawned worker processes, delegates ledger recording
to `jace.eval.arena.run_arena_match(..., deal_runner=...)`, and exposes a CLI
(`python -m jace.forks.arena match --search-worlds 8,16,32,64 ...`).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import multiprocessing as mp
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Self

import managym
import torch
from manabot.backends.torch.checkpoint import load_checkpoint

from jace.belief import (
    BayesianDeckPoolPrior,
    BeliefNet,
    FormatPrior,
    fit_format_prior,
    load_belief_net,
    load_format_prior,
)
from jace.collect.decks import elemental_decks, load_human_deck_catalog
from jace.eval.arena import ArenaMatchRecord, SearchConfigSummary, run_arena_match
from jace.eval.deals import Deal, EnsemblePolicyPlayer, PlayerFn, PolicyPlayer
from jace.eval.parallel import PairedDealResult, PlayerSpec, _open_spec_player
from jace.forks.batch import _engine_error_types
from jace.forks.search import SearchConfig, SearchResult, build_belief_from_obs, run_local_search

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace

__all__ = (
    "SearchPlayer",
    "SearchPlayerSpec",
    "fit_arena_format_prior",
    "main",
    "play_search_deals_parallel",
    "resolve_search_player_spec",
    "run_search_arena_match",
)

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchPlayerSpec:
    checkpoint: Path
    search: SearchConfig
    prior: FormatPrior
    belief_net_path: Path | None = None


@dataclass(slots=True)
class _Timer:
    ms: float = 0.0
    count: int = 0


class SearchPlayer:
    def __init__(
        self,
        agent: ActorCritic,
        obs_space: ObservationSpace,
        *,
        prior: FormatPrior,
        config: SearchConfig,
        belief_net: BeliefNet | None = None,
        seed: int = 0,
    ) -> None:
        self._agent, self._obs_space, self._prior = agent, obs_space, prior
        self._config, self._belief_net, self._backend = config, belief_net, BayesianDeckPoolPrior()
        self._seed, self._step, self._scope = seed, 0, None

    def __enter__(self) -> Self:
        self._agent.eval()
        self._scope = self._agent.inference_scope()
        self._scope.__enter__()
        return self

    def __exit__(self, *_: object) -> None:
        if self._scope is not None:
            self._scope.__exit__(None, None, None)
            self._scope = None

    @property
    def needs_decision_memory(self) -> bool:
        return self._obs_space.config.max_decision_records > 0

    def choose(self, env: managym.Env, obs: managym.Observation) -> tuple[int, SearchResult]:
        if self._scope is None:
            msg = "SearchPlayer must be entered as a context manager before choose()"
            raise RuntimeError(msg)
        belief, rev = build_belief_from_obs(
            obs,
            prior=self._prior,
            hero_seat=int(obs.agent.player_index),
            belief_net=self._belief_net,
            prior_only=self._config.prior_only,
            prior_backend=self._backend,
        )
        d_seed, self._step = self._seed + self._step * 1009, self._step + 1
        res = run_local_search(
            env,
            obs,
            scope=self._scope,
            obs_space=self._obs_space,
            belief=belief,
            config=self._config,
            seed=d_seed,
            revealed_counts=rev,
            copy_limits=self._prior.copy_limits,
        )
        return res.chosen_action, res


def fit_arena_format_prior(
    *, format_id: str, decks_path: Path | None = None, deals: Sequence[Deal] = ()
) -> FormatPrior:
    if decks_path is not None:
        if not decks_path.exists():
            msg = f"decks catalog not found: {decks_path}"
            raise FileNotFoundError(msg)
        cat = load_human_deck_catalog(decks_path)
        return fit_format_prior(format_id, cat.card_names, cat.train_decks)
    decks: list[Mapping[str, int]] = (
        [d for deal in deals for d in (deal.deck_a, deal.deck_b)] if deals else [d.main for _, d in elemental_decks()]
    )
    names = tuple(sorted({n for d in decks for n in d}))
    idx = {n: i for i, n in enumerate(names)}
    mat = torch.zeros((len(decks), len(names)), dtype=torch.float32)
    for r, deck in enumerate(decks):
        for n, cnt in deck.items():
            mat[r, idx[n]] = float(cnt)
    return fit_format_prior(format_id, names, mat)


def _resolve_belief(
    runs_dir: Path, *, format_id: str, belief_id: str | None, decks_path: Path | None, deals: Sequence[Deal]
) -> tuple[FormatPrior, Path | None, bool]:
    if belief_id is None or belief_id == "prior-only":
        return fit_arena_format_prior(format_id=format_id, decks_path=decks_path, deals=deals), None, True
    cand = Path(belief_id) if Path(belief_id).exists() else (runs_dir / "belief" / format_id / belief_id)
    if cand.is_dir():
        prior_p, net_p = cand / "prior.pt", cand / "belief_net.pt"
        prior = (
            load_format_prior(prior_p)
            if prior_p.exists()
            else fit_arena_format_prior(format_id=format_id, decks_path=decks_path, deals=deals)
        )
        return prior, (net_p if net_p.exists() else None), not net_p.exists()
    pt_p = cand if cand.exists() else cand.with_suffix(".pt")
    if pt_p.exists():
        return load_format_prior(pt_p), None, True
    msg = f"belief artifact not found for belief_id={belief_id!r} under {runs_dir / 'belief' / format_id}"
    raise FileNotFoundError(msg)


def resolve_search_player_spec(
    spec: PlayerSpec,
    *,
    worlds: int,
    horizon_plies: int | None,
    belief_id: str | None,
    runs_dir: Path,
    format_id: str,
    decks_path: Path | None = None,
    deals: Sequence[Deal] = (),
) -> SearchPlayerSpec:
    if spec in {"heuristic", "random"} or not isinstance(spec, (Path, tuple)):
        msg = f"Search requires a policy checkpoint, not {spec!r}"
        raise ValueError(msg)
    ckpt = spec[0] if isinstance(spec, tuple) else spec
    prior, net_p, prior_only = _resolve_belief(
        runs_dir, format_id=format_id, belief_id=belief_id, decks_path=decks_path, deals=deals
    )
    cfg = SearchConfig(worlds=worlds, horizon_plies=horizon_plies, prior_only=prior_only)
    return SearchPlayerSpec(ckpt, cfg, prior, net_p)


type ArenaPlayerSpec = PlayerSpec | SearchPlayerSpec
type _RuntimePlayer = PlayerFn | SearchPlayer


def _open_player(
    spec: ArenaPlayerSpec, *, seed: int, deterministic: bool
) -> tuple[_RuntimePlayer, PolicyPlayer | EnsemblePolicyPlayer | SearchPlayer | None]:
    if not isinstance(spec, SearchPlayerSpec):
        return _open_spec_player(spec, seed=seed, deterministic=deterministic)
    policy = load_checkpoint(str(spec.checkpoint))[0].eval()
    net = load_belief_net(spec.belief_net_path) if spec.belief_net_path is not None else None
    p = SearchPlayer(policy, policy.observation_space, prior=spec.prior, config=spec.search, belief_net=net, seed=seed)
    return p, p


def _play_leg(
    deal: Deal, *, seat0: _RuntimePlayer, seat1: _RuntimePlayer, t0: _Timer, t1: _Timer, max_actions: int
) -> tuple[int | None, int, int]:
    mem = (PolicyPlayer, EnsemblePolicyPlayer, SearchPlayer)
    use_mem = any(isinstance(p, mem) and p.needs_decision_memory for p in (seat0, seat1))
    env = managym.Env(seed=deal.seed, skip_trivial=True, decision_memory=use_mem)
    decisions, turns, winner = 0, 0, None
    try:
        obs, _ = env.reset(deal.game_setup(swapped=False))
        term = trunc = False
        while not term and not trunc and decisions < max_actions:
            actor, timer = (seat0, t0) if env.current_agent_index() == 0 else (seat1, t1)
            start = time.perf_counter()
            act = actor.choose(env, obs)[0] if isinstance(actor, SearchPlayer) else actor(obs)
            timer.ms, timer.count = timer.ms + (time.perf_counter() - start) * 1000.0, timer.count + 1
            obs, _, term, trunc, _ = env.step(act)
            decisions, turns = decisions + 1, max(turns, int(obs.turn.turn_number))
        winner = env.winner_index() if term else None
    except _engine_error_types() as exc:
        _LOGGER.warning("search eval leg seed=%d aborted (%s): %s", deal.seed, type(exc).__name__, exc)
    return (int(winner) if winner is not None else None), decisions, turns


def _play_deal(
    idx: int, deal: Deal, *, player_a: _RuntimePlayer, player_b: _RuntimePlayer, max_actions: int
) -> PairedDealResult:
    ta, tb = _Timer(), _Timer()
    wins = losses = draws = tot_dec = tot_turns = 0
    for leg, (s0, s1, t0, t1) in enumerate(((player_a, player_b, ta, tb), (player_b, player_a, tb, ta))):
        win, dec, turns = _play_leg(deal, seat0=s0, seat1=s1, t0=t0, t1=t1, max_actions=max_actions)
        tot_dec, tot_turns = tot_dec + dec, tot_turns + turns
        wins += int(win == leg)
        losses += int(win is not None and win != leg)
        draws += int(win is None)
    return PairedDealResult(idx, wins, losses, draws, tot_dec, tot_turns, ta.ms, tb.ms, ta.count, tb.count)


def _search_worker(
    args: tuple[ArenaPlayerSpec, ArenaPlayerSpec, list[tuple[int, Deal]], int, bool, int],
) -> list[PairedDealResult]:
    spec_a, spec_b, indexed, seed, det, max_actions = args
    logging.getLogger("manabot").setLevel(logging.WARNING)
    torch.set_num_threads(1)
    pa, ca = _open_player(spec_a, seed=seed, deterministic=det)
    pb, cb = _open_player(spec_b, seed=seed + 10_000, deterministic=det)
    with contextlib.ExitStack() as stack:
        for ctx in (ca, cb):
            if ctx is not None:
                stack.enter_context(ctx)
        return [_play_deal(i, d, player_a=pa, player_b=pb, max_actions=max_actions) for i, d in indexed]


def play_search_deals_parallel(
    player_a: ArenaPlayerSpec,
    player_b: ArenaPlayerSpec,
    *,
    deals: Sequence[Deal],
    num_workers: int = 8,
    seed: int = 0,
    deterministic: bool = True,
    max_actions: int = 1200,
) -> list[PairedDealResult]:
    if not deals:
        return []
    workers = max(1, min(num_workers, len(deals)))
    shards = [[(i, d) for i, d in enumerate(deals) if i % workers == r] for r in range(workers)]
    tasks = [(player_a, player_b, s, seed + r * 1000, deterministic, max_actions) for r, s in enumerate(shards) if s]
    if len(tasks) == 1:
        combined = _search_worker(tasks[0])
    else:
        with mp.get_context("spawn").Pool(len(tasks)) as pool:
            combined = [item for part in pool.map(_search_worker, tasks) for item in part]
    return sorted(combined, key=lambda item: item.deal_index)


def run_search_arena_match(
    runs_dir: Path,
    *,
    format_id: str,
    model_a: str,
    model_b: str,
    deals: int = 200,
    seed: int = 9901,
    num_workers: int = 8,
    decks_path: Path | None = None,
    deterministic: bool = True,
    max_actions: int = 1200,
    search_config_a: SearchConfigSummary | None = None,
    search_config_b: SearchConfigSummary | None = None,
    model_a_spec: PlayerSpec | None = None,
    model_b_spec: PlayerSpec | None = None,
) -> ArenaMatchRecord:
    resolved_decks = (
        decks_path
        if decks_path is not None
        else (default_p if (default_p := runs_dir / "data" / "decks.pt").exists() else None)
    )

    def _run_deals(spec_a: PlayerSpec, spec_b: PlayerSpec, deal_list: Sequence[Deal]) -> list[PairedDealResult]:  # noqa: PLR0917 -- matches DealRunner callback signature
        arena_a: ArenaPlayerSpec = (
            resolve_search_player_spec(
                spec_a,
                worlds=search_config_a.worlds,
                horizon_plies=search_config_a.horizon,
                belief_id=search_config_a.belief_id,
                runs_dir=runs_dir,
                format_id=format_id,
                decks_path=resolved_decks,
                deals=deal_list,
            )
            if search_config_a is not None
            else spec_a
        )
        arena_b: ArenaPlayerSpec = (
            resolve_search_player_spec(
                spec_b,
                worlds=search_config_b.worlds,
                horizon_plies=search_config_b.horizon,
                belief_id=search_config_b.belief_id,
                runs_dir=runs_dir,
                format_id=format_id,
                decks_path=resolved_decks,
                deals=deal_list,
            )
            if search_config_b is not None
            else spec_b
        )
        return play_search_deals_parallel(
            arena_a,
            arena_b,
            deals=deal_list,
            num_workers=num_workers,
            seed=seed,
            deterministic=deterministic,
            max_actions=max_actions,
        )

    runner = _run_deals if (search_config_a is not None or search_config_b is not None) else None
    return run_arena_match(
        runs_dir,
        format_id=format_id,
        model_a=model_a,
        model_b=model_b,
        deals=deals,
        seed=seed,
        num_workers=num_workers,
        decks_path=decks_path,
        deterministic=deterministic,
        max_actions=max_actions,
        search_config_a=search_config_a,
        search_config_b=search_config_b,
        model_a_spec=model_a_spec,
        model_b_spec=model_b_spec,
        deal_runner=runner,
    )


def _parse_search_configs(args: argparse.Namespace) -> list[SearchConfigSummary | None]:
    if args.no_search or not args.search_worlds:
        return [None]
    belief_id = "prior-only" if args.belief_prior_only else args.belief_id
    worlds_list = [int(part.strip()) for part in str(args.search_worlds).split(",") if part.strip()]
    return [SearchConfigSummary(worlds=worlds, horizon=args.horizon, belief_id=belief_id) for worlds in worlds_list]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jace.forks.arena", description="Search-enabled Arena match sweeps")
    sub = parser.add_subparsers(dest="command", required=True)
    p_match = sub.add_parser("match", help="Run mirrored deals with optional belief-conditioned search")
    p_match.add_argument("--runs-dir", required=True)
    p_match.add_argument("--format", dest="format_id", required=True)
    p_match.add_argument("--model-a", required=True)
    p_match.add_argument("--model-b", required=True)
    p_match.add_argument("--deals", type=int, default=200)
    p_match.add_argument("--seed", type=int, default=9901)
    p_match.add_argument("--num-workers", type=int, default=8)
    p_match.add_argument("--decks-path", default=None)
    p_match.add_argument("--search-worlds", default=None)
    p_match.add_argument("--horizon", type=int, default=None)
    p_match.add_argument("--belief-id", default=None)
    p_match.add_argument("--no-search", action="store_true")
    p_match.add_argument("--belief-prior-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)
    configs = _parse_search_configs(args)
    records = [
        run_search_arena_match(
            Path(args.runs_dir),
            format_id=args.format_id,
            model_a=args.model_a,
            model_b=args.model_b,
            deals=args.deals,
            seed=args.seed,
            num_workers=args.num_workers,
            decks_path=Path(args.decks_path) if args.decks_path else None,
            search_config_a=cfg,
        )
        for cfg in configs
    ]
    payload = [asdict(rec) for rec in records] if len(records) > 1 else asdict(records[0])
    sys.stdout.write(json.dumps(payload, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

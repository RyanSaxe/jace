"""Behavioral decision forensics over live mirrored games against `managym.Heuristic`.

Audits Magic-specific failure modes directly from real game decisions instead of relying
on win rate alone. Tracks wrong-side targeting whenever both friendly and opponent
targets are offered, attack and block selectivity compared to the heuristic, missed
main-phase land drops with an empty stack, and per-decision-kind entropy including
`attack_declare_spread`.
"""

import logging
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import managym
from manabot.backends.torch.policy import ActorCritic
from manabot.env.observation import ObservationSpace
from managym.enums import ActionEnum, ActionSpaceEnum
from managym.recording import ScalarCapture

from jace.eval.deals import Deal, PolicyPlayer, wilson_interval

__all__ = (
    "ForensicsReport",
    "run_forensics",
)

_ACTION_SPACE_NAMES = {int(member): member.name.lower() for member in ActionSpaceEnum}
_MIN_PROB = 1e-12
_LOGGER = logging.getLogger(__name__)


def _forensics_engine_errors() -> tuple[type[BaseException], ...]:
    panics = tuple(cls for cls in BaseException.__subclasses__() if cls.__name__ == "PanicException")
    return (*panics, ValueError, RuntimeError)


@dataclass(frozen=True, slots=True)
class ForensicsReport:
    games: int
    wins: int
    losses: int
    draws: int
    win_rate: float
    ci_low: float
    ci_high: float
    total_decisions: int
    attack_decisions: int
    attack_rate: float
    heuristic_attack_rate: float
    attack_agreement: float
    attack_declare_spread: float
    block_decisions: int
    block_rate: float
    heuristic_block_rate: float
    block_agreement: float
    land_drop_opportunities: int
    missed_land_drops: int
    missed_land_drop_rate: float
    target_decisions: int
    target_both_sides_decisions: int
    wrong_side_targets: int
    wrong_side_target_rate: float
    entropy_by_kind: dict[str, float]
    heuristic_agreement_by_kind: dict[str, float]
    decision_counts_by_kind: dict[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "games": self.games,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "win_rate": round(self.win_rate, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "total_decisions": self.total_decisions,
            "attack_decisions": self.attack_decisions,
            "attack_rate": round(self.attack_rate, 4),
            "heuristic_attack_rate": round(self.heuristic_attack_rate, 4),
            "attack_agreement": round(self.attack_agreement, 4),
            "attack_declare_spread": round(self.attack_declare_spread, 4),
            "block_decisions": self.block_decisions,
            "block_rate": round(self.block_rate, 4),
            "heuristic_block_rate": round(self.heuristic_block_rate, 4),
            "block_agreement": round(self.block_agreement, 4),
            "land_drop_opportunities": self.land_drop_opportunities,
            "missed_land_drops": self.missed_land_drops,
            "missed_land_drop_rate": round(self.missed_land_drop_rate, 4),
            "target_decisions": self.target_decisions,
            "target_both_sides_decisions": self.target_both_sides_decisions,
            "wrong_side_targets": self.wrong_side_targets,
            "wrong_side_target_rate": round(self.wrong_side_target_rate, 4),
            "entropy_by_kind": {k: round(v, 4) for k, v in sorted(self.entropy_by_kind.items())},
            "heuristic_agreement_by_kind": {
                k: round(v, 4) for k, v in sorted(self.heuristic_agreement_by_kind.items())
            },
            "decision_counts_by_kind": dict(sorted(self.decision_counts_by_kind.items())),
        }


@dataclass(slots=True)
class _AuditState:
    total_decisions: int = 0
    attack_decisions: int = 0
    attack_chosen: int = 0
    heuristic_attack_chosen: int = 0
    attack_agreed: int = 0
    attack_entropy_sum: float = 0.0
    block_decisions: int = 0
    block_chosen: int = 0
    heuristic_block_chosen: int = 0
    block_agreed: int = 0
    land_drop_opportunities: int = 0
    missed_land_drops: int = 0
    target_decisions: int = 0
    target_both_sides: int = 0
    wrong_side_targets: int = 0
    kind_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    kind_entropy: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    kind_agree: dict[str, int] = field(default_factory=lambda: defaultdict(int))


def _entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log(max(p, _MIN_PROB)) for p in probs if p > 0.0)


def _target_side(obs: managym.Observation, action_index: int) -> str | None:
    action = obs.action_space.actions[action_index]
    if not action.focus:
        return None
    target_id = int(action.focus[0])
    friendly_ids = {int(obs.agent.id)} | {int(p.id) for p in obs.agent_permanents}
    opponent_ids = {int(obs.opponent.id)} | {int(p.id) for p in obs.opponent_permanents}
    if target_id in friendly_ids:
        return "friendly"
    if target_id in opponent_ids:
        return "opponent"
    return None


def _audit_attack(state: _AuditState, *, chosen_declared: bool, heur_declared: bool, agreed: bool, ent: float) -> None:
    state.attack_decisions += 1
    state.attack_entropy_sum += ent
    state.attack_chosen += int(chosen_declared)
    state.heuristic_attack_chosen += int(heur_declared)
    state.attack_agreed += int(agreed)


def _audit_block(state: _AuditState, *, chosen_declared: bool, heur_declared: bool, agreed: bool) -> None:
    state.block_decisions += 1
    state.block_chosen += int(chosen_declared)
    state.heuristic_block_chosen += int(heur_declared)
    state.block_agreed += int(agreed)


def _audit_land_drop(state: _AuditState, obs: managym.Observation, *, action: int) -> None:
    if obs.stack_objects:
        return
    actions = obs.action_space.actions
    if not any(int(a.action_type) == int(ActionEnum.PRIORITY_PLAY_LAND) for a in actions):
        return
    state.land_drop_opportunities += 1
    state.missed_land_drops += int(int(actions[action].action_type) == int(ActionEnum.PRIORITY_PASS_PRIORITY))


def _audit_target(state: _AuditState, obs: managym.Observation, *, action: int, heur_action: int) -> None:
    state.target_decisions += 1
    sides = {_target_side(obs, idx) for idx in range(len(obs.action_space.actions))}
    if not ({"friendly", "opponent"} <= sides):
        return
    state.target_both_sides += 1
    chosen_side = _target_side(obs, action)
    heur_side = _target_side(obs, heur_action)
    if chosen_side is not None and heur_side is not None and chosen_side != heur_side:
        state.wrong_side_targets += 1


def _record_decision(
    state: _AuditState,
    obs: managym.Observation,
    *,
    action: int,
    probs: Sequence[float],
    heur_action: int,
) -> None:
    actions = obs.action_space.actions
    if len(actions) <= 1:
        return
    space_type = int(obs.action_space.action_space_type)
    kind = _ACTION_SPACE_NAMES.get(space_type, f"kind_{space_type}")
    ent = _entropy(probs)
    agreed = action == heur_action

    state.total_decisions += 1
    state.kind_counts[kind] += 1
    state.kind_entropy[kind] += ent
    state.kind_agree[kind] += int(agreed)

    chosen_declared = bool(actions[action].declared)
    heur_declared = bool(actions[heur_action].declared)
    if space_type == int(ActionSpaceEnum.DECLARE_ATTACKER):
        _audit_attack(state, chosen_declared=chosen_declared, heur_declared=heur_declared, agreed=agreed, ent=ent)
    elif space_type == int(ActionSpaceEnum.DECLARE_BLOCKER):
        _audit_block(state, chosen_declared=chosen_declared, heur_declared=heur_declared, agreed=agreed)
    elif space_type == int(ActionSpaceEnum.PRIORITY):
        _audit_land_drop(state, obs, action=action)
    elif space_type == int(ActionSpaceEnum.CHOOSE_TARGET):
        _audit_target(state, obs, action=action, heur_action=heur_action)


def _play_audited_leg(
    deal: Deal,
    *,
    leg: int,
    player: PolicyPlayer,
    heuristic: managym.Heuristic,
    state: _AuditState,
    games_path: Path | None,
    max_actions: int,
) -> int | None:
    setup = deal.game_setup(swapped=False)
    use_memory = player.needs_decision_memory
    env = managym.Env(seed=deal.seed, skip_trivial=True, decision_memory=use_memory)
    names = ["agent", "heuristic"] if leg == 0 else ["heuristic", "agent"]
    capture = (
        ScalarCapture(
            seed=deal.seed,
            skip_trivial=True,
            decision_memory=use_memory,
            games_path=games_path,
            record_fraction=1.0,
            context={"players": names},
        )
        if games_path is not None
        else None
    )
    winner: int | None = None
    try:
        obs, _ = capture.reset(env, setup) if capture is not None else env.reset(setup)
        terminal = False
        truncated = False
        steps = 0
        while not terminal and not truncated and steps < max_actions:
            seat = env.current_agent_index()
            if seat == leg:
                action, probs, _ = player.evaluate(obs)
                heur_action = int(heuristic.select_action(obs))
                _record_decision(state, obs, action=action, probs=probs, heur_action=heur_action)
            else:
                action = int(heuristic.select_action(obs))
            obs, _, terminal, truncated, _ = capture.step(env, action) if capture is not None else env.step(action)
            steps += 1
        winner = env.winner_index() if terminal else None
    except _forensics_engine_errors() as exc:
        if isinstance(exc, ValueError) and "Observation capacity exceeded" not in str(exc):
            raise
        _LOGGER.warning("forensics leg seed=%d aborted (%s): %s", deal.seed, type(exc).__name__, exc)
        winner = None
    finally:
        if capture is not None:
            capture.close()
    return int(winner) if winner is not None else None


def _ratio(num: float, denom: int) -> float:
    return num / denom if denom > 0 else 0.0


def run_forensics(
    agent: ActorCritic,
    obs_space: ObservationSpace,
    *,
    deals: Sequence[Deal],
    seed: int = 0,
    deterministic: bool = False,
    games_path: Path | None = None,
    max_actions: int = 1200,
) -> ForensicsReport:
    """Audit `agent` across mirrored games against `managym.Heuristic`.

    Plays both legs of every deal (so 20 deals = 40 games), measuring win rate
    alongside combat selectivity, missed land drops, wrong-side targeting, and
    per-kind entropy. Writes native `managym` recordings when `games_path` is given.
    """
    state = _AuditState()
    heuristic = managym.Heuristic()
    wins = 0
    losses = 0
    draws = 0

    with PolicyPlayer(agent, obs_space, seed=seed, deterministic=deterministic) as player:
        for deal in deals:
            for leg in (0, 1):
                winner = _play_audited_leg(
                    deal,
                    leg=leg,
                    player=player,
                    heuristic=heuristic,
                    state=state,
                    games_path=games_path,
                    max_actions=max_actions,
                )
                if winner is None:
                    draws += 1
                elif winner == leg:
                    wins += 1
                else:
                    losses += 1

    games = wins + losses + draws
    score_points = wins + 0.5 * draws
    ci_low, ci_high = wilson_interval(score_points, games)
    return ForensicsReport(
        games=games,
        wins=wins,
        losses=losses,
        draws=draws,
        win_rate=_ratio(score_points, games),
        ci_low=ci_low,
        ci_high=ci_high,
        total_decisions=state.total_decisions,
        attack_decisions=state.attack_decisions,
        attack_rate=_ratio(state.attack_chosen, state.attack_decisions),
        heuristic_attack_rate=_ratio(state.heuristic_attack_chosen, state.attack_decisions),
        attack_agreement=_ratio(state.attack_agreed, state.attack_decisions),
        attack_declare_spread=_ratio(state.attack_entropy_sum, state.attack_decisions),
        block_decisions=state.block_decisions,
        block_rate=_ratio(state.block_chosen, state.block_decisions),
        heuristic_block_rate=_ratio(state.heuristic_block_chosen, state.block_decisions),
        block_agreement=_ratio(state.block_agreed, state.block_decisions),
        land_drop_opportunities=state.land_drop_opportunities,
        missed_land_drops=state.missed_land_drops,
        missed_land_drop_rate=_ratio(state.missed_land_drops, state.land_drop_opportunities),
        target_decisions=state.target_decisions,
        target_both_sides_decisions=state.target_both_sides,
        wrong_side_targets=state.wrong_side_targets,
        wrong_side_target_rate=_ratio(state.wrong_side_targets, state.target_both_sides),
        entropy_by_kind={k: state.kind_entropy[k] / n for k, n in state.kind_counts.items()},
        heuristic_agreement_by_kind={k: state.kind_agree[k] / n for k, n in state.kind_counts.items()},
        decision_counts_by_kind=dict(state.kind_counts),
    )

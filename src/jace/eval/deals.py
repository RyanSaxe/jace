"""Paired-deal mirrored evaluation matches with native Academy game recording.

Every deal is played twice with seats swapped (leg 0: player A on the play with deck A;
leg 1: player B on the play with deck A) so deal RNG and play/draw advantage cancel out.
When `games_path` is provided, every completed game is written through `managym`'s
native `ScalarCapture` so it opens directly in `uv run managym academy replay`.
"""

import logging
import math
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import managym
from manabot.backends.torch.bot import evaluate_observation
from manabot.backends.torch.policy import ActorCritic, PolicyInference
from manabot.env.observation import ObservationSpace
from managym.disclosure import DecklistDisclosure
from managym.recording import ScalarCapture

__all__ = (
    "Deal",
    "MatchScore",
    "PlayerFn",
    "PolicyPlayer",
    "build_deals",
    "elemental_decks",
    "heuristic_player",
    "play_match",
    "random_player",
    "wilson_interval",
)

_WILSON_Z = 1.959963984540054
_LOGGER = logging.getLogger(__name__)


def _eval_engine_errors() -> tuple[type[BaseException], ...]:
    panics = tuple(cls for cls in BaseException.__subclasses__() if cls.__name__ == "PanicException")
    return (*panics, ValueError, RuntimeError)


@dataclass(frozen=True, slots=True)
class Deal:
    seed: int
    deck_a_name: str
    deck_b_name: str
    deck_a: dict[str, int]
    deck_b: dict[str, int]
    sideboard_a: dict[str, int]
    sideboard_b: dict[str, int]
    decklists: DecklistDisclosure = "closed"

    def game_setup(self, *, swapped: bool = False) -> managym.GameSetup:
        first = managym.PlayerConfig(
            self.deck_b_name if swapped else self.deck_a_name,
            dict(self.deck_b if swapped else self.deck_a),
            dict(self.sideboard_b if swapped else self.sideboard_a),
            managym.WORLD_SETUP,
        )
        second = managym.PlayerConfig(
            self.deck_a_name if swapped else self.deck_b_name,
            dict(self.deck_a if swapped else self.deck_b),
            dict(self.sideboard_a if swapped else self.sideboard_b),
            managym.WORLD_SETUP,
        )
        return managym.GameSetup([first, second], decklists=self.decklists)


@dataclass(frozen=True, slots=True)
class MatchScore:
    wins: int
    losses: int
    draws: int
    games: int
    win_rate: float
    ci_low: float
    ci_high: float
    paired_diff: float
    mean_decisions: float
    mean_turns: float

    def to_dict(self) -> dict[str, int | float]:
        return {
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "games": self.games,
            "win_rate": round(self.win_rate, 4),
            "ci_low": round(self.ci_low, 4),
            "ci_high": round(self.ci_high, 4),
            "paired_diff": round(self.paired_diff, 4),
            "mean_decisions": round(self.mean_decisions, 1),
            "mean_turns": round(self.mean_turns, 2),
        }


type PlayerFn = Callable[[managym.Observation], int]


class PolicyPlayer:
    """Inference-scoped `ActorCritic` player that caches the card catalog across games.

    Use as a context manager (`with PolicyPlayer(...) as player:`) so the policy's
    catalog cache is released cleanly once evaluation finishes. Exposes `evaluate`
    for callers that also inspect action probabilities and value estimates.
    """

    def __init__(
        self,
        agent: ActorCritic,
        obs_space: ObservationSpace,
        *,
        seed: int = 0,
        deterministic: bool = False,
    ) -> None:
        self._agent = agent
        self._obs_space = obs_space
        self._rng = random.Random(seed)
        self._deterministic = deterministic
        self._scope: PolicyInference | None = None

    def __enter__(self) -> Self:
        self._agent.eval()
        self._scope = self._agent.inference_scope()
        self._scope.__enter__()
        return self

    def __exit__(self, *_: object) -> None:
        if self._scope is not None:
            self._scope.__exit__(None, None, None)
            self._scope = None

    def evaluate(self, obs: managym.Observation) -> tuple[int, tuple[float, ...], float]:
        if self._scope is None:
            msg = "PolicyPlayer must be entered as a context manager before evaluation"
            raise RuntimeError(msg)
        probs, value = evaluate_observation(self._scope, self._obs_space, obs)
        if self._deterministic:
            action = max(range(len(probs)), key=probs.__getitem__)
        else:
            action = self._rng.choices(range(len(probs)), weights=probs, k=1)[0]
        return action, probs, value

    def __call__(self, obs: managym.Observation) -> int:
        action, _, _ = self.evaluate(obs)
        return action


def heuristic_player() -> PlayerFn:
    native = managym.Heuristic()

    def choose(obs: managym.Observation) -> int:
        choice = int(native.select_action(obs))
        if choice < 0 or choice >= len(obs.action_space.actions):
            msg = f"heuristic selected out-of-range action {choice}"
            raise RuntimeError(msg)
        return choice

    return choose


def random_player(*, seed: int = 0) -> PlayerFn:
    rng = random.Random(seed)

    def choose(obs: managym.Observation) -> int:
        count = len(obs.action_space.actions)
        if count <= 0:
            msg = "no legal actions available"
            raise RuntimeError(msg)
        return rng.randrange(count)

    return choose


def elemental_decks() -> list[tuple[str, managym.Deck]]:
    pack = "elemental-world-v2"
    result: list[tuple[str, managym.Deck]] = []
    for key in managym.authored_deck_keys(pack):
        setup = managym.authored_deck_setup(pack, key)
        result.append((key, managym.Deck(dict(setup.decklist), dict(setup.sideboard))))
    return result


def build_deals(
    decks: Sequence[tuple[str, managym.Deck]],
    *,
    count: int,
    seed: int = 1,
    decklists: DecklistDisclosure = "closed",
) -> list[Deal]:
    if not decks or count < 1:
        msg = "build_deals requires at least one deck and a positive count"
        raise ValueError(msg)
    pairs = [(i, j) for i in range(len(decks)) for j in range(len(decks)) if i != j or len(decks) == 1]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    deals: list[Deal] = []
    for index in range(count):
        i, j = pairs[index % len(pairs)]
        name_a, deck_a = decks[i]
        name_b, deck_b = decks[j]
        deals.append(
            Deal(
                seed=seed + index * 1009,
                deck_a_name=name_a,
                deck_b_name=name_b,
                deck_a=dict(deck_a.main),
                deck_b=dict(deck_b.main),
                sideboard_a=dict(deck_a.sideboard),
                sideboard_b=dict(deck_b.sideboard),
                decklists=decklists,
            )
        )
    return deals


def wilson_interval(wins: float, total: int) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = wins / total
    z2 = _WILSON_Z * _WILSON_Z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    half = (_WILSON_Z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _play_leg(
    deal: Deal,
    *,
    seat0: PlayerFn,
    seat1: PlayerFn,
    games_path: Path | None,
    player_names: tuple[str, str],
    max_actions: int,
) -> tuple[int | None, int, int]:
    setup = deal.game_setup(swapped=False)
    env = managym.Env(seed=deal.seed, skip_trivial=True)
    capture = (
        ScalarCapture(
            seed=deal.seed,
            skip_trivial=True,
            games_path=games_path,
            record_fraction=1.0,
            context={"players": list(player_names)},
        )
        if games_path is not None
        else None
    )
    decisions = 0
    turns = 0
    winner: int | None = None
    try:
        obs, _ = capture.reset(env, setup) if capture is not None else env.reset(setup)
        terminal = False
        truncated = False
        while not terminal and not truncated and decisions < max_actions:
            seat = env.current_agent_index()
            action = seat0(obs) if seat == 0 else seat1(obs)
            obs, _, terminal, truncated, _ = capture.step(env, action) if capture is not None else env.step(action)
            decisions += 1
            turns = max(turns, int(obs.turn.turn_number))
        winner = env.winner_index() if terminal else None
    except _eval_engine_errors() as exc:
        if isinstance(exc, ValueError) and "Observation capacity exceeded" not in str(exc):
            raise
        _LOGGER.warning("eval leg seed=%d aborted (%s): %s", deal.seed, type(exc).__name__, exc)
        winner = None
    finally:
        if capture is not None:
            capture.close()
    return (int(winner) if winner is not None else None), decisions, turns


def play_match(
    player_a: PlayerFn,
    player_b: PlayerFn,
    *,
    deals: Sequence[Deal],
    games_path: Path | None = None,
    player_names: tuple[str, str] = ("agent", "opponent"),
    max_actions: int = 1200,
) -> MatchScore:
    wins = 0
    losses = 0
    draws = 0
    total_decisions = 0
    total_turns = 0
    paired_sum = 0.0
    name_a, name_b = player_names

    for deal in deals:
        deal_score = 0.0
        for leg in (0, 1):
            seat0, seat1 = (player_a, player_b) if leg == 0 else (player_b, player_a)
            names = (name_a, name_b) if leg == 0 else (name_b, name_a)
            winner, decisions, turns = _play_leg(
                deal,
                seat0=seat0,
                seat1=seat1,
                games_path=games_path,
                player_names=names,
                max_actions=max_actions,
            )
            total_decisions += decisions
            total_turns += turns
            if winner is None:
                draws += 1
            elif winner == leg:
                wins += 1
                deal_score += 0.5
            else:
                losses += 1
                deal_score -= 0.5
        paired_sum += deal_score

    games = wins + losses + draws
    score_points = wins + 0.5 * draws
    win_rate = score_points / games if games else 0.0
    ci_low, ci_high = wilson_interval(score_points, games)
    return MatchScore(
        wins=wins,
        losses=losses,
        draws=draws,
        games=games,
        win_rate=win_rate,
        ci_low=ci_low,
        ci_high=ci_high,
        paired_diff=paired_sum / len(deals) if deals else 0.0,
        mean_decisions=total_decisions / games if games else 0.0,
        mean_turns=total_turns / games if games else 0.0,
    )

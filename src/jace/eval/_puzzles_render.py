"""Board summary, formatting, and locked JSONL serialization for tactical puzzles.

Defines the tactical puzzle dataclasses (`PuzzleRecord`, `PuzzleVerificationResult`,
`PuzzleScoreRow`), locked JSONL persistence under `<runs_dir>/puzzles/{format_id}.jsonl`,
and observation-to-record formatting helpers (`build_puzzle_record`).
"""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

from managym.enums import ActionEnum, ActionSpaceEnum

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import managym

__all__ = (
    "PuzzleRecord",
    "PuzzleScoreRow",
    "PuzzleSource",
    "PuzzleVerificationResult",
    "append_puzzle",
    "build_puzzle_record",
    "load_puzzles",
    "puzzles_path",
)

type PuzzleSource = Literal["policy_disagreement", "search_disagreement", "high_swing_loss", "watchlist_promotion"]

_SOURCE_MAP: dict[str, PuzzleSource] = {
    "policy_disagreement": "policy_disagreement",
    "search_disagreement": "search_disagreement",
    "high_swing_loss": "high_swing_loss",
    "watchlist_promotion": "watchlist_promotion",
}


def _as_num(value: object) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"expected numeric value, got {value!r}"
        raise TypeError(msg)
    return value


def _parse_puzzle_source(raw: str) -> PuzzleSource:
    if raw in _SOURCE_MAP:
        return _SOURCE_MAP[raw]
    msg = f"unsupported puzzle source {raw!r}; expected one of {sorted(_SOURCE_MAP)}"
    raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PuzzleRecord:
    puzzle_id: str
    format_id: str
    source: PuzzleSource
    run_id: str
    iteration: int
    game_idx: int
    step: int
    turn: int
    phase: str
    step_name: str
    decision_kind: str
    hero_seat: int
    seed: int
    hero_summary: str
    opp_summary: str
    key_cards: tuple[str, ...]
    root_value_v: float
    best_action_index: int
    best_action_label: str
    alt_action_index: int
    alt_action_label: str
    best_playout_wr: float
    alt_playout_wr: float
    playouts_per_action: int
    max_plies: int
    margin: float
    paired_ci_low: float
    paired_ci_high: float

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["key_cards"] = list(self.key_cards)
        return data

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> PuzzleRecord:
        cards = raw.get("key_cards", ())
        key_cards = tuple(str(c) for c in cards) if isinstance(cards, (list, tuple)) else ()
        return cls(
            puzzle_id=str(raw["puzzle_id"]),
            format_id=str(raw["format_id"]),
            source=_parse_puzzle_source(str(raw["source"])),
            run_id=str(raw["run_id"]),
            iteration=int(_as_num(raw["iteration"])),
            game_idx=int(_as_num(raw["game_idx"])),
            step=int(_as_num(raw["step"])),
            turn=int(_as_num(raw["turn"])),
            phase=str(raw["phase"]),
            step_name=str(raw["step_name"]),
            decision_kind=str(raw["decision_kind"]),
            hero_seat=int(_as_num(raw["hero_seat"])),
            seed=int(_as_num(raw["seed"])),
            hero_summary=str(raw["hero_summary"]),
            opp_summary=str(raw["opp_summary"]),
            key_cards=key_cards,
            root_value_v=float(_as_num(raw["root_value_v"])),
            best_action_index=int(_as_num(raw["best_action_index"])),
            best_action_label=str(raw["best_action_label"]),
            alt_action_index=int(_as_num(raw["alt_action_index"])),
            alt_action_label=str(raw["alt_action_label"]),
            best_playout_wr=float(_as_num(raw["best_playout_wr"])),
            alt_playout_wr=float(_as_num(raw["alt_playout_wr"])),
            playouts_per_action=int(_as_num(raw["playouts_per_action"])),
            max_plies=int(_as_num(raw["max_plies"])),
            margin=float(_as_num(raw["margin"])),
            paired_ci_low=float(_as_num(raw["paired_ci_low"])),
            paired_ci_high=float(_as_num(raw["paired_ci_high"])),
        )


@dataclass(frozen=True, slots=True)
class PuzzleVerificationResult:
    admitted: bool
    best_action: int
    alt_action: int
    best_playout_wr: float
    alt_playout_wr: float
    playouts_per_action: int
    valid_pairs: int
    max_plies: int
    margin: float
    paired_se: float
    paired_ci_low: float
    paired_ci_high: float
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PuzzleScoreRow:
    puzzle_id: str
    format_id: str
    source: PuzzleSource
    solved: bool
    chosen_action_index: int
    best_action_index: int
    best_action_prob: float
    alt_action_prob: float
    root_value_v: float
    margin: float


def puzzles_path(runs_dir: Path, format_id: str) -> Path:
    if not format_id or "/" in format_id or "\\" in format_id or ".." in format_id:
        msg = f"invalid format_id for puzzle store: {format_id!r}"
        raise ValueError(msg)
    return runs_dir / "puzzles" / f"{format_id}.jsonl"


def load_puzzles(runs_dir: Path, format_id: str) -> list[PuzzleRecord]:
    path = puzzles_path(runs_dir, format_id)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [
            PuzzleRecord.from_dict(parsed)
            for raw_line in handle
            if (line := raw_line.strip()) and isinstance(parsed := json.loads(line), Mapping)
        ]


def append_puzzle(runs_dir: Path, record: PuzzleRecord) -> Path:
    """Append one verified `PuzzleRecord` under an exclusive `fcntl.LOCK_EX` file lock.

    Deduplicates by `record.puzzle_id` so repeated mining or promotion retries stay
    idempotent, then flushes and `os.fsync`s before releasing the lock.
    """
    path = puzzles_path(runs_dir, record.format_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_dict(), sort_keys=True) + "\n"
    with path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            handle.seek(0)
            existing = {
                parsed["puzzle_id"]
                for raw_line in handle
                if (s := raw_line.strip())
                and isinstance(parsed := json.loads(s), Mapping)
                and isinstance(parsed.get("puzzle_id"), str)
            }
            if record.puzzle_id not in existing:
                handle.seek(0, os.SEEK_END)
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return path


def _format_action_label(action: managym.Action) -> str:
    if action.description:
        return str(action.description)
    kind = ActionEnum(int(action.action_type)).name.replace("_", " ").title()
    if int(action.action_type) in (int(ActionEnum.DECLARE_ATTACKER), int(ActionEnum.DECLARE_BLOCKER)):
        return f"{kind}: {'yes' if action.declared is True else 'no'}"
    return kind


def _summarize_side(player: managym.Player, perms: Sequence[managym.Permanent]) -> str:
    creatures = sum(1 for p in perms if p.characteristics.card_types.is_creature)
    lands = sum(
        1 for p in perms if p.characteristics.card_types.is_land and not p.characteristics.card_types.is_creature
    )
    hand_size = int(player.zone_counts[1]) if len(player.zone_counts) > 1 else 0
    return f"life={int(player.life)} hand={hand_size} lands={lands} creatures={creatures}"


def _extract_key_cards(obs: managym.Observation, *, limit: int = 6) -> tuple[str, ...]:
    names: list[str] = []
    for perm in (*obs.agent_permanents, *obs.opponent_permanents):
        name = str(perm.characteristics.name)
        if not perm.characteristics.card_types.is_land and name not in names:
            names.append(name)
            if len(names) >= limit:
                break
    return tuple(names)


def build_puzzle_record(
    obs: managym.Observation,
    verification: PuzzleVerificationResult,
    *,
    puzzle_id: str,
    format_id: str,
    source: PuzzleSource,
    run_id: str,
    iteration: int,
    game_idx: int,
    step: int,
    hero_seat: int,
    seed: int,
    root_value_v: float,
    key_cards: Sequence[str] = (),
) -> PuzzleRecord:
    actions = obs.action_space.actions
    acting_is_hero = int(obs.agent.player_index) == hero_seat
    hero_pack = (obs.agent, obs.agent_permanents) if acting_is_hero else (obs.opponent, obs.opponent_permanents)
    opp_pack = (obs.opponent, obs.opponent_permanents) if acting_is_hero else (obs.agent, obs.agent_permanents)
    space_kind = ActionSpaceEnum(int(obs.action_space.action_space_type)).name
    turn = int(obs.turn.turn_number)
    is_pregame = turn == 0 or space_kind == "MULLIGAN"
    return PuzzleRecord(
        puzzle_id,
        format_id,
        source,
        run_id,
        iteration,
        game_idx,
        step,
        turn,
        "Pregame" if is_pregame else str(obs.turn.phase).split(".")[-1],
        "Mulligan" if is_pregame else str(obs.turn.step).split(".")[-1],
        space_kind,
        hero_seat,
        seed,
        _summarize_side(*hero_pack),
        _summarize_side(*opp_pack),
        tuple(key_cards) if key_cards else _extract_key_cards(obs),
        round(root_value_v, 4),
        verification.best_action,
        _format_action_label(actions[verification.best_action]),
        verification.alt_action,
        _format_action_label(actions[verification.alt_action]),
        verification.best_playout_wr,
        verification.alt_playout_wr,
        verification.playouts_per_action,
        verification.max_plies,
        verification.margin,
        verification.paired_ci_low,
        verification.paired_ci_high,
    )

"""Run one booster-draft pod as an explicit state machine.

A pod is always in exactly one of two states. In `Picking`, every seat holds a
pack and owes one pick. In `Complete`, the draft is over and only the pools
remain. `advance` is the only transition, and it takes every seat's pick at
once because seats pick simultaneously at a real table. States are immutable
values, so a caller can keep, compare or replay any point of a draft. This
module does not generate boosters and does not choose cards.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from jace.cards import CardId, Pool

type Pack = tuple[CardId, ...]


class InvalidDraftError(ValueError):
    """The format, the boosters or a set of picks does not describe a legal draft.

    It is raised before a new state is built, so the state the caller holds
    is still valid. The message names the round and seat at fault when there
    is one. It subclasses `ValueError` so callers that validate input in bulk
    can catch both.
    """


class IllegalPickError(InvalidDraftError):
    def __init__(self, *, seat: int, pick: CardId, pack: Pack) -> None:
        super().__init__(f"seat {seat} picked card {pick}, which is not in its pack {pack}")


@dataclass(frozen=True, slots=True)
class DraftFormat:
    """The shape of a draft: how many seats, rounds, and picks in each round.

    A round is one booster per seat. Seats may take fewer cards than a booster
    holds, and the cards left over at the end of a round are discarded. A pod
    of eight drafting three 14-card boosters to the last card is
    `DraftFormat(seats=8, rounds=3, picks_per_round=14)`.
    """

    seats: int
    rounds: int
    picks_per_round: int

    def __post_init__(self) -> None:
        if min(self.seats, self.rounds, self.picks_per_round) < 1:
            message = f"every field of a draft format must be positive: {self}"
            raise InvalidDraftError(message)


@dataclass(frozen=True, slots=True)
class Picking:
    """Every seat holds a pack and owes one pick.

    `packs[seat]` is the pack in front of that seat and `pools[seat]` is what
    the seat has taken so far. `unopened[0][seat]` is the booster that seat
    opens when the next round starts.
    """

    draft_format: DraftFormat
    round_index: int
    pick_index: int
    packs: tuple[Pack, ...]
    pools: tuple[Pool, ...]
    unopened: tuple[tuple[Pack, ...], ...]


@dataclass(frozen=True, slots=True)
class Complete:
    draft_format: DraftFormat
    pools: tuple[Pool, ...]


type PodState = Picking | Complete


def open_pod(draft_format: DraftFormat, boosters: Sequence[Sequence[Pack]]) -> Picking:
    """Seat the pod and open the first round.

    `boosters[round][seat]` is the booster that seat opens in that round. The
    cards in a booster keep their order, and a card may appear more than once.

    Raises `InvalidDraftError` when the boosters do not match the format: the wrong
    number of rounds or seats, or a booster too small for a round's picks.
    """
    if len(boosters) != draft_format.rounds:
        message = f"expected {draft_format.rounds} rounds of boosters, got {len(boosters)}"
        raise InvalidDraftError(message)
    for round_index, round_boosters in enumerate(boosters):
        if len(round_boosters) != draft_format.seats:
            message = f"round {round_index} has {len(round_boosters)} boosters for {draft_format.seats} seats"
            raise InvalidDraftError(message)
        for seat, booster in enumerate(round_boosters):
            if len(booster) < draft_format.picks_per_round:
                message = (
                    f"round {round_index} seat {seat}: a booster of {len(booster)} cards "
                    f"cannot supply {draft_format.picks_per_round} picks"
                )
                raise InvalidDraftError(message)
    rounds = tuple(tuple(tuple(booster) for booster in round_boosters) for round_boosters in boosters)
    return Picking(
        draft_format=draft_format,
        round_index=0,
        pick_index=0,
        packs=rounds[0],
        pools=tuple(() for _ in range(draft_format.seats)),
        unopened=rounds[1:],
    )


def advance(state: Picking, picks: Sequence[CardId]) -> PodState:
    """Apply one pick from every seat, where `picks[seat]` is that seat's card.

    The result is `Picking` again after the packs are passed or the next round
    is opened, and `Complete` after the last pick of the last round. Packs
    pass to the next seat in even rounds and to the previous seat in odd
    rounds. Raises `IllegalPickError` when a seat picks a card that is not in
    its pack, and `InvalidDraftError` when the number of picks differs from the
    number of seats.
    """
    draft_format = state.draft_format
    if len(picks) != draft_format.seats:
        message = f"expected one pick from each of {draft_format.seats} seats, got {len(picks)}"
        raise InvalidDraftError(message)
    pools = tuple((*pool, pick) for pool, pick in zip(state.pools, picks, strict=True))
    remaining = tuple(
        _without(pack, pick, seat=seat) for seat, (pack, pick) in enumerate(zip(state.packs, picks, strict=True))
    )
    if state.pick_index + 1 < draft_format.picks_per_round:
        return Picking(
            draft_format=draft_format,
            round_index=state.round_index,
            pick_index=state.pick_index + 1,
            packs=_passed(remaining, round_index=state.round_index),
            pools=pools,
            unopened=state.unopened,
        )
    if state.unopened:
        return Picking(
            draft_format=draft_format,
            round_index=state.round_index + 1,
            pick_index=0,
            packs=state.unopened[0],
            pools=pools,
            unopened=state.unopened[1:],
        )
    return Complete(draft_format=draft_format, pools=pools)


def _without(pack: Pack, pick: CardId, *, seat: int) -> Pack:
    if pick not in pack:
        raise IllegalPickError(seat=seat, pick=pick, pack=pack)
    position = pack.index(pick)
    return pack[:position] + pack[position + 1 :]


def _passed(packs: tuple[Pack, ...], *, round_index: int) -> tuple[Pack, ...]:
    step = 1 if round_index % 2 == 0 else -1
    seats = len(packs)
    return tuple(packs[(seat - step) % seats] for seat in range(seats))


__all__ = (
    "Complete",
    "DraftFormat",
    "IllegalPickError",
    "InvalidDraftError",
    "Pack",
    "Picking",
    "PodState",
    "advance",
    "open_pod",
)

"""Run whole drafts for many pods at once and record what each seat saw.

A pod keeps no history, so `run_drafts` builds each seat's `SeatView` as the
draft goes and returns every view with the final pools. All pods pick in
step, and each distinct drafter is called once per pick with the views of
every seat it fills in every pod, so a model picks for all of them in one
forward pass. Two seats share a drafter only when they hold the same object.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from jace.cards import CardId, Pool
from jace.draft.pod import DraftFormat, InvalidDraftError, Pack, Picking, PodState, advance, open_pod

__all__ = ("DraftRecord", "Drafter", "SeatView", "run_drafts")


@dataclass(frozen=True, slots=True)
class SeatView:
    """Everything one seat knows when it owes a pick.

    `seen[i]` is the pack the seat held at pick `i`, before it took
    `picks[i]` from it. Both have `pick_number` entries, and they run across
    rounds. A seat never sees another seat's picks. It can infer them only
    from the cards missing when a pack comes back around.
    """

    pick_number: int
    pack: Pack
    seen: tuple[Pack, ...]
    picks: tuple[CardId, ...]


# Returns one card for each view, in the order of the views. One call picks for many seats.
type Drafter = Callable[[Sequence[SeatView]], Sequence[CardId]]


@dataclass(frozen=True, slots=True)
class DraftRecord:
    # Indexed by seat, then by pick number.
    views: tuple[tuple[SeatView, ...], ...]
    pools: tuple[Pool, ...]


def run_drafts(
    draft_format: DraftFormat, boosters: Sequence[Sequence[Sequence[Pack]]], *, drafters: Sequence[Drafter]
) -> tuple[DraftRecord, ...]:
    """Draft every pod to the end and return one record per pod.

    `boosters[pod][round][seat]` is the booster that seat opens in that round,
    and `drafters[seat]` picks for that seat in every pod. A drafter that
    fills several seats gets its views ordered by pod and then by seat.

    Raises `InvalidDraftError` when the number of drafters differs from the
    number of seats, when a pod's boosters do not match the format, or when a
    drafter returns a number of picks other than its number of views. Raises
    `IllegalPickError` when a drafter picks a card outside the seat's pack.
    """
    if len(drafters) != draft_format.seats:
        message = f"expected one drafter for each of {draft_format.seats} seats, got {len(drafters)}"
        raise InvalidDraftError(message)
    seat_groups = _seats_by_drafter(drafters)
    states: tuple[PodState, ...] = tuple(open_pod(draft_format, pod_boosters) for pod_boosters in boosters)
    # Indexed by pod, then seat, then pick number.
    views: list[list[list[SeatView]]] = [[[] for _ in drafters] for _ in states]
    while picking := tuple(state for state in states if isinstance(state, Picking)):
        for pod_views, state in zip(views, picking, strict=True):
            _add_views(pod_views, state)
        picks = _ask_drafters(drafters, seat_groups, views=views)
        states = tuple(advance(state, pod_picks) for state, pod_picks in zip(picking, picks, strict=True))
    return tuple(
        DraftRecord(views=tuple(tuple(seat_views) for seat_views in pod_views), pools=state.pools)
        for pod_views, state in zip(views, states, strict=True)
    )


def _seats_by_drafter(drafters: Sequence[Drafter]) -> list[list[int]]:
    # One group of seats for each distinct drafter object, in the order of each drafter's first seat.
    groups: dict[int, list[int]] = {}
    for seat, drafter in enumerate(drafters):
        groups.setdefault(id(drafter), []).append(seat)
    return list(groups.values())


def _add_views(pod_views: Sequence[list[SeatView]], state: Picking) -> None:
    # Appends to each seat's views what the seat knows at the pick it now owes.
    for seat_views, pack, pool in zip(pod_views, state.packs, state.pools, strict=True):
        seen = tuple(view.pack for view in seat_views)
        seat_views.append(SeatView(pick_number=len(seat_views), pack=pack, seen=seen, picks=pool))


def _ask_drafters(
    drafters: Sequence[Drafter], seat_groups: Sequence[Sequence[int]], *, views: Sequence[Sequence[Sequence[SeatView]]]
) -> list[list[CardId]]:
    # Calls each distinct drafter once with the newest view of every seat it fills in every pod.
    # Returns the picks indexed by pod, then seat.
    picks: list[dict[int, CardId]] = [{} for _ in views]
    for seats in seat_groups:
        asked = [(pod, seat) for pod in range(len(views)) for seat in seats]
        chosen = drafters[seats[0]]([views[pod][seat][-1] for pod, seat in asked])
        if len(chosen) != len(asked):
            message = f"the drafter of seats {seats} returned {len(chosen)} picks for {len(asked)} views"
            raise InvalidDraftError(message)
        for (pod, seat), card in zip(asked, chosen, strict=True):
            picks[pod][seat] = card
    return [[pod_picks[seat] for seat in range(len(drafters))] for pod_picks in picks]

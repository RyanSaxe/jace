"""Check the draft pod's transitions through its public states.

The tests drive whole small drafts and single picks through `open_pod` and
`advance` only. They never look at private helpers, so the pod's internals
can change without touching them.
"""

import pytest

from mtg_rl.draft import (
    Complete,
    DraftFormat,
    IllegalPickError,
    InvalidDraftError,
    Picking,
    PodState,
    advance,
    open_pod,
)

__all__ = ()


def _first_card_draft(state: Picking) -> Complete:
    current: PodState = state
    while isinstance(current, Picking):
        current = advance(current, [pack[0] for pack in current.packs])
    return current


def test_whole_draft_gives_every_seat_one_card_per_pick() -> None:
    draft_format = DraftFormat(seats=3, rounds=2, picks_per_round=2)
    boosters = [
        [(1, 2), (3, 4), (5, 6)],
        [(7, 8), (9, 10), (11, 12)],
    ]

    finished = _first_card_draft(open_pod(draft_format, boosters))

    assert [len(pool) for pool in finished.pools] == [4, 4, 4]
    assert sorted(card for pool in finished.pools for card in pool) == list(range(1, 13))


def test_packs_pass_to_the_next_seat_in_the_first_round() -> None:
    draft_format = DraftFormat(seats=3, rounds=1, picks_per_round=2)
    state = open_pod(draft_format, [[(1, 2), (3, 4), (5, 6)]])

    after = advance(state, [1, 3, 5])

    assert isinstance(after, Picking)
    assert after.packs == ((6,), (2,), (4,))


def test_packs_pass_to_the_previous_seat_in_the_second_round() -> None:
    draft_format = DraftFormat(seats=3, rounds=2, picks_per_round=2)
    boosters = [
        [(1, 2), (3, 4), (5, 6)],
        [(7, 8), (9, 10), (11, 12)],
    ]
    state = open_pod(draft_format, boosters)
    state = advance(state, [1, 3, 5])
    assert isinstance(state, Picking)
    state = advance(state, [6, 2, 4])
    assert isinstance(state, Picking)
    assert (state.round_index, state.pick_index) == (1, 0)

    after = advance(state, [7, 9, 11])

    assert isinstance(after, Picking)
    assert after.packs == ((10,), (12,), (8,))


def test_cards_left_in_a_booster_after_the_round_are_discarded() -> None:
    draft_format = DraftFormat(seats=2, rounds=1, picks_per_round=1)

    finished = advance(open_pod(draft_format, [[(1, 2), (3, 4)]]), [2, 3])

    assert finished == Complete(draft_format=draft_format, pools=((2,), (3,)))


def test_a_pick_outside_the_seats_pack_is_rejected() -> None:
    state = open_pod(DraftFormat(seats=2, rounds=1, picks_per_round=1), [[(1, 2), (3, 4)]])

    with pytest.raises(IllegalPickError, match="seat 0 picked card 3"):
        advance(state, [3, 4])


def test_boosters_that_do_not_match_the_format_are_rejected() -> None:
    with pytest.raises(InvalidDraftError, match="1 boosters for 2 seats"):
        open_pod(DraftFormat(seats=2, rounds=1, picks_per_round=1), [[(1, 2)]])

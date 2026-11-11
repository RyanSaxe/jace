"""Check the manabot adapter by playing real games through `GameBatch`.

Every test runs the real engine on two 40-card decks, 17 basic lands and 23
creatures of one colour each. The tests read only the adapter's public names
and the arrays a caller receives. Two tests name board arrays, `decklists` and
`semantic_cards`, because what a seat may see is the behaviour under test.
"""

import random

import numpy as np
import numpy.typing as npt
import pytest
import torch

from jace.cards import CardPool, CardRecord, Deck, Rarity
from jace.engine import Decisions, Finished
from jace.engine.manabot import DeckRejectedError, engine_card_ids, engine_cards, start_batch

__all__ = ()

_WHITE_CREATURES = ("Avatar Enthusiasts", "Compassionate Healer", "Curious Farm Animals", "Glider Kids")
_RED_CREATURES = ("Fire Nation Cadets", "Fire Nation Raider", "Fire Sages", "Firebending Student")
_OPENING_HAND = 7
_DECK_SIZE = 40


def _engine_pool() -> CardPool:
    # The engine has no rarity. Nothing here reads it, so every card is filed as common.
    records = [
        CardRecord(name=card.name, set_code=card.set_code, rarity=Rarity.COMMON, text=card.text, program=card.program)
        for card in engine_cards()
    ]
    return CardPool(records)


def _deck(pool: CardPool, *, land: str, creatures: tuple[str, ...]) -> Deck:
    spells = [pool.card_id(creatures[copy % len(creatures)]) for copy in range(23)]
    return (*[pool.card_id(land)] * 17, *spells)


def _matchup(pool: CardPool) -> tuple[Deck, Deck]:
    white = _deck(pool, land="Plains", creatures=_WHITE_CREATURES)
    red = _deck(pool, land="Mountain", creatures=_RED_CREATURES)
    return white, red


def _random_actions(decisions: Decisions, rng: random.Random) -> npt.NDArray[np.int64]:
    actions = [rng.randrange(count) for count in decisions.action_counts.tolist()]
    return np.array(actions, dtype=np.int64)


def test_two_batches_with_one_seed_and_the_same_actions_play_identically() -> None:
    pool = _engine_pool()
    matchup = _matchup(pool)
    first = start_batch(lambda: matchup, pool=pool, games=2, seed=11)
    second = start_batch(lambda: matchup, pool=pool, games=2, seed=11)
    rng = random.Random(0)
    finished_games = 0

    for _ in range(400):
        decisions, repeated = first.decisions(), second.decisions()
        assert decisions.boards.keys() == repeated.boards.keys()
        for name, board in decisions.boards.items():
            assert np.array_equal(board, repeated.boards[name]), name
        actions = _random_actions(decisions, rng)
        finished = first.act(actions)
        assert finished == second.act(actions)
        finished_games += len(finished)

    # The comparison must reach past a game's end, where each slot draws a new seed.
    assert finished_games > 0


def test_a_finished_game_reports_the_decks_it_was_played_with_and_its_slot_deals_a_new_game() -> None:
    pool = _engine_pool()
    red = _deck(pool, land="Mountain", creatures=_RED_CREATURES)
    issued: list[tuple[Deck, Deck]] = []

    def distinct_matchups() -> tuple[Deck, Deck]:
        # Rotating the creature list changes the copy counts, so no two issued matchups are equal.
        turn = len(issued) % len(_WHITE_CREATURES)
        white = _deck(pool, land="Plains", creatures=_WHITE_CREATURES[turn:] + _WHITE_CREATURES[:turn])
        issued.append((white, red))
        return issued[-1]

    batch = start_batch(distinct_matchups, pool=pool, games=2, seed=5)
    in_play = list(issued)
    rng = random.Random(1)
    finished: tuple[Finished, ...] = ()
    while not finished:
        finished = batch.act(_random_actions(batch.decisions(), rng))

    game = finished[0]
    assert game.decks == in_play[game.slot]
    hidden = batch.hidden_cards()
    assert hidden.hands[game.slot].sum(axis=-1).tolist() == [_OPENING_HAND, _OPENING_HAND]
    next_decks = issued[len(in_play)]
    for seat in (0, 1):
        dealt = hidden.hands[game.slot, seat] + hidden.libraries[game.slot, seat]
        assert np.array_equal(dealt, np.bincount(next_decks[seat], minlength=len(pool)))


def test_hidden_hand_equals_the_hand_the_acting_seat_sees_at_its_first_decision() -> None:
    pool = _engine_pool()
    matchup = _matchup(pool)
    batch = start_batch(lambda: matchup, pool=pool, games=3, seed=2)
    decisions, hidden = batch.decisions(), batch.hidden_cards()
    card_ids = engine_card_ids(pool)
    # Slots share one matchup, so equal hands would mean they were dealt from one seed.
    assert not np.array_equal(hidden.hands[0], hidden.hands[1])

    for slot, seat in enumerate(decisions.seats.tolist()):
        # Before any card leaves a hand, the acting seat's own card rows are exactly its hand.
        definitions = torch.from_numpy(decisions.boards["semantic_cards"][slot, 0]).long()
        seen = card_ids[definitions[definitions > 0] - 1]
        assert np.array_equal(np.bincount(seen.numpy(), minlength=len(pool)), hidden.hands[slot, seat])


def test_hand_plus_library_never_exceeds_the_deck_during_random_play() -> None:
    pool = _engine_pool()
    matchup = _matchup(pool)
    deck_counts = np.stack([np.bincount(deck, minlength=len(pool)) for deck in matchup])
    batch = start_batch(lambda: matchup, pool=pool, games=2, seed=3)
    rng = random.Random(2)
    cards_out_of_library = 0

    for _ in range(120):
        batch.act(_random_actions(batch.decisions(), rng))
        hidden = batch.hidden_cards()
        assert (hidden.hands + hidden.libraries <= deck_counts).all()
        assert (hidden.libraries >= 0).all()
        cards_out_of_library = max(cards_out_of_library, _DECK_SIZE - int(hidden.libraries[0, 0].sum()))

    # Cards were drawn and played, so the bound was tested on more than the opening deal.
    assert cards_out_of_library > _OPENING_HAND


def test_boards_show_the_acting_seats_decklist_and_never_the_opponents() -> None:
    pool = _engine_pool()
    matchup = _matchup(pool)
    batch = start_batch(lambda: matchup, pool=pool, games=2, seed=7)
    rng = random.Random(3)

    for _ in range(60):
        decisions = batch.decisions()
        decklists = decisions.boards["decklists"]
        assert decklists[:, 0].any(axis=(1, 2)).all()
        assert not decklists[:, 1].any()
        batch.act(_random_actions(decisions, rng))


def test_a_deck_with_a_card_the_engine_lacks_is_rejected_by_name() -> None:
    lotus = CardRecord(name="Black Lotus", set_code="LEA", rarity=Rarity.RARE, text="", program=())
    plains = CardRecord(name="Plains", set_code="LEA", rarity=Rarity.BASIC, text="", program=())
    pool = CardPool([plains, lotus])
    unknown: Deck = (pool.card_id("Black Lotus"),) * _DECK_SIZE

    with pytest.raises(DeckRejectedError, match="Black Lotus"):
        start_batch(lambda: (unknown, unknown), pool=pool, games=1, seed=0)

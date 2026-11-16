"""Check the manabot adapter by playing real games through `GameBatch`.

The tests play real games on the engine with 40-card decks, each made of 17
basic lands and 23 creatures of one colour. Only the last test plays no
game, because `start_batch` rejects its deck before any game starts. The
tests read only the adapter's public names and the arrays a caller
receives. Two tests name board arrays, `decklists` and `semantic_cards`,
because what a seat may see is the behaviour under test.
"""

import random

import numpy as np
import numpy.typing as npt
import pytest
import torch

from jace.cards import CardPool, CardRecord, Deck, Rarity
from jace.engine import Decisions, Finished, Outcome
from jace.engine.manabot import DeckRejectedError, engine_card_ids, engine_cards, start_batch
from jace.play import BOARD_ARRAYS

__all__ = ()

_WHITE_CREATURES = ("Avatar Enthusiasts", "Compassionate Healer", "Curious Farm Animals", "Glider Kids")
_RED_CREATURES = ("Fire Nation Cadets", "Fire Nation Raider", "Fire Sages", "Firebending Student")
_OPENING_HAND = 7
_DECK_SIZE = 40


def _engine_pool() -> CardPool:
    # The engine reports no rarity for its cards. No test here reads rarity, so every card is given common rarity.
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
    first = start_batch([matchup] * 4, pool=pool, games=2, seed=11, boards=BOARD_ARRAYS)
    second = start_batch([matchup] * 4, pool=pool, games=2, seed=11, boards=BOARD_ARRAYS)
    rng = random.Random(0)
    finished_games = 0

    for _ in range(400):
        decisions, repeated = first.decisions(), second.decisions()
        assert np.array_equal(decisions.games, repeated.games)
        assert decisions.boards.keys() == repeated.boards.keys()
        for name, board in decisions.boards.items():
            assert np.array_equal(board, repeated.boards[name]), name
        actions = _random_actions(decisions, rng)
        finished = first.act(actions)
        assert finished == second.act(actions)
        finished_games += len(finished)

    # The two batches must also give the same arrays after a game ends, when the next game of the list starts in its
    # place on a new seed. So at least one game must have ended.
    assert finished_games > 0


def test_the_arrays_of_a_decision_keep_their_values_after_later_steps() -> None:
    # `jace.train.play_games` keeps every step's arrays without copying them, so no later step may write into
    # them.
    pool = _engine_pool()
    batch = start_batch([_matchup(pool)] * 2, pool=pool, games=2, seed=3, boards=BOARD_ARRAYS)
    rng = random.Random(2)
    kept = batch.decisions()
    arrays = {"games": kept.games, "seats": kept.seats, **kept.boards}
    copies = {name: array.copy() for name, array in arrays.items()}

    decisions = kept
    for _ in range(50):
        batch.act(_random_actions(decisions, rng))
        decisions = batch.decisions()

    for name, array in arrays.items():
        assert np.array_equal(array, copies[name]), name


def test_a_batch_plays_exactly_its_matchups_each_with_its_own_decks_and_then_holds_no_game() -> None:
    pool = _engine_pool()
    red = _deck(pool, land="Mountain", creatures=_RED_CREATURES)
    # Rotating the creature list changes the copy counts, so no two matchups are equal.
    matchups = [
        (_deck(pool, land="Plains", creatures=_WHITE_CREATURES[turn:] + _WHITE_CREATURES[:turn]), red)
        for turn in range(len(_WHITE_CREATURES))
    ]
    batch = start_batch(matchups, pool=pool, games=2, seed=5, boards=BOARD_ARRAYS)
    rng = random.Random(1)
    finished: list[Finished] = []

    decisions = batch.decisions()
    while len(decisions.games):
        assert len(decisions.games) <= 2
        ended = batch.act(_random_actions(decisions, rng))
        decisions = batch.decisions()
        if ended and not finished and len(decisions.games) == 2:
            # When a game ends, the engine first deals a new game from the old decks in its place, and the batch
            # then deals the next matchup over it. The first game to end makes room for game 2, whose hands and
            # libraries must together be exactly the decks of `matchups[2]`.
            row = int(decisions.games.argmax())
            hidden = batch.hidden_cards()
            for seat in (0, 1):
                dealt = hidden.hands[row, seat] + hidden.libraries[row, seat]
                assert np.array_equal(dealt, np.bincount(matchups[2][seat], minlength=len(pool)))
        finished += ended

    assert sorted(game.game for game in finished) == list(range(len(matchups)))
    for game in finished:
        # Every game must end with a winner. A game the engine failed on or cut short would end as abandoned.
        assert game.outcome in {Outcome.SEAT_0_WON, Outcome.SEAT_1_WON}
        assert game.decks == matchups[game.game]


def test_hidden_hand_equals_the_hand_the_acting_seat_sees_at_its_first_decision() -> None:
    pool = _engine_pool()
    matchup = _matchup(pool)
    batch = start_batch([matchup] * 3, pool=pool, games=3, seed=2, boards=BOARD_ARRAYS)
    decisions, hidden = batch.decisions(), batch.hidden_cards()
    card_ids = engine_card_ids(pool)
    # The games share one matchup, so equal hands would mean they were dealt from one seed.
    assert not np.array_equal(hidden.hands[0], hidden.hands[1])

    for slot, seat in enumerate(decisions.seats.tolist()):
        # At a game's first decision no card has left a hand, so the acting seat's own rows of
        # `semantic_cards` list exactly its hand.
        definitions = torch.from_numpy(decisions.boards["semantic_cards"][slot, 0]).long()
        seen = card_ids[definitions[definitions > 0] - 1]
        assert np.array_equal(np.bincount(seen.numpy(), minlength=len(pool)), hidden.hands[slot, seat])


def test_hand_plus_library_never_exceeds_the_deck_during_random_play() -> None:
    pool = _engine_pool()
    matchup = _matchup(pool)
    deck_counts = np.stack([np.bincount(deck, minlength=len(pool)) for deck in matchup])
    batch = start_batch([matchup] * 2, pool=pool, games=2, seed=3, boards=BOARD_ARRAYS)
    rng = random.Random(2)
    cards_out_of_library = 0

    for _ in range(120):
        batch.act(_random_actions(batch.decisions(), rng))
        hidden = batch.hidden_cards()
        assert (hidden.hands + hidden.libraries <= deck_counts).all()
        assert (hidden.libraries >= 0).all()
        cards_out_of_library = max(cards_out_of_library, _DECK_SIZE - int(hidden.libraries[0, 0].sum()))

    # More cards left seat 0's library in the first game than the opening hand takes, so the bound was checked
    # after cards were drawn and played, not only on the opening deal.
    assert cards_out_of_library > _OPENING_HAND


def test_boards_show_the_acting_seats_decklist_and_never_the_opponents() -> None:
    pool = _engine_pool()
    matchup = _matchup(pool)
    batch = start_batch([matchup] * 2, pool=pool, games=2, seed=7, boards=("decklists",))
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
        start_batch([(unknown, unknown)], pool=pool, games=1, seed=0, boards=BOARD_ARRAYS)

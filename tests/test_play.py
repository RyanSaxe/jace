"""Check the guess model, the player model and the win-chance model on real engine boards.

The boards in `tests/data/manabot_v10_boards.npz` are three decisions from
real games, encoded by the engine adapter at input version 10, each with
permanents, events and the acting seat's decklist on the board. No
permanent on them blocks or is attached to anything, so the test of those
links adds one. On every board the opponent has cards in hand that it has
not revealed. The guess model's meta is fitted from two 40-card decks onto
a format whose boosters hold the first 400 cards, so the cards from 400 on
are not legal. The models are built from the layout the adapter reports,
and the tests call them only through `jace.play`, so the token layout and
the network can change without touching them.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from jace.cards import CardTable
from jace.engine.manabot import board_layout
from jace.play import (
    Boards,
    Earlier,
    Guess,
    GuessModel,
    Meta,
    PlayerModel,
    WinChanceModel,
    day_zero_meta,
    fit_meta,
    random_policy,
)

__all__ = ()

ENGINE_DEFINITIONS = 480
CARDS = 480
MORE_CARDS = 520
CARD_WIDTH = 16
LEGAL_CARDS = 400
DECK_SIZE = 40
# Board 0's opponent shows five copies of card 267 and one of card 302, and board 1's four of card 213.
META_DECKS = (
    ((267, 5), (302, 2), *((card, 1) for card in range(33))),
    ((213, 6), *((card, 1) for card in range(100, 134))),
)
NO_EARLIER_GUESSES = Earlier(
    guesses=Guess(deck=torch.zeros(3, 0, CARDS), hand=torch.zeros(3, 0, CARDS)), mask=torch.ones(3, 0, dtype=torch.bool)
)


def _real_boards() -> dict[str, Tensor]:
    with np.load(Path(__file__).parent / "data" / "manabot_v10_boards.npz") as arrays:
        return {name: torch.from_numpy(arrays[name]) for name in arrays.files}


def _card_table(cards: int) -> CardTable:
    # The first `cards` rows of one fixed table, so a larger pool keeps the smaller pool's vectors.
    starting = torch.randn(MORE_CARDS, CARD_WIDTH, generator=torch.Generator().manual_seed(0))
    return CardTable(starting[:cards])


def _meta(cards: int) -> Meta:
    booster_copies = torch.zeros(cards)
    booster_copies[:LEGAL_CARDS] = 1.0
    copies = torch.zeros(len(META_DECKS), cards)
    for deck, held in enumerate(META_DECKS):
        for card, count in held:
            copies[deck, card] += count
    return fit_meta(copies, fallback=day_zero_meta(booster_copies, deck_size=DECK_SIZE))


def _models(cards: int) -> tuple[PlayerModel, GuessModel]:
    torch.manual_seed(0)
    embedding = _card_table(cards)
    engine_to_card = torch.arange(ENGINE_DEFINITIONS)
    player = PlayerModel(
        embedding,
        engine_to_card=engine_to_card,
        layout=board_layout(),
        width=64,
        layers=2,
        heads=4,
        guess_cards=8,
        deck_guess_cards=8,
    )
    guesser = GuessModel(embedding, engine_to_card=engine_to_card, layout=board_layout(), width=64, layers=2, heads=4)
    guesser.set_meta(_meta(cards))
    return player.eval(), guesser.eval()


def _win_chance_model(cards: int) -> WinChanceModel:
    torch.manual_seed(0)
    engine_to_card = torch.arange(ENGINE_DEFINITIONS)
    model = WinChanceModel(
        _card_table(cards), engine_to_card=engine_to_card, layout=board_layout(), width=64, layers=2, heads=4
    )
    return model.eval()


def _hidden_cards(*, in_hand: range) -> Guess:
    # The opponent holds one copy of each of the first twenty cards unseen, those of `in_hand` in hand.
    truth = Guess(deck=torch.zeros(3, CARDS), hand=torch.zeros(3, CARDS))
    truth.deck[:, :20] = 1 / 20
    truth.hand[:, in_hand] = 1.0
    return truth


def _move_weights(guesser: GuessModel, *, seed: int) -> None:
    # Moves each weight a little at random, as training would, so the corrections are not zero. The card
    # table has no weights, so only the model's own move.
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in guesser.parameters():
            parameter.add_(torch.randn(parameter.shape, generator=generator) * 0.1)


def _some_guess(shape: tuple[int, ...], *, seed: int) -> Guess:
    generator = torch.Generator().manual_seed(seed)
    return Guess(
        deck=torch.rand(shape, generator=generator).softmax(dim=-1),
        hand=torch.rand(shape, generator=generator),
    )


def _with_padding_overwritten(boards: Boards) -> dict[str, Tensor]:
    changed = {name: rows.clone() for name, rows in boards.items()}
    for name, valid in boards.items():
        if name.endswith("_valid"):
            changed[name.removesuffix("_valid")][valid == 0] = 7
    return changed


def test_player_gives_no_probability_to_illegal_actions_on_real_boards() -> None:
    boards = _real_boards()
    player, _ = _models(CARDS)

    with torch.no_grad():
        logits = player.policy(boards, _some_guess((3, CARDS), seed=1))

    legal = boards["actions_valid"] > 0
    probabilities = logits.softmax(dim=-1)
    assert legal.sum(dim=-1).tolist() == [5, 4, 2]
    assert torch.equal(probabilities > 0, legal)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3))


def test_player_reads_both_the_hand_guess_and_the_deck_guess() -> None:
    boards = _real_boards()
    player, _ = _models(CARDS)
    guess = _some_guess((3, CARDS), seed=1)
    other = _some_guess((3, CARDS), seed=2)

    with torch.no_grad():
        logits = player.policy(boards, guess)
        with_other_hand = player.policy(boards, Guess(deck=guess.deck, hand=other.hand))
        with_other_deck = player.policy(boards, Guess(deck=other.deck, hand=guess.hand))

    legal = boards["actions_valid"] > 0
    assert not torch.allclose(logits[legal], with_other_hand[legal])
    assert not torch.allclose(logits[legal], with_other_deck[legal])


def _action_odds(player: PlayerModel, boards: Boards) -> Tensor:
    with torch.no_grad():
        logits = player.policy(boards, _some_guess((3, CARDS), seed=1))
    return logits.softmax(dim=-1)


def test_player_reads_which_object_a_permanent_blocks_or_is_attached_to() -> None:
    # A link names an object by its index plus one in the engine's order: both players, the acting
    # seat's cards, the opponent's cards, then the acting seat's permanents. The acting seat's
    # permanents 0 and 3 on board 0 are different cards.
    boards = _real_boards()
    player, _ = _models(CARDS)
    first_permanent = 2 + boards["agent_cards"].shape[1] + boards["opponent_cards"].shape[1]

    for link in board_layout().permanent_links:
        odds = []
        for linked_permanent in (0, 3):
            linked = {name: rows.clone() for name, rows in boards.items()}
            linked["opponent_permanents"][0, 0, link] = first_permanent + linked_permanent + 1
            odds.append(_action_odds(player, linked)[0])
        assert not torch.allclose(odds[0], odds[1])


@pytest.mark.parametrize(("row", "unseen_after"), [(1, 0), (11, 8)], ids=["last copy", "one of nine"])
def test_player_reads_the_copies_left_unseen_in_its_own_deck(row: int, unseen_after: int) -> None:
    # Board 0's decklist row 1 is a card with one unseen copy, and row 11 a land with nine.
    boards = _real_boards()
    player, _ = _models(CARDS)
    unseen, _ = board_layout().decklist_copies
    drawn = {name: rows.clone() for name, rows in boards.items()}
    drawn["decklists"][0, 0, row, unseen] = unseen_after

    assert not torch.allclose(_action_odds(player, boards)[0], _action_odds(player, drawn)[0])


def test_random_policy_spreads_probability_evenly_over_legal_actions() -> None:
    boards = _real_boards()
    logits = random_policy(boards, _some_guess((3, CARDS), seed=1))

    probabilities = logits.softmax(dim=-1)
    assert torch.allclose(probabilities[:, 0], torch.tensor([1 / 5, 1 / 4, 1 / 2]))
    assert torch.equal(probabilities > 0, boards["actions_valid"] > 0)


def test_win_chance_reads_which_cards_the_opponent_truly_holds_in_hand() -> None:
    # The same board and the same unseen cards, with different ones of them in the opponent's hand.
    boards = _real_boards()
    model = _win_chance_model(CARDS)

    with torch.no_grad():
        chance = model.win_chance(boards, _hidden_cards(in_hand=range(7)))
        other_hand_chance = model.win_chance(boards, _hidden_cards(in_hand=range(10, 17)))

    assert chance.shape == (3,)
    assert bool(((chance > 0) & (chance < 1)).all())
    assert not torch.allclose(chance, other_hand_chance)


@pytest.mark.parametrize("earlier_guesses", [0, 3])
def test_guess_is_a_deck_distribution_and_hand_probabilities(earlier_guesses: int) -> None:
    boards = _real_boards()
    _, guesser = _models(CARDS)
    mask = torch.ones(3, earlier_guesses, dtype=torch.bool)
    earlier = Earlier(guesses=_some_guess((3, earlier_guesses, CARDS), seed=2), mask=mask)

    with torch.no_grad():
        guess = guesser.guess(boards, earlier)

    assert guess.deck.shape == guess.hand.shape == (3, CARDS)
    assert torch.allclose(guess.deck.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert bool((guess.deck >= 0).all())
    assert bool(((guess.hand >= 0) & (guess.hand <= 1)).all())


def test_padding_rows_do_not_change_the_policy_the_guess_or_the_win_chance() -> None:
    boards = _real_boards()
    padded_boards = _with_padding_overwritten(boards)
    player, guesser = _models(CARDS)
    win_chance_model = _win_chance_model(CARDS)
    truth = _hidden_cards(in_hand=range(7))
    guesses = _some_guess((3, 2, CARDS), seed=2)
    junk = _some_guess((3, 1, CARDS), seed=3)
    earlier = Earlier(guesses=guesses, mask=torch.ones(3, 2, dtype=torch.bool))
    padded_earlier = Earlier(
        guesses=Guess(
            deck=torch.cat((guesses.deck, junk.deck), dim=1), hand=torch.cat((guesses.hand, junk.hand), dim=1)
        ),
        mask=torch.tensor([[True, True, False]]).expand(3, -1),
    )

    with torch.no_grad():
        guess = guesser.guess(boards, earlier)
        padded_guess = guesser.guess(padded_boards, padded_earlier)
        logits = player.policy(boards, guess)
        padded_logits = player.policy(padded_boards, guess)
        win_chance = win_chance_model.win_chance(boards, truth)
        padded_win_chance = win_chance_model.win_chance(padded_boards, truth)

    assert torch.allclose(padded_guess.deck, guess.deck, atol=1e-5)
    assert torch.allclose(padded_guess.hand, guess.hand, atol=1e-5)
    assert torch.allclose(padded_logits, logits, atol=1e-4)
    assert torch.allclose(padded_win_chance, win_chance, atol=1e-5)


def test_models_score_a_larger_pool_and_old_cards_keep_their_scores() -> None:
    boards = _real_boards()
    player, guesser = _models(CARDS)
    larger_player, larger_guesser = _models(CARDS)
    larger_player.embedding = larger_guesser.embedding = _models(MORE_CARDS)[0].embedding
    win_chance_model, larger_win_chance_model = _win_chance_model(CARDS), _win_chance_model(CARDS)
    larger_win_chance_model.embedding = larger_player.embedding
    truth = _hidden_cards(in_hand=range(7))
    no_new_cards = Guess(
        deck=torch.cat((truth.deck, torch.zeros(3, MORE_CARDS - CARDS)), dim=-1),
        hand=torch.cat((truth.hand, torch.zeros(3, MORE_CARDS - CARDS)), dim=-1),
    )
    larger_guesser.set_meta(_meta(MORE_CARDS))
    _move_weights(guesser, seed=3)
    _move_weights(larger_guesser, seed=3)
    no_mask = torch.ones(3, 0, dtype=torch.bool)

    with torch.no_grad():
        guess = guesser.guess(boards, Earlier(guesses=_some_guess((3, 0, CARDS), seed=2), mask=no_mask))
        larger_guess = larger_guesser.guess(
            boards, Earlier(guesses=_some_guess((3, 0, MORE_CARDS), seed=2), mask=no_mask)
        )
        new_cards_unlikely = Guess(
            deck=torch.cat((guess.deck, torch.zeros(3, MORE_CARDS - CARDS)), dim=-1),
            hand=torch.cat((guess.hand, torch.zeros(3, MORE_CARDS - CARDS)), dim=-1),
        )
        logits = player.policy(boards, guess)
        larger_logits = larger_player.policy(boards, new_cards_unlikely)
        win_chance = win_chance_model.win_chance(boards, truth)
        larger_win_chance = larger_win_chance_model.win_chance(boards, no_new_cards)

    assert larger_guess.hand.shape == (3, MORE_CARDS)
    assert torch.allclose(larger_guess.hand[:, :CARDS], guess.hand, atol=1e-5)
    assert torch.allclose(larger_guess.deck.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(larger_logits, logits, atol=1e-4)
    assert torch.allclose(larger_win_chance, win_chance, atol=1e-5)


def test_the_guess_never_reads_the_opponents_decklist_whatever_the_weights() -> None:
    # An engine that opens the decklists reports the opponent's deck in the decklist rows of seat 1. Two
    # different decks there, with the same board and the same meta, give the same guess.
    boards = _real_boards()
    _, guesser = _models(CARDS)
    _move_weights(guesser, seed=3)
    layout = board_layout()
    with_decks = []
    for first_card in (0, 200):
        opened = {name: rows.clone() for name, rows in boards.items()}
        opened["decklists"][:, 1, :20, layout.decklist_definition] = torch.arange(first_card, first_card + 20) + 1.0
        opened["decklists"][:, 1, :20, list(layout.decklist_copies)] = 2.0
        with_decks.append(opened)

    with torch.no_grad():
        guesses = [guesser.guess(opened, NO_EARLIER_GUESSES) for opened in with_decks]

    assert torch.equal(guesses[0].deck, guesses[1].deck)
    assert torch.equal(guesses[0].hand, guesses[1].hand)


@pytest.mark.parametrize("empty_library", [False, True], ids=["library as dealt", "library empty"])
def test_every_legal_card_keeps_a_chance_and_the_chances_add_up_to_at_most_the_hand(*, empty_library: bool) -> None:
    # No deck behind the meta holds most legal cards, and the opponent shows all five copies of card 267
    # that the decks hold, yet the meta cannot know the opponent's deck holds no more. With its library
    # empty, every card the opponent has not shown is in its hand, which still holds only 6, 5 and 5 cards.
    boards = {name: rows.clone() for name, rows in _real_boards().items()}
    if empty_library:
        boards["opponent_player"][:, 0, board_layout().library_count] = 0.0
    _, guesser = _models(CARDS)
    _move_weights(guesser, seed=3)

    with torch.no_grad():
        untrained = _models(CARDS)[1].guess(boards, NO_EARLIER_GUESSES)
        trained = guesser.guess(boards, NO_EARLIER_GUESSES)

    for guess in (untrained, trained):
        assert bool((guess.deck[:, :LEGAL_CARDS] > 0).all())
        assert bool((guess.hand[:, :LEGAL_CARDS] > 0).all())
        assert bool((guess.deck[:, LEGAL_CARDS:] == 0).all())
        assert bool((guess.hand[:, LEGAL_CARDS:] == 0).all())
    assert bool((untrained.hand.sum(dim=-1) <= torch.tensor([6.0, 5.0, 5.0]) + 1e-4).all())
    assert not torch.allclose(trained.hand, untrained.hand, atol=1e-3)

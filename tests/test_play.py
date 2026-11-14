"""Check the guess model and the player model on real engine boards.

The boards in `tests/data/manabot_v10_boards.npz` are three decisions from
real games, encoded by the engine adapter at input version 10, each with
permanents, events and the acting seat's decklist on the board. No
permanent on them blocks or is attached to anything, so the test of those
links adds one. The models are built from the layout the adapter reports,
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
from jace.play import Boards, Earlier, Guess, GuessModel, PlayerModel, random_policy

__all__ = ()

ENGINE_DEFINITIONS = 480
CARDS = 480
MORE_CARDS = 520
CARD_WIDTH = 16


def _real_boards() -> dict[str, Tensor]:
    with np.load(Path(__file__).parent / "data" / "manabot_v10_boards.npz") as arrays:
        return {name: torch.from_numpy(arrays[name]) for name in arrays.files}


def _models(cards: int) -> tuple[PlayerModel, GuessModel]:
    torch.manual_seed(0)
    starting = torch.randn(MORE_CARDS, CARD_WIDTH, generator=torch.Generator().manual_seed(0))
    embedding = CardTable(starting[:cards])
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
    return player.eval(), guesser.eval()


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
        logits, win_chance = player.policy(boards, _some_guess((3, CARDS), seed=1))

    legal = boards["actions_valid"] > 0
    probabilities = logits.softmax(dim=-1)
    assert legal.sum(dim=-1).tolist() == [5, 4, 2]
    assert torch.equal(probabilities > 0, legal)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3))
    assert bool(((win_chance > 0) & (win_chance < 1)).all())


def test_player_reads_both_the_hand_guess_and_the_deck_guess() -> None:
    boards = _real_boards()
    player, _ = _models(CARDS)
    guess = _some_guess((3, CARDS), seed=1)
    other = _some_guess((3, CARDS), seed=2)

    with torch.no_grad():
        logits, _ = player.policy(boards, guess)
        with_other_hand, _ = player.policy(boards, Guess(deck=guess.deck, hand=other.hand))
        with_other_deck, _ = player.policy(boards, Guess(deck=other.deck, hand=guess.hand))

    legal = boards["actions_valid"] > 0
    assert not torch.allclose(logits[legal], with_other_hand[legal])
    assert not torch.allclose(logits[legal], with_other_deck[legal])


def _action_odds(player: PlayerModel, boards: Boards) -> Tensor:
    with torch.no_grad():
        logits, _ = player.policy(boards, _some_guess((3, CARDS), seed=1))
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
    logits, win_chance = random_policy(boards, _some_guess((3, CARDS), seed=1))

    probabilities = logits.softmax(dim=-1)
    assert torch.allclose(probabilities[:, 0], torch.tensor([1 / 5, 1 / 4, 1 / 2]))
    assert torch.equal(probabilities > 0, boards["actions_valid"] > 0)
    assert win_chance.tolist() == [0.5, 0.5, 0.5]


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


def test_padding_rows_do_not_change_the_policy_or_the_guess() -> None:
    boards = _real_boards()
    padded_boards = _with_padding_overwritten(boards)
    player, guesser = _models(CARDS)
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
        logits, win_chance = player.policy(boards, guess)
        padded_logits, padded_win_chance = player.policy(padded_boards, guess)

    assert torch.allclose(padded_guess.deck, guess.deck, atol=1e-5)
    assert torch.allclose(padded_guess.hand, guess.hand, atol=1e-5)
    assert torch.allclose(padded_logits, logits, atol=1e-4)
    assert torch.allclose(padded_win_chance, win_chance, atol=1e-5)


def test_models_score_a_larger_pool_and_old_cards_keep_their_scores() -> None:
    boards = _real_boards()
    player, guesser = _models(CARDS)
    larger_player, larger_guesser = _models(CARDS)
    larger_player.embedding = larger_guesser.embedding = _models(MORE_CARDS)[0].embedding
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
        logits, _ = player.policy(boards, guess)
        larger_logits, _ = larger_player.policy(boards, new_cards_unlikely)

    assert larger_guess.hand.shape == (3, MORE_CARDS)
    assert torch.allclose(larger_guess.hand[:, :CARDS], guess.hand, atol=1e-5)
    assert torch.allclose(larger_guess.deck.sum(dim=-1), torch.ones(3), atol=1e-5)
    assert torch.allclose(larger_logits, logits, atol=1e-4)

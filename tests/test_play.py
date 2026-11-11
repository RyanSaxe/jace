"""Check the guess model and the player model on real engine boards.

The boards in `tests/data/manabot_v6_boards.npz` are three decisions from one
game, encoded by the manabot engine at input version 6: the opening decision
and two later ones with permanents, graveyards and events. The tests call the
models only through `jace.play`, so the token layout and the network can
change without touching them.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

from jace.cards import CardEmbedding
from jace.play import Boards, Earlier, Guess, GuessModel, PlayerModel, random_policy

__all__ = ()

ENGINE_DEFINITIONS = 360
CARDS = 360
MORE_CARDS = 400
CARD_WIDTH = 16


def _real_boards() -> dict[str, Tensor]:
    with np.load(Path(__file__).parent / "data" / "manabot_v6_boards.npz") as arrays:
        return {name: torch.from_numpy(arrays[name]) for name in arrays.files}


def _models(cards: int, boards: Boards) -> tuple[PlayerModel, GuessModel]:
    torch.manual_seed(0)
    starting = torch.randn(MORE_CARDS, CARD_WIDTH, generator=torch.Generator().manual_seed(0))
    embedding = CardEmbedding(starting[:cards], hidden=CARD_WIDTH, offset_bound=None)
    engine_to_card = torch.arange(ENGINE_DEFINITIONS)
    player = PlayerModel(
        embedding, engine_to_card=engine_to_card, sample=boards, width=64, layers=2, heads=4, guess_cards=8
    )
    guesser = GuessModel(embedding, engine_to_card=engine_to_card, sample=boards, width=64, layers=2, heads=4)
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
    player, _ = _models(CARDS, boards)

    with torch.no_grad():
        logits, win_chance = player.policy(boards, _some_guess((3, CARDS), seed=1))

    legal = boards["actions_valid"] > 0
    probabilities = logits.softmax(dim=-1)
    assert legal.sum(dim=-1).tolist() == [5, 4, 2]
    assert torch.equal(probabilities > 0, legal)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3))
    assert bool(((win_chance > 0) & (win_chance < 1)).all())


def test_random_policy_spreads_probability_evenly_over_legal_actions() -> None:
    boards = _real_boards()
    policy = random_policy(torch.arange(ENGINE_DEFINITIONS))

    logits, win_chance = policy(boards, _some_guess((3, CARDS), seed=1))

    probabilities = logits.softmax(dim=-1)
    assert torch.allclose(probabilities[:, 0], torch.tensor([1 / 5, 1 / 4, 1 / 2]))
    assert torch.equal(probabilities > 0, boards["actions_valid"] > 0)
    assert win_chance.tolist() == [0.5, 0.5, 0.5]


@pytest.mark.parametrize("earlier_guesses", [0, 3])
def test_guess_is_a_deck_distribution_and_hand_probabilities(earlier_guesses: int) -> None:
    boards = _real_boards()
    _, guesser = _models(CARDS, boards)
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
    player, guesser = _models(CARDS, boards)
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
    player, guesser = _models(CARDS, boards)
    larger_player, larger_guesser = _models(CARDS, boards)
    larger_player.embedding = larger_guesser.embedding = _models(MORE_CARDS, boards)[0].embedding
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

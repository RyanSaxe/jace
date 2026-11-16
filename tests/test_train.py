"""Check the training pieces on the toy game and on hand-worked examples.

The tests use only what `jace.train` exports, with tiny real models and the
toy engine. The toy game is High Card, where the best play is always to
pick the highest card. Because that best play is known, one test can check
that a player really gets better. The tests of credit and of earlier
guesses use a few rows written by hand, where every expected value can be
worked out.
"""

from typing import override

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import Tensor, nn

from jace.engine import Finished, Outcome
from jace.engine.toy import HAND_SIZE, TOY_CARDS, ToyBatch
from jace.play import Boards, Earlier, Guess
from jace.train import PlayerRule, advantages, earlier_guesses, play_games, sequences, update_player

__all__ = ()

HISTORY = 2


def _uniform_policy(boards: Boards, _guess: Guess) -> Tensor:
    rows = len(boards["hand"])
    return torch.zeros(rows, HAND_SIZE)


def _blank_guesser(boards: Boards, _earlier: Earlier) -> Guess:
    rows = len(boards["hand"])
    return Guess(deck=torch.full((rows, TOY_CARDS), 1 / TOY_CARDS), hand=torch.full((rows, TOY_CARDS), 0.3))


def test_advantages_credit_each_decision_with_the_later_surprises_of_its_own_seat_and_game() -> None:
    # In game 0, seat 0 decides at rows 0 and 2 and wins, and seat 1 decides at row 1 and loses. A decision's
    # surprise is the change from its value to its seat's next value, or, at the seat's last decision, to the
    # result.
    games = torch.tensor([0, 0, 0])
    seats = torch.tensor([0, 1, 0])
    values = torch.tensor([0.5, 0.5, 0.7])
    results = torch.tensor([1.0, 0.0, 1.0])
    layout = sequences(games, seats)

    whole_game = advantages(values, results, layout=layout, lam=1.0)
    next_step_only = advantages(values, results, layout=layout, lam=0.0)

    assert torch.allclose(whole_game, torch.tensor([0.5, -0.5, 0.3]))
    assert torch.allclose(next_step_only, torch.tensor([0.2, -0.5, 0.3]))


def test_earlier_guesses_are_the_same_seats_previous_guesses_newest_first() -> None:
    # Rows 0, 2 and 3 are seat 0 of game 0, and row 1 is seat 1. Each guess is marked with its row number, so
    # the test can see which rows' guesses come back.
    games = torch.tensor([0, 0, 0, 0])
    seats = torch.tensor([0, 1, 0, 0])
    marks = torch.arange(4.0)[:, None]
    guesses = Guess(deck=marks, hand=marks)

    earlier = earlier_guesses(guesses, sequences(games, seats), rows=torch.tensor([3, 2, 1]), history=HISTORY)

    assert earlier.guesses.deck[0, :, 0].tolist() == [2.0, 0.0]
    assert earlier.mask.tolist() == [[True, True], [True, False], [False, False]]
    assert earlier.guesses.deck[1, 0, 0].item() == 0.0


def test_play_games_plays_exactly_the_games_it_was_given_and_returns_rows_of_no_other_game() -> None:
    # Each matchup's decks appear in no other matchup, so a finished game's decks show which matchup it played.
    matchups = [((game,), (game + 100,)) for game in range(23)]

    collected = play_games(
        lambda games: ToyBatch(games, games=4, seed=3),
        matchups,
        learner=_uniform_policy,
        opponent=_uniform_policy,
        guesser=_blank_guesser,
        cards=TOY_CARDS,
        history=HISTORY,
        generator=torch.Generator().manual_seed(3),
    )

    rollout = collected.rollout
    assert sorted(game.game for game in collected.games) == list(range(len(matchups)))
    assert all(game.decks == matchups[game.game] for game in collected.games)
    assert set(rollout.games.tolist()) == set(range(len(matchups)))
    assert (collected.abandoned, collected.lost) == (0, 0)
    for game in range(len(matchups)):
        first_seat = (rollout.games == game) & (rollout.seats == 0)
        second_seat = (rollout.games == game) & (rollout.seats == 1)
        assert int(first_seat.sum()) == int(second_seat.sum()) > 0
        assert float(rollout.results[first_seat][0] + rollout.results[second_seat][0]) == 1.0


@pytest.mark.parametrize("several_shuffled_passes_on_normalized_advantages", [False, True])
def test_a_player_learns_to_pick_its_highest_card(*, several_shuffled_passes_on_normalized_advantages: bool) -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    options = several_shuffled_passes_on_normalized_advantages
    model = nn.Sequential(nn.Linear(HAND_SIZE + 1, 32), nn.Tanh(), nn.Linear(32, HAND_SIZE))
    win_chance_model = nn.Sequential(nn.Linear(HAND_SIZE + 1 + TOY_CARDS, 32), nn.Tanh(), nn.Linear(32, 1))

    def policy(boards: Boards, _guess: Guess) -> Tensor:
        view = torch.cat([boards["hand"], boards["seen"]], dim=-1)
        return model(view)

    def win_chance(boards: Boards, truth: Guess) -> Tensor:
        view_and_truth = torch.cat([boards["hand"], boards["seen"], truth.hand], dim=-1)
        return win_chance_model(view_and_truth).squeeze(-1).sigmoid()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    win_chance_optimizer = torch.optim.Adam(win_chance_model.parameters(), lr=3e-3)
    rule = PlayerRule(clip=0.2, stay=0.1, spread=0.01, keep_quantile=0.0, keep_floor=0.0, normalize=options)
    best_pick_rates = []
    for iteration in range(40):
        collected = play_games(
            lambda games, seed=iteration: ToyBatch(games, games=64, seed=seed),
            [((0,), (1,))] * 512,
            learner=policy,
            opponent=_uniform_policy,
            guesser=_blank_guesser,
            cards=TOY_CARDS,
            history=HISTORY,
            generator=generator,
        )
        rollout = collected.rollout
        mine = rollout.learner
        best_pick_rates.append(
            float((rollout.boards["hand"][mine].argmax(dim=-1) == rollout.actions[mine]).float().mean())
        )
        update_player(
            policy,
            rollout,
            win_chance=win_chance,
            rule=rule,
            lam_advantage=0.5,
            lam_value=0.8,
            slice_rows=256,
            max_grad_norm=1.0,
            policy_optimizers=[optimizer],
            win_chance_optimizers=[win_chance_optimizer],
            passes=2 if options else 1,
            generator=generator,
        )

    assert best_pick_rates[0] < 0.45
    assert best_pick_rates[-1] > 0.65


class _AbandonsEveryThirdGame(ToyBatch):
    """The toy game, except that every third game of the list is reported as abandoned when it ends.

    It stands in for an engine that cannot finish some games, which the toy
    engine never does. The other games finish with real results, so a test
    can tell the two kinds of ending apart.
    """

    @override
    def act(self, actions: npt.NDArray[np.int64]) -> tuple[Finished, ...]:
        return tuple(
            Finished(game=ended.game, outcome=Outcome.ABANDONED, decks=ended.decks) if ended.game % 3 == 0 else ended
            for ended in super().act(actions)
        )


def test_an_abandoned_game_is_counted_and_none_of_its_decisions_are_returned() -> None:
    collected = play_games(
        lambda games: _AbandonsEveryThirdGame(games, games=4, seed=5),
        [((0,), (1,))] * 12,
        learner=_uniform_policy,
        opponent=_uniform_policy,
        guesser=_blank_guesser,
        cards=TOY_CARDS,
        history=HISTORY,
        generator=torch.Generator().manual_seed(5),
    )

    finished = {game.game for game in collected.games}
    assert collected.abandoned == 4
    assert finished == {game for game in range(12) if game % 3}
    assert set(collected.rollout.games.tolist()) == finished

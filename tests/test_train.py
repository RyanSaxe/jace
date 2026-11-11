"""Check the training pieces on the toy game and on hand-worked examples.

The tests use only what `jace.train` exports, with a tiny real model and the
toy engine. They assert outcomes a loop author relies on: credit reaches the
right decisions, games are never lost or split between calls, and a player
actually gets better at a game whose best play is known.
"""

import torch
from torch import Tensor, nn

from jace.engine.toy import HAND_SIZE, TOY_CARDS, ToyBatch
from jace.modules import run
from jace.play import Boards, Earlier, Guess
from jace.train import PlayerRule, advantages, collect, earlier_guesses, sequences, update_player

__all__ = ()

HISTORY = 2


def _uniform_policy(boards: Boards, _guess: Guess) -> tuple[Tensor, Tensor]:
    rows = len(boards["hand"])
    return torch.zeros(rows, HAND_SIZE), torch.full((rows,), 0.5)


def _blank_guesser(boards: Boards, _earlier: Earlier) -> Guess:
    rows = len(boards["hand"])
    return Guess(deck=torch.full((rows, TOY_CARDS), 1 / TOY_CARDS), hand=torch.full((rows, TOY_CARDS), 0.3))


def test_advantages_credit_each_decision_with_the_later_surprises_of_its_own_seat_and_game() -> None:
    # Game 0, seat 0 decides at rows 0 and 2 and wins. Seat 1 decides at row 1 and loses.
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
    # Rows 0, 2, 3 are game 0 seat 0. Row 1 is the other seat. Each guess is marked by its row number.
    games = torch.tensor([0, 0, 0, 0])
    seats = torch.tensor([0, 1, 0, 0])
    marks = torch.arange(4.0)[:, None]
    guesses = Guess(deck=marks, hand=marks)

    earlier = earlier_guesses(guesses, sequences(games, seats), rows=torch.tensor([3, 2, 1]), history=HISTORY)

    assert earlier.guesses.deck[0, :, 0].tolist() == [2.0, 0.0]
    assert earlier.mask.tolist() == [[True, True], [True, False], [False, False]]
    assert earlier.guesses.deck[1, 0, 0].item() == 0.0


def test_a_game_still_running_when_collect_stops_is_returned_whole_by_a_later_call() -> None:
    batch = ToyBatch(lambda: ((0,), (1,)), games=16, seed=3)
    generator = torch.Generator().manual_seed(3)
    rollouts = []
    games_carried = 0
    pending = None
    for _ in range(4):
        rollout, pending = collect(
            batch,
            pending,
            learner=_uniform_policy,
            opponent=_uniform_policy,
            guesser=_blank_guesser,
            cards=TOY_CARDS,
            history=HISTORY,
            decisions=20,
            generator=generator,
        )
        rollouts.append(rollout)
        games_carried += 0 if pending.rows is None else len(pending.rows.games.unique())

    assert games_carried > 0
    game_sets = [set(rollout.games.tolist()) for rollout in rollouts]
    assert sum(len(games) for games in game_sets) == len(set().union(*game_sets))
    for rollout in rollouts:
        for game in set(rollout.games.tolist()):
            rows = rollout.games == game
            first_seat = rows & (rollout.seats == 0)
            second_seat = rows & (rollout.seats == 1)
            assert int(first_seat.sum()) == int(second_seat.sum())
            assert float(rollout.results[first_seat][0] + rollout.results[second_seat][0]) == 1.0


def test_a_player_learns_to_pick_its_highest_card() -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = nn.Sequential(nn.Linear(HAND_SIZE + 1, 32), nn.Tanh(), nn.Linear(32, HAND_SIZE + 1))

    def policy(boards: Boards, _guess: Guess) -> tuple[Tensor, Tensor]:
        output = run(model, torch.cat([boards["hand"], boards["seen"]], dim=-1))
        return output[:, :HAND_SIZE], output[:, HAND_SIZE].sigmoid()

    batch = ToyBatch(lambda: ((0,), (1,)), games=64, seed=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    rule = PlayerRule(clip=0.2, stay=0.1, spread=0.01, keep_quantile=0.0, keep_floor=0.0, value_weight=0.5)
    best_pick_rates = []
    pending = None
    for _ in range(40):
        rollout, pending = collect(
            batch,
            pending,
            learner=policy,
            opponent=_uniform_policy,
            guesser=_blank_guesser,
            cards=TOY_CARDS,
            history=HISTORY,
            decisions=512,
            generator=generator,
        )
        mine = rollout.learner
        best_pick_rates.append(
            float((rollout.boards["hand"][mine].argmax(dim=-1) == rollout.actions[mine]).float().mean())
        )
        update_player(
            policy,
            rollout,
            rule=rule,
            lam_advantage=0.5,
            lam_value=0.8,
            slice_rows=256,
            max_grad_norm=1.0,
            optimizers=[optimizer],
        )

    assert best_pick_rates[0] < 0.45
    assert best_pick_rates[-1] > 0.65

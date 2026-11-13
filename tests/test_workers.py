"""Check that a round played in worker processes survives a worker that stops moving.

Both tests run real worker processes on the toy game. One makes a worker
stall for good in its first round and checks that the round still returns
the other workers' games, counts the stalled worker's as lost, and that the
next round is whole again. The other checks that every decision of a round
was made with the weights sent for that round.
"""

import functools
import os
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from jace.engine.toy import HAND_SIZE, TOY_CARDS, ToyBatch
from jace.play import Boards, Earlier, Guess
from jace.train.workers import Actor, play_on_workers, start_workers, stop_workers

__all__ = ()


def _blank_guesser(boards: Boards, _earlier: Earlier) -> Guess:
    rows = len(boards["hand"])
    return Guess(deck=torch.full((rows, TOY_CARDS), 1 / TOY_CARDS), hand=torch.full((rows, TOY_CARDS), 0.3))


def _toy_actor(seed: int, *, stall_marker: Path | None) -> Actor:
    # The toy game, four games at once, and a policy whose only weights are one score for each of
    # the three picks. It reports the first score as its win chance, so a decision shows which
    # weights made it. With a marker path, the first worker to make a decision creates the file
    # and never returns.
    scores = nn.Embedding(1, HAND_SIZE)

    def policy(boards: Boards, _guess: Guess) -> tuple[Tensor, Tensor]:
        if stall_marker is not None:
            try:
                os.close(os.open(stall_marker, os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                pass
            else:
                time.sleep(600)
        rows = len(boards["hand"])
        return scores.weight.expand(rows, -1), scores.weight[0, 0].expand(rows)

    return Actor(
        start=lambda matchups: ToyBatch(matchups, games=4, seed=seed),
        policy=policy,
        guesser=_blank_guesser,
        modules={"scores": scores},
        cards=TOY_CARDS,
        history=1,
    )


def test_a_worker_that_stops_moving_is_replaced_and_the_other_workers_games_are_kept(tmp_path: Path) -> None:
    weights = {"scores": {"weight": torch.zeros(1, HAND_SIZE)}}
    matchups = [((game,), (game + 100,)) for game in range(30)]
    workers = start_workers(functools.partial(_toy_actor, stall_marker=tmp_path / "stalled"), count=3, seed=0)
    try:
        first, workers = play_on_workers(workers, matchups, weights=weights, stall_seconds=3.0)
        second, workers = play_on_workers(workers, matchups, weights=weights, stall_seconds=3.0)
    finally:
        stop_workers(workers)

    # One of three workers stalled in the first round, and its ten games are the ones missing.
    assert (first.lost, second.lost) == (10, 0)
    assert len(first.games) == 20
    assert sorted(game.game for game in second.games) == list(range(30))
    for collected in (first, second):
        assert all(game.decks == matchups[game.game] for game in collected.games)
        assert set(collected.rollout.games.tolist()) == {game.game for game in collected.games}


def test_every_decision_of_a_round_is_made_with_the_weights_sent_for_that_round() -> None:
    matchups = [((0,), (1,))] * 20
    workers = start_workers(functools.partial(_toy_actor, stall_marker=None), count=2, seed=0)
    try:
        rounds = []
        for score in (0.25, 0.75):
            weights = {"scores": {"weight": torch.tensor([[score, -30.0, -30.0]])}}
            collected, workers = play_on_workers(workers, matchups, weights=weights, stall_seconds=30.0)
            rounds.append(collected.rollout)
    finally:
        stop_workers(workers)

    for score, rollout in zip((0.25, 0.75), rounds, strict=True):
        assert len(rollout.games.unique()) == len(matchups)
        assert bool((rollout.values == score).all())
        assert bool((rollout.actions == 0).all())

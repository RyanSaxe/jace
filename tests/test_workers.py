"""Check rounds of games played in worker processes, through `jace.train.workers`.

Both tests start real worker processes that play the toy game. The workers
share no memory, so the stalling test uses a marker file to choose which
worker stalls. The first worker to create the file stalls. Every other
worker, including the one started to replace it, finds the file already
there and plays on.
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

_CPU = torch.device("cpu")


def _blank_guesser(boards: Boards, _earlier: Earlier) -> Guess:
    rows = len(boards["hand"])
    return Guess(deck=torch.full((rows, TOY_CARDS), 1 / TOY_CARDS), hand=torch.full((rows, TOY_CARDS), 0.3))


def _toy_actor(seed: int, *, stall_marker: Path | None) -> Actor:
    # An actor that plays the toy game, four games at once. Its policy's only weights are one score for each
    # of the three picks, so a decision's logits show which weights made it. With a marker path, the first
    # worker to make a decision creates the file and then sleeps for ten minutes instead of returning.
    scores = nn.Embedding(1, HAND_SIZE)

    def policy(boards: Boards, _guess: Guess) -> Tensor:
        if stall_marker is not None:
            try:
                os.close(os.open(stall_marker, os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                pass
            else:
                time.sleep(600)
        return scores.weight.expand(len(boards["hand"]), -1)

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
        first, workers = play_on_workers(workers, matchups, weights=weights, stall_seconds=3.0, device=_CPU)
        second, workers = play_on_workers(workers, matchups, weights=weights, stall_seconds=3.0, device=_CPU)
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
            collected, workers = play_on_workers(workers, matchups, weights=weights, stall_seconds=30.0, device=_CPU)
            rounds.append(collected.rollout)
    finally:
        stop_workers(workers)

    for score, rollout in zip((0.25, 0.75), rounds, strict=True):
        assert len(rollout.games.unique()) == len(matchups)
        assert bool((rollout.logits[:, 0] == score).all())
        assert bool((rollout.actions == 0).all())

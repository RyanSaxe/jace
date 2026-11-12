"""Check that games collected in worker processes survive a worker that stops moving.

Both tests run real worker processes on the toy game. One makes a worker
stall for good in its first round and checks that the round still returns
the other workers' games and that the next round is whole again. The other
checks that a round is played with the weights sent for it.
"""

import functools
import os
import time
from pathlib import Path

import torch
from torch import Tensor, nn

from jace.engine.toy import HAND_SIZE, TOY_CARDS, ToyBatch
from jace.play import Boards, Earlier, Guess
from jace.train.workers import Actor, gather, start_workers, stop_workers

__all__ = ()


def _blank_guesser(boards: Boards, _earlier: Earlier) -> Guess:
    rows = len(boards["hand"])
    return Guess(deck=torch.full((rows, TOY_CARDS), 1 / TOY_CARDS), hand=torch.full((rows, TOY_CARDS), 0.3))


def _toy_actor(seed: int, *, stall_marker: Path | None) -> Actor:
    # Four toy games and a policy whose only weights are one score for each of the three picks.
    # With a marker path, the first worker to make a decision creates the file and never returns.
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
        return scores.weight.expand(rows, -1), torch.full((rows,), 0.5)

    return Actor(
        batch=ToyBatch(lambda: ((0,), (1,)), games=4, seed=seed),
        policy=policy,
        guesser=_blank_guesser,
        modules={"scores": scores},
        cards=TOY_CARDS,
        history=1,
    )


def test_a_worker_that_stops_moving_is_replaced_and_the_other_workers_games_are_kept(tmp_path: Path) -> None:
    weights = {"scores": {"weight": torch.zeros(1, HAND_SIZE)}}
    workers = start_workers(functools.partial(_toy_actor, stall_marker=tmp_path / "stalled"), count=3, seed=0)
    try:
        first = gather(workers, weights=weights, decisions=300, stall_seconds=3.0)
        workers = first.workers
        second = gather(workers, weights=weights, decisions=300, stall_seconds=3.0)
        workers = second.workers
    finally:
        stop_workers(workers)

    assert (first.lost_workers, second.lost_workers) == (1, 0)
    # Two of three workers delivered their share of the first round, and all three of the second.
    assert 200 <= len(first.rollout.games) < 300 <= len(second.rollout.games)
    for gathered in (first, second):
        ids = [game.game for game in gathered.games]
        assert len(set(ids)) == len(ids)
        assert set(gathered.rollout.games.tolist()) == set(ids)


def test_workers_play_a_round_with_the_weights_sent_for_it() -> None:
    always_last = {"scores": {"weight": torch.tensor([[-30.0, -30.0, 30.0]])}}
    workers = start_workers(functools.partial(_toy_actor, stall_marker=None), count=2, seed=0)
    try:
        gathered = gather(workers, weights=always_last, decisions=200, stall_seconds=30.0)
    finally:
        stop_workers(workers)

    assert bool((gathered.rollout.actions == HAND_SIZE - 1).all())

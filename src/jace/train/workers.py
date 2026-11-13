"""Play a round's games in worker processes, and replace a worker whose games stop moving.

A step of a game engine can take very long on a rare board, and a call
into it cannot be interrupted from Python. When games are played in the
training process, that one step freezes the run. Here each worker process
has its own copy of the models and plays its share of a round's matchups
with `jace.train.play_games`. The training process sends every worker the
round's weights and its share, takes back the decisions, and kills a
worker that completes no step for a time limit. The games of that worker's
share are lost and counted, the other workers' are not, and a new process
takes its place for the next round. Nothing carries over from one round to
the next, so every decision of a round was made with that round's weights.

A worker's life is explicit: it is `Starting` while its process builds the
models, `Ready` between rounds and `Playing` during one. Each change of
state is one function below. Workers are started with the spawn method, so
a builder must be a module-level function or a `functools.partial` of one
over plain values.
"""

import ctypes
import io
import multiprocessing
import multiprocessing.connection
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

import torch
from torch import Tensor, nn

from jace.engine import Matchup
from jace.play import Boards, Guess, Guesser, Policy
from jace.train.collect import Collected, StartBatch, join, play_games

__all__ = (
    "Actor",
    "BuildActor",
    "Playing",
    "Ready",
    "Starting",
    "Weights",
    "Worker",
    "WorkerFailedError",
    "play_on_workers",
    "start_workers",
    "stop_workers",
)

# The state dict of each module a worker plays with, by the name the worker's `Actor` gives it.
type Weights = Mapping[str, Mapping[str, Tensor]]

# How often the training process looks at the workers' step counters while it waits for games.
_WATCH_SECONDS = 0.5
# A worker's seed has this many bits, the most a PyTorch generator takes.
_SEED_BITS = 63


@dataclass(frozen=True, slots=True)
class Actor:
    """What a worker plays with, built inside its own process.

    `start` makes a batch for the matchups of one round. `policy` and
    `guesser` play both seats of every game. `modules` are the modules
    behind them, by name: at the start of each round the worker loads the
    state dict sent under each name into the module of that name. `cards`
    is the number of cards in the pool and `history` how many earlier
    guesses the guesser is shown.
    """

    start: StartBatch
    policy: Policy
    guesser: Guesser
    modules: Mapping[str, nn.Module]
    cards: int
    history: int


# (seed) -> the actor of one worker. It runs in the worker's process.
type BuildActor = Callable[[int], Actor]


@dataclass(frozen=True, slots=True)
class _Child:
    process: BaseProcess
    connection: Connection
    # steps: how many steps of its games the child has completed, which the child raises itself.
    steps: ctypes.c_int64
    build: BuildActor
    seed: int


@dataclass(frozen=True, slots=True)
class Starting:
    # The process is building its actor. since: when it was started, in `time.monotonic` seconds.
    child: _Child
    since: float


@dataclass(frozen=True, slots=True)
class Ready:
    # The actor is built and the worker waits for a round.
    child: _Child


@dataclass(frozen=True, slots=True)
class Playing:
    # The worker is playing its share of a round. share: the position in the round's list of each
    # matchup it was given. steps: its step count when last looked at. since: when that count last rose.
    child: _Child
    share: tuple[int, ...]
    steps: int
    since: float


type Worker = Starting | Ready | Playing


class WorkerFailedError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"a worker process {reason}")


def start_workers(build: BuildActor, *, count: int, seed: int) -> tuple[Worker, ...]:
    rng = random.Random(seed)
    return tuple(_spawn(build, seed=rng.getrandbits(_SEED_BITS)) for _ in range(count))


def stop_workers(workers: Sequence[Worker]) -> None:
    for worker in workers:
        _kill(worker.child)


def play_on_workers(
    workers: Sequence[Worker], matchups: Sequence[Matchup], *, weights: Weights, stall_seconds: float
) -> tuple[Collected, tuple[Worker, ...]]:
    """Play every game of `matchups` on the workers and return the decisions and the workers.

    The matchups are dealt out to the workers in turn. Each worker loads
    `weights` and plays its share to the end in self-play. A game in the
    result is known by its position in `matchups`. The workers that come
    back are the ones to pass to the next round.

    A worker that completes no step of its games for `stall_seconds` is
    killed, and a new process is started in its place for the next round.
    The games of its share are not in the result and `Collected.lost`
    counts them. A new process has the same `stall_seconds` to build its
    actor. Raises `WorkerFailedError` when a process exits by itself, does
    not finish building in time, or when every worker was killed.
    """
    packed = io.BytesIO()
    torch.save({name: dict(state) for name, state in weights.items()}, packed)
    ready = [_await_ready(worker, stall_seconds=stall_seconds) for worker in workers]
    shares = [tuple(range(first, len(matchups), len(ready))) for first in range(len(ready))]
    states: list[Worker] = [
        _assign(worker, weights=packed.getvalue(), matchups=matchups, share=share) if share else worker
        for worker, share in zip(ready, shares, strict=True)
    ]
    played: list[tuple[tuple[int, ...], Collected]] = []
    while any(isinstance(state, Playing) for state in states):
        multiprocessing.connection.wait(
            [state.child.connection for state in states if isinstance(state, Playing)], timeout=_WATCH_SECONDS
        )
        states = [
            _advance(state, played=played, stall_seconds=stall_seconds) if isinstance(state, Playing) else state
            for state in states
        ]
    if not played:
        stop_workers(states)
        reason = "returned no game: every one that was playing stopped making progress"
        raise WorkerFailedError(reason)
    return _combined(played, games=len(matchups)), tuple(states)


def _spawn(build: BuildActor, *, seed: int) -> Starting:
    context = multiprocessing.get_context("spawn")
    parent_end, child_end = context.Pipe()
    steps = context.RawValue(ctypes.c_int64, 0)
    process = context.Process(target=_work, args=(child_end, steps), kwargs={"build": build, "seed": seed}, daemon=True)
    process.start()
    child_end.close()
    child = _Child(process=process, connection=parent_end, steps=steps, build=build, seed=seed)
    return Starting(child=child, since=time.monotonic())


def _await_ready(worker: Worker, *, stall_seconds: float) -> Ready:
    # Starting -> Ready, once the process says its actor is built. A ready worker is returned as it is.
    if isinstance(worker, Ready):
        return worker
    if isinstance(worker, Playing):
        reason = "is still playing the round before"
        raise WorkerFailedError(reason)
    remaining = worker.since + stall_seconds - time.monotonic()
    if not worker.child.connection.poll(max(remaining, 0.0)):
        _kill(worker.child)
        reason = f"did not build its actor within {stall_seconds} seconds"
        raise WorkerFailedError(reason)
    _receive(worker.child.connection)
    return Ready(child=worker.child)


def _assign(worker: Ready, *, weights: bytes, matchups: Sequence[Matchup], share: tuple[int, ...]) -> Playing:
    # Ready -> Playing.
    worker.child.connection.send((weights, [matchups[position] for position in share]))
    return Playing(child=worker.child, share=share, steps=worker.child.steps.value, since=time.monotonic())


def _advance(worker: Playing, *, played: list[tuple[tuple[int, ...], Collected]], stall_seconds: float) -> Worker:
    # Playing -> Ready once the worker sends its games, which are added to `played` with its
    # share. Playing -> Playing while its step count rises. Playing -> Starting, as a new
    # process, once the count has not risen for `stall_seconds`.
    if worker.child.connection.poll():
        collected = _receive(worker.child.connection)
        if collected is None:
            reason = "said it was ready in the middle of a round"
            raise WorkerFailedError(reason)
        played.append((worker.share, collected))
        return Ready(child=worker.child)
    now = time.monotonic()
    steps = worker.child.steps.value
    if steps != worker.steps:
        return replace(worker, steps=steps, since=now)
    if now - worker.since < stall_seconds:
        return worker
    _kill(worker.child)
    return _spawn(worker.child.build, seed=random.Random(worker.child.seed).getrandbits(_SEED_BITS))


def _kill(child: _Child) -> None:
    child.process.kill()
    child.process.join()
    child.connection.close()


def _receive(connection: Connection) -> Collected | None:
    # The next message of a child: the games of its share, or None when it has built its actor.
    try:
        message = connection.recv()
    except EOFError as error:
        reason = "exited by itself; its own error is printed above"
        raise WorkerFailedError(reason) from error
    if message is None or isinstance(message, Collected):
        return message
    reason = f"sent {type(message).__name__} where games were expected"
    raise WorkerFailedError(reason)


def _combined(played: Sequence[tuple[tuple[int, ...], Collected]], *, games: int) -> Collected:
    # Joins the workers' games, giving each game its position in the round's list in place of
    # its position in its worker's share. Whatever no worker returned is counted as lost.
    rollouts = [
        replace(collected.rollout, games=torch.tensor(share)[collected.rollout.games]) for share, collected in played
    ]
    results = [replace(game, game=share[game.game]) for share, collected in played for game in collected.games]
    return Collected(
        rollout=join(rollouts),
        games=tuple(results),
        abandoned=sum(collected.abandoned for _, collected in played),
        lost=games - sum(len(share) for share, _ in played),
    )


def _work(connection: Connection, steps: ctypes.c_int64, *, build: BuildActor, seed: int) -> None:
    # Runs in the worker's process. It says None once the actor is built. For each round it gets
    # the weights and its matchups, and sends back the `Collected` of playing them.
    # One thread, because every worker would otherwise start a thread for each core.
    torch.set_num_threads(1)
    actor = build(seed)
    generator = torch.Generator().manual_seed(seed)

    def counting(boards: Boards, guess: Guess) -> tuple[Tensor, Tensor]:
        # The policy runs once for each step of the games, so this count is what the training process watches.
        steps.value += 1
        return actor.policy(boards, guess)

    connection.send(None)
    while True:
        weights, matchups = connection.recv()
        for name, state in torch.load(io.BytesIO(weights), weights_only=True).items():
            actor.modules[name].load_state_dict(state)
        connection.send(
            play_games(
                actor.start,
                matchups,
                learner=counting,
                opponent=counting,
                guesser=actor.guesser,
                cards=actor.cards,
                history=actor.history,
                generator=generator,
            )
        )

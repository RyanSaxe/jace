"""Collect self-play games in worker processes, and replace a worker whose games stop moving.

A game engine can stay inside one step for minutes, and a call into it
cannot be interrupted from Python. When games are collected in the training
process, that one step freezes the run. Here each worker process owns a
small `GameBatch` and its own copy of the models, and runs
`jace.train.collect` on it. The training process sends the weights at the
start of a round, takes back the finished games in pieces, and kills a
worker that completes no step for a time limit. The killed worker's games
in progress are lost, the other workers' are not, and a new process takes
its place for the next round. Games in progress stay inside their worker
between rounds, so nothing is discarded when a round ends.

A worker's life is explicit: it is `Starting` while its process builds the
models and the games, `Ready` between rounds and `Collecting` during one.
Each change of state is one function below. Workers are started with the
spawn method, so a builder must be a module-level function or a
`functools.partial` of one over plain values.
"""

import ctypes
import io
import math
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

from jace.engine import GameBatch
from jace.play import Boards, Guess, Guesser, Policy
from jace.train.collect import GameResult, Pending, Rollout, collect, join

__all__ = (
    "Actor",
    "BuildActor",
    "Collecting",
    "Gathered",
    "NoGamesError",
    "Ready",
    "Starting",
    "Weights",
    "Worker",
    "WorkerFailedError",
    "gather",
    "start_workers",
    "stop_workers",
)

# The state dict of each module a worker plays with, by the name the worker's `Actor` gives it.
type Weights = Mapping[str, Mapping[str, Tensor]]

# A worker sends its finished games whenever they hold this many more decisions, so a worker
# that is killed has already handed over most of what it finished.
_PIECE_DECISIONS = 256
# How often the training process looks at the workers' step counters while it waits for games.
_WATCH_SECONDS = 0.5
# A worker's seed has this many bits, the most a PyTorch generator takes.
_SEED_BITS = 63


@dataclass(frozen=True, slots=True)
class Actor:
    """What a worker plays with, built inside its own process.

    `policy` and `guesser` play both seats of every game of `batch`.
    `modules` are the modules behind them, by name: at the start of each
    round the worker loads the state dict sent under each name into the
    module of that name. `cards` is the number of cards in the pool and
    `history` how many earlier guesses the guesser is shown.
    """

    batch: GameBatch
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
class Collecting:
    # The worker is playing a round. steps: its step count when last looked at. since: when that count last rose.
    child: _Child
    steps: int
    since: float


type Worker = Starting | Ready | Collecting


@dataclass(frozen=True, slots=True)
class Gathered:
    """One round of games from every worker that kept moving.

    `rollout` holds the decisions of the finished games, with game ids
    that are unique across workers, and `games` their results under the
    same ids. `abandoned` counts the games the engine could not finish.
    `lost_workers` counts the workers killed in this round for completing
    no step within the time limit. `workers` are the workers to pass to the
    next round.
    """

    rollout: Rollout
    games: tuple[GameResult, ...]
    abandoned: int
    lost_workers: int
    workers: tuple[Worker, ...]


class WorkerFailedError(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"a worker process {reason}")


class NoGamesError(RuntimeError):
    def __init__(self, lost: int) -> None:
        super().__init__(f"no worker finished a game in this round, and {lost} were killed for making no progress")


def start_workers(build: BuildActor, *, count: int, seed: int) -> tuple[Worker, ...]:
    rng = random.Random(seed)
    return tuple(_spawn(build, seed=rng.getrandbits(_SEED_BITS)) for _ in range(count))


def stop_workers(workers: Sequence[Worker]) -> None:
    for worker in workers:
        _kill(worker.child)


def gather(workers: Sequence[Worker], *, weights: Weights, decisions: int, stall_seconds: float) -> Gathered:
    """Play one round on every worker and return the games that finished in it.

    Each worker loads `weights`, then plays until its finished games hold
    its share of `decisions` decisions, so the round holds about that many
    unless a worker is lost. A worker that completes no step of its games
    for `stall_seconds` is killed and a new process is started in its
    place, which joins the next round. Its games in progress are lost, and
    the round returns with what it had already sent and what the other
    workers produce. A new process has the same `stall_seconds` to build
    its actor.

    Raises `WorkerFailedError` when a process exits by itself or does not
    finish building in time, and `NoGamesError` when the round ends with no
    finished game.
    """
    packed = io.BytesIO()
    torch.save({name: dict(state) for name, state in weights.items()}, packed)
    ready = [_await_ready(worker, stall_seconds=stall_seconds) for worker in workers]
    quota = math.ceil(decisions / len(ready))
    states: list[Worker] = [_assign(worker, weights=packed.getvalue(), quota=quota) for worker in ready]
    pieces: list[_Piece] = []
    while any(isinstance(state, Collecting) for state in states):
        multiprocessing.connection.wait(
            [state.child.connection for state in states if isinstance(state, Collecting)], timeout=_WATCH_SECONDS
        )
        states = [
            _advance(state, pieces=pieces, stall_seconds=stall_seconds) if isinstance(state, Collecting) else state
            for state in states
        ]
    # A worker that is starting now replaced one that was killed in this round.
    lost = sum(isinstance(state, Starting) for state in states)
    if not pieces:
        stop_workers(states)
        raise NoGamesError(lost)
    rollout, games = _renumbered(pieces)
    return Gathered(
        rollout=rollout,
        games=games,
        abandoned=sum(piece.abandoned for piece in pieces),
        lost_workers=lost,
        workers=tuple(states),
    )


@dataclass(frozen=True, slots=True)
class _Piece:
    # Some finished games of one worker, with game ids that are unique within that worker only.
    rollout: Rollout
    games: tuple[GameResult, ...]
    abandoned: int


def _spawn(build: BuildActor, *, seed: int) -> Starting:
    context = multiprocessing.get_context("spawn")
    ours, theirs = context.Pipe()
    steps = context.RawValue(ctypes.c_int64, 0)
    process = context.Process(target=_work, args=(theirs, steps), kwargs={"build": build, "seed": seed}, daemon=True)
    process.start()
    theirs.close()
    child = _Child(process=process, connection=ours, steps=steps, build=build, seed=seed)
    return Starting(child=child, since=time.monotonic())


def _await_ready(worker: Worker, *, stall_seconds: float) -> Ready:
    # Starting -> Ready, once the process says its actor is built. A ready worker is returned as it is.
    if isinstance(worker, Ready):
        return worker
    if isinstance(worker, Collecting):
        reason = "is still collecting the round before"
        raise WorkerFailedError(reason)
    remaining = worker.since + stall_seconds - time.monotonic()
    if not worker.child.connection.poll(max(remaining, 0.0)):
        _kill(worker.child)
        reason = f"did not build its actor within {stall_seconds} seconds"
        raise WorkerFailedError(reason)
    _receive(worker.child.connection)
    return Ready(child=worker.child)


def _assign(worker: Ready, *, weights: bytes, quota: int) -> Collecting:
    # Ready -> Collecting.
    worker.child.connection.send((weights, quota))
    return Collecting(child=worker.child, steps=worker.child.steps.value, since=time.monotonic())


def _advance(worker: Collecting, *, pieces: list[_Piece], stall_seconds: float) -> Worker:
    # Collecting -> Ready once the worker says its round is done, after adding what it sent to
    # `pieces`. Collecting -> Collecting while its step count rises. Collecting -> Starting, as
    # a new process, once the count has not risen for `stall_seconds`.
    while worker.child.connection.poll():
        piece = _receive(worker.child.connection)
        if piece is None:
            return Ready(child=worker.child)
        pieces.append(piece)
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


def _receive(connection: Connection) -> _Piece | None:
    # The next message of a child: a piece of finished games, or None when it is ready or its round is done.
    try:
        message = connection.recv()
    except EOFError as error:
        reason = "exited by itself; its own error is printed above"
        raise WorkerFailedError(reason) from error
    if message is None or isinstance(message, _Piece):
        return message
    reason = f"sent {type(message).__name__} where finished games were expected"
    raise WorkerFailedError(reason)


def _renumbered(pieces: Sequence[_Piece]) -> tuple[Rollout, tuple[GameResult, ...]]:
    # Joins the pieces after giving every game an id no other piece uses. A game is whole within one piece.
    rollouts: list[Rollout] = []
    games: list[GameResult] = []
    used = 0
    for piece in pieces:
        old_ids, rows = piece.rollout.games.unique(return_inverse=True)
        new_id = {int(old): used + at for at, old in enumerate(old_ids.tolist())}
        rollouts.append(replace(piece.rollout, games=rows + used))
        games += [replace(game, game=new_id[game.game]) for game in piece.games]
        used += len(old_ids)
    return join(rollouts), tuple(games)


def _work(connection: Connection, steps: ctypes.c_int64, *, build: BuildActor, seed: int) -> None:
    # Runs in the worker's process. It says None once the actor is built. For each round it gets
    # (weights, quota), sends pieces of finished games until they hold `quota` decisions, then says None.
    # One thread, because every worker would otherwise start a thread for each core.
    torch.set_num_threads(1)
    actor = build(seed)
    generator = torch.Generator().manual_seed(seed)

    def counting(boards: Boards, guess: Guess) -> tuple[Tensor, Tensor]:
        # The policy runs once for each step of the games, so this count is what the training process watches.
        steps.value += 1
        return actor.policy(boards, guess)

    pending: Pending | None = None
    connection.send(None)
    while True:
        weights, quota = connection.recv()
        for name, state in torch.load(io.BytesIO(weights), weights_only=True).items():
            actor.modules[name].load_state_dict(state)
        while quota > 0:
            collected = collect(
                actor.batch,
                pending,
                learner=counting,
                opponent=counting,
                guesser=actor.guesser,
                cards=actor.cards,
                history=actor.history,
                decisions=min(quota, _PIECE_DECISIONS),
                generator=generator,
            )
            pending = collected.pending
            quota -= len(collected.rollout.games)
            connection.send(_Piece(rollout=collected.rollout, games=collected.games, abandoned=collected.abandoned))
        connection.send(None)

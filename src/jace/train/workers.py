"""Play a round's games in worker processes, and replace a worker whose games stop moving.

On a rare board, one step of the game engine can take a very long time, and
Python cannot interrupt a call into the engine. When games are played in the
training process, that one step freezes the run. Here each worker process
has its own copy of the models and plays its share of a round's matchups
with `jace.train.play_games`. The training process sends each worker the
round's weights and its share, and takes back the decisions.

When a worker completes no step of its games within a time limit, the training
process kills it. That worker's games are lost, and the result counts them, but
the other workers' games are kept. A new process takes the killed worker's
place for the next round. No game continues from one round into the next, so
every decision of a round was made with that round's weights. A worker does
keep its actor and its random generator from one round to the next.

Each worker is in one of three states. It is `Starting` while its process
builds the models, `Ready` between rounds and `Playing` during one. Each
change of state is one function below.

Workers are started with the spawn method, so multiprocessing pickles a
worker's `BuildActor` to send it to the new process. A `BuildActor` must
therefore be a module-level function, or a `functools.partial` of one over
plain values.
"""

import ctypes
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
from jace.train.collect import Collected, StartBatch, join, play_games, to_device

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

# The state dict of each module a worker plays with, keyed by the module's name in `Actor.modules`.
type Weights = Mapping[str, Mapping[str, Tensor]]

# How often the training process looks at the workers' step counters while it waits for games.
_WATCH_SECONDS = 0.5
# A worker seeds a PyTorch generator with its seed. A PyTorch generator accepts any seed from
# -2**63 to 2**64 - 1, so a seed of this many bits fits.
_SEED_BITS = 63


@dataclass(frozen=True, slots=True)
class Actor:
    """What a worker plays with, built inside its own process.

    `start` makes a batch for the matchups of one round. `policy` plays both
    seats of every game, and `guesser` makes both seats' guesses. `modules`
    are the modules that they call, by name. At the start of each round, the
    worker loads the state dict sent under each name into the module of that
    name. `cards` is the number of cards in the pool, and `history` is how
    many earlier guesses the guesser is shown.
    """

    start: StartBatch
    policy: Policy
    guesser: Guesser
    modules: Mapping[str, nn.Module]
    cards: int
    history: int


# Builds one worker's actor from the worker's seed. It runs in the worker's process.
type BuildActor = Callable[[int], Actor]


@dataclass(frozen=True, slots=True)
class _Child:
    process: BaseProcess
    connection: Connection
    # How many steps of its games the child has completed. The child raises this count itself, in
    # shared memory, so the training process can read it while the child is inside the engine.
    steps: ctypes.c_int64
    build: BuildActor
    seed: int


@dataclass(frozen=True, slots=True)
class Starting:
    # The process is building its actor. `since` is when the process was started, in
    # `time.monotonic` seconds.
    child: _Child
    since: float


@dataclass(frozen=True, slots=True)
class Ready:
    # The actor is built and the worker waits for a round.
    child: _Child


@dataclass(frozen=True, slots=True)
class Playing:
    # The worker is playing its share of a round.
    child: _Child
    # The position in the round's list of each matchup the worker was given.
    share: tuple[int, ...]
    # The worker's step count when the training process last read it.
    steps: int
    # When that count last rose, or when the round began, in `time.monotonic` seconds.
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
    workers: Sequence[Worker],
    matchups: Sequence[Matchup],
    *,
    weights: Weights,
    stall_seconds: float,
    device: torch.device,
) -> tuple[Collected, tuple[Worker, ...]]:
    """Play every game of `matchups` on the workers and return the decisions and the workers.

    The matchups are dealt out to the workers in turn. Each worker loads
    `weights` and plays its share to the end in self-play. Each game in the
    result is identified by its position in `matchups`. The decisions are
    on `device`. Each worker's decisions are moved there as soon as they
    arrive, while the other workers are still playing. Pass the returned
    workers to the next round.

    A worker that completes no step of its games for `stall_seconds` is
    killed, and a new process is started in its place for the next round.
    The killed worker's games are not in the result, and `Collected.lost`
    counts them. The new process has the same `stall_seconds` to build its
    actor.

    Raises `WorkerFailedError` when a process exits by itself or does not
    finish building its actor in time, and when every worker given a share
    was killed.
    """
    # Sending a tensor through a connection sends a handle to shared memory, and each worker maps
    # that memory. The main process therefore sends each worker a handle, not a copy of every weight.
    shared = {name: dict(state) for name, state in weights.items()}
    ready = [_await_ready(worker, stall_seconds=stall_seconds) for worker in workers]
    # Matchup i goes to worker i % workers.
    shares = [tuple(range(first, len(matchups), len(ready))) for first in range(len(ready))]
    states: list[Worker] = [
        _assign(worker, weights=shared, matchups=matchups, share=share) if share else worker
        for worker, share in zip(ready, shares, strict=True)
    ]
    played: list[Collected] = []
    while any(isinstance(state, Playing) for state in states):
        multiprocessing.connection.wait(
            [state.child.connection for state in states if isinstance(state, Playing)], timeout=_WATCH_SECONDS
        )
        states = [
            _advance(state, played=played, stall_seconds=stall_seconds, device=device)
            if isinstance(state, Playing)
            else state
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
    # Starting -> Ready, once the process reports that its actor is built. A ready worker is returned as it is.
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


def _assign(worker: Ready, *, weights: Weights, matchups: Sequence[Matchup], share: tuple[int, ...]) -> Playing:
    # Ready -> Playing.
    worker.child.connection.send((weights, [matchups[position] for position in share]))
    return Playing(child=worker.child, share=share, steps=worker.child.steps.value, since=time.monotonic())


def _advance(worker: Playing, *, played: list[Collected], stall_seconds: float, device: torch.device) -> Worker:
    # Playing -> Ready once the worker has sent its games, which are added to `played` as games of
    # the round, on `device`. Playing -> Starting, as a new process, once the step count has not
    # risen for `stall_seconds`. Otherwise the worker stays Playing.
    if worker.child.connection.poll():
        collected = _receive(worker.child.connection)
        if collected is None:
            reason = "said it was ready in the middle of a round"
            raise WorkerFailedError(reason)
        played.append(_in_round(collected, share=worker.share, device=device))
        return Ready(child=worker.child)
    now = time.monotonic()
    steps = worker.child.steps.value
    if steps != worker.steps:
        return replace(worker, steps=steps, since=now)
    if now - worker.since < stall_seconds:
        return worker
    _kill(worker.child)
    # The new process's seed is drawn from the stalled worker's seed, so a run's seeds depend only on
    # the run's own seed.
    return _spawn(worker.child.build, seed=random.Random(worker.child.seed).getrandbits(_SEED_BITS))


def _kill(child: _Child) -> None:
    child.process.kill()
    child.process.join()
    child.connection.close()


def _receive(connection: Connection) -> Collected | None:
    # The child's next message: the games of its share, or None, which the child sends once it has
    # built its actor.
    try:
        message = connection.recv()
    except EOFError as error:
        reason = "exited by itself; its own error is printed above"
        raise WorkerFailedError(reason) from error
    if message is None or isinstance(message, Collected):
        return message
    reason = f"sent {type(message).__name__} where games were expected"
    raise WorkerFailedError(reason)


def _in_round(collected: Collected, *, share: tuple[int, ...], device: torch.device) -> Collected:
    # Moves one worker's games to `device`, and renumbers each game from its position in the
    # worker's share to its position in the round's list.
    rollout = replace(collected.rollout, games=torch.tensor(share)[collected.rollout.games])
    return replace(
        collected,
        rollout=to_device(rollout, device),
        games=tuple(replace(game, game=share[game.game]) for game in collected.games),
    )


def _combined(played: Sequence[Collected], *, games: int) -> Collected:
    # Joins the workers' games. A worker plays every game of its share to an end, finished or
    # abandoned, so every game that no worker returned is counted as lost.
    returned = sum(len(collected.games) + collected.abandoned for collected in played)
    return Collected(
        rollout=join([collected.rollout for collected in played]),
        games=tuple(game for collected in played for game in collected.games),
        abandoned=sum(collected.abandoned for collected in played),
        lost=games - returned,
    )


def _work(connection: Connection, steps: ctypes.c_int64, *, build: BuildActor, seed: int) -> None:
    # Runs in the worker's process. It sends None once the actor is built. Then, for each round, it
    # receives the weights and its matchups and sends back the `Collected` from playing them.
    # One thread, because PyTorch would otherwise start a thread for each core in every worker.
    torch.set_num_threads(1)
    actor = build(seed)
    generator = torch.Generator().manual_seed(seed)

    def counting(boards: Boards, guess: Guess) -> Tensor:
        # The policy runs once for each step of the games, so this counts steps. The training process
        # reads the count to find a worker whose games have stopped advancing.
        steps.value += 1
        return actor.policy(boards, guess)

    connection.send(None)
    while True:
        weights, matchups = connection.recv()
        for name, state in weights.items():
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

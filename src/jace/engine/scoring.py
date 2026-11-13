"""Games between any two players on manabot's engine, each played to the end under a time limit.

Training steps games in batches through `GameBatch`. Scoring needs three
things a batch does not give. manabot's scripted players read manabot's own
scalar game. Two different players need a seat each, with their own memory of
the game. And the engine can enter a step that never returns, which only
killing its process ends. So this module plays one game at a time on
manabot's `Env`, in child processes, and `play_games` kills a child whose
game runs past its time limit and records that game as hung. Boards are
encoded at the policy input version of `jace.engine.manabot`, so a model
reads here what it read in training. Decklists are closed, as in training,
unless the caller opens them for a search player, which refuses to run
without the opponent's list. When the engine fails on a step, manabot's
`Env` writes that game's inputs to `.runs/game-failures` under the working
directory. `fit_ratings` is manabot's rating fit behind plain records.
"""

import collections
import logging
import multiprocessing
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Literal

import managym
import numpy.typing as npt
import torch
from manabot.env.env import Env
from manabot.env.match import Match, Reward
from manabot.eval.rating import bootstrap_population, fit_population
from manabot.infra.config import MatchConfig, RewardConfig
from manabot.sim.anchors import RandomMatchupPlayer, ScriptedGreedyPlayer
from manabot.sim.heuristic import HeuristicPlayer
from manabot.sim.players import DemoSearchPlayer
from torch import Tensor

from jace.cards import Deck
from jace.engine.batch import SEATS, Outcome
from jace.engine.manabot import observation_space

__all__ = (
    "Observation",
    "Played",
    "Player",
    "RatedGame",
    "Rating",
    "RatingNotFittedError",
    "Seat",
    "Seating",
    "WorkerDiedError",
    "fit_ratings",
    "flat_seat",
    "greedy_seat",
    "heuristic_seat",
    "play_games",
    "random_seat",
)

# The acting seat's boards as manabot encodes them, each array without a batch axis.
type Observation = dict[str, npt.NDArray[Any]]

# One seat's choice at one decision: (the game, the acting seat's boards) -> the index of the
# action, and the guess of the opponent's hand as float [cards] from a player that makes one.
type Seat = Callable[[Env, Observation], tuple[int, Tensor | None]]

# (seed) -> a seat for one game. It is called once for each game, in the process that plays the
# game, so whatever a player remembers within a game starts empty. `play_games` sends players to
# other processes, so a player must be a module-level function or a `functools.partial` of one.
type Player = Callable[[int], Seat]

# manabot's demo search plays this many games from each world it samples, and stops a game at this many steps.
_ROLLOUTS_PER_WORLD = 4
_MAX_PLAYOUT_STEPS = 2000
# manabot sets these on a step whose board has more actions, cards or permanents than the boards hold.
_OVERFLOW_FLAGS = ("action_space_truncated", "card_space_truncated", "permanent_space_truncated")


@dataclass(frozen=True, slots=True)
class Seating:
    # One game to play. Each tuple is for seat 0, then seat 1. Seat 0 takes the first turn.
    seed: int
    decks: tuple[Deck, Deck]
    players: tuple[Player, Player]
    player_seeds: tuple[int, int]


@dataclass(frozen=True, slots=True)
class Played[T]:
    outcome: Outcome
    # decisions: how many actions both seats took.
    decisions: int
    # guesses: for seat 0 and then seat 1, what the caller's `check_guesses` made of the hand
    # guesses of that seat's player in a game with a result, or None when it made none.
    guesses: tuple[T | None, T | None]


@dataclass(frozen=True, slots=True)
class RatedGame:
    player_a: str
    player_b: str
    a_seat: int
    # a_result: player A's result, 1 for a win, 0.5 for a draw, 0 for a loss.
    a_result: float
    # deal: the deal the game belongs to. Games of every pair that played the same deal share it.
    deal: int


@dataclass(frozen=True, slots=True)
class Rating:
    rating: float
    # low, high: the ends of the range that holds 95% of the ratings fitted when deals are resampled.
    low: float
    high: float


class WorkerDiedError(RuntimeError):
    def __init__(self, exit_code: int | None) -> None:
        super().__init__(f"a process playing games ended with exit code {exit_code} before reporting its game")


class RatingNotFittedError(RuntimeError):
    def __init__(self, gradient: float) -> None:
        super().__init__(f"manabot's rating fit did not converge: its largest gradient was {gradient}")


def random_seat(seed: int) -> Seat:
    player = RandomMatchupPlayer(seed)
    return _without_guess(player.act)


def greedy_seat(_seed: int) -> Seat:
    player = ScriptedGreedyPlayer()
    return _without_guess(player.act)


def heuristic_seat(_seed: int) -> Seat:
    player = HeuristicPlayer()
    return _without_guess(player.act)


def flat_seat(seed: int, *, sims: int) -> Seat:
    """Seat manabot's flat Monte Carlo search, which plays `sims` random games for each legal action.

    The search deals the opponent a hand from the cards of its deck that
    are not in view, so it raises unless `play_games` runs with open
    decklists. It reads the opponent's list and never its hand. Each
    decision costs `sims` games for each legal action, which is seconds at
    256. On manabot 0.2.1 those random games often enter an engine step
    that does not return within minutes, so many of its games end as hung.
    """
    spec = {"sims": sims, "rollouts_per_world": _ROLLOUTS_PER_WORLD, "max_steps": _MAX_PLAYOUT_STEPS}
    player = DemoSearchPlayer(spec, seed)
    return _without_guess(player.act)


def play_games[T](
    games: Sequence[Seating],
    *,
    check_guesses: Callable[[Tensor, Tensor], T],
    names: Sequence[str],
    workers: int,
    game_seconds: float,
    decklists: Literal["open", "closed"],
) -> tuple[Played[T], ...]:
    """Play every game to its end and return one `Played` for each, in order.

    `names` holds the engine's name of each card by `CardId`. The games are
    split among `workers` child processes. A game that takes longer than
    `game_seconds` is recorded as `Outcome.HUNG`: its process is killed and
    a new one plays the games that process had left. A game the engine
    fails on, cuts short, or whose board outgrows the board arrays is
    recorded as `Outcome.ABANDONED`, and so is a game during which the
    engine ends its process. No game is skipped or played twice.

    `check_guesses` is called in the child process, once for each seat that
    guessed in a game with a result. It gets that seat's hand guesses and
    the opponent's true hand at each, both float `[decisions, cards]`, and
    what it returns comes back in `Played.guesses`. It must be a
    module-level function, and its result should be small, because it
    crosses a pipe. Raises `WorkerDiedError` when a child process ends
    before it is ready to play.
    """
    shares = [games[first::workers] for first in range(min(workers, len(games)))]

    def supervise(share: Sequence[Seating]) -> list[Played[T]]:
        played: list[Played[T]] = []
        while len(played) < len(share):
            played += _attempt(
                share[len(played) :],
                check_guesses=check_guesses,
                names=names,
                game_seconds=game_seconds,
                decklists=decklists,
            )
        return played

    with ThreadPoolExecutor(max_workers=max(len(shares), 1)) as threads:
        parts = list(threads.map(supervise, shares))
    return tuple(parts[position % workers][position // workers] for position in range(len(games)))


def fit_ratings(games: Sequence[RatedGame], *, anchor: str, resamples: int, seed: int) -> dict[str, Rating]:
    """Rate every player in `games` with manabot's fit, on a scale where `anchor` is 1000.

    The fit is a Bradley-Terry model with a term for the advantage of seat
    0 and a prior that pulls each rating toward the others, so a player
    that wins every game still gets a finite rating. A difference of 400
    points is ten-to-one odds. Each range comes from refitting on
    `resamples` resamples of whole deals. Pass only games with a result.
    Raises `RatingNotFittedError` when the fit does not converge.
    """
    rows: list[dict[str, Any]] = [
        {
            "player_a": game.player_a,
            "player_b": game.player_b,
            "player_a_seat": game.a_seat,
            "score_a": game.a_result,
            "deal_block": game.deal,
        }
        for game in games
    ]
    fit = fit_population(rows, anchor=anchor)
    if not fit.converged:
        raise RatingNotFittedError(fit.gradient_norm)
    # ranges: player -> the 2.5th, 50th and 97.5th percentile of its resampled ratings.
    ranges = bootstrap_population(rows, replicates=resamples, seed=seed, anchor=anchor)["ratings"]
    return {
        player: Rating(rating=fit.ratings[player], low=float(ranges[player][0]), high=float(ranges[player][2]))
        for player in fit.players
    }


def _without_guess(act: Callable[[Env, Observation], int]) -> Seat:
    def seat(game: Env, observation: Observation) -> tuple[int, Tensor | None]:
        return act(game, observation), None

    return seat


def _attempt[T](
    games: Sequence[Seating],
    *,
    check_guesses: Callable[[Tensor, Tensor], T],
    names: Sequence[str],
    game_seconds: float,
    decklists: Literal["open", "closed"],
) -> list[Played[T]]:
    # Plays `games` in order in one new process and returns the results it reported, which stop
    # after the first game that hangs or ends the process. The time limit starts when the process says it is ready,
    # so loading the players does not count against the first game.
    context = multiprocessing.get_context("spawn")
    ours, theirs = context.Pipe()
    process = context.Process(
        target=_work,
        args=(theirs, games),
        kwargs={"check_guesses": check_guesses, "names": names, "decklists": decklists},
        daemon=True,
    )
    process.start()
    theirs.close()
    played: list[Played[T]] = []
    crashed = False
    try:
        try:
            ours.recv()
        except EOFError as error:
            # The process ended before it was ready, so its players or the engine failed to load.
            process.join()
            raise WorkerDiedError(process.exitcode) from error
        while len(played) < len(games) and ours.poll(game_seconds):
            played.append(ours.recv())
    except EOFError:
        # The engine can panic inside a game, which ends the process. That game has no result.
        crashed = True
    finally:
        process.kill()
        process.join()
        ours.close()
    if len(played) < len(games):
        outcome = Outcome.ABANDONED if crashed else Outcome.HUNG
        played.append(Played(outcome=outcome, decisions=0, guesses=(None, None)))
    return played


def _work[T](
    connection: Connection,
    games: Sequence[Seating],
    *,
    check_guesses: Callable[[Tensor, Tensor], T],
    names: Sequence[str],
    decklists: Literal["open", "closed"],
) -> None:
    # Runs in a child process: sends None when ready, then one `Played` for each game.
    # One thread, because every process of a match would otherwise start a thread for each core.
    torch.set_num_threads(1)
    # manabot logs every board it encodes at debug level.
    logging.getLogger("manabot").setLevel(logging.WARNING)
    space = observation_space()
    cards_by_name = {name: card for card, name in enumerate(names)}
    connection.send(None)
    for game in games:
        copies = [dict(collections.Counter(names[card] for card in deck)) for deck in game.decks]
        config = MatchConfig(
            world=managym.WORLD_SETUP,
            hero="seat 0",
            villain="seat 1",
            hero_deck=copies[0],
            villain_deck=copies[1],
            decklists=decklists,
        )
        env = Env(Match(config), space, Reward(RewardConfig()), seed=game.seed)
        seats = (game.players[0](game.player_seeds[0]), game.players[1](game.player_seeds[1]))
        connection.send(_play(env, seats, check_guesses=check_guesses, cards_by_name=cards_by_name))
    connection.close()


def _play[T](
    env: Env,
    seats: tuple[Seat, Seat],
    *,
    check_guesses: Callable[[Tensor, Tensor], T],
    cards_by_name: Mapping[str, int],
) -> Played[T]:
    # guessed, truths: for each seat, its hand guesses and the opponent's true hand at each, [cards] each.
    guessed: tuple[list[Tensor], list[Tensor]] = ([], [])
    truths: tuple[list[Tensor], list[Tensor]] = ([], [])
    observation: Observation
    observation, _ = env.reset()
    decisions = 0
    outcome = None
    while outcome is None:
        seat = int(env.last_raw_obs.agent.player_index)
        action, hand_guess = seats[seat](env, observation)
        if hand_guess is not None:
            guessed[seat].append(hand_guess)
            truths[seat].append(_held(env.hand_snapshot(1 - seat).cards, cards_by_name=cards_by_name))
        try:
            observation, _, terminated, truncated, info = env.step(action)
        # manabot raises `AgentError` when the engine fails the step, and `ValueError` when the
        # board after it has more cards or permanents than the board arrays hold.
        except (managym.AgentError, ValueError):
            return Played(outcome=Outcome.ABANDONED, decisions=decisions, guesses=(None, None))
        decisions += 1
        outcome = _outcome(info, terminated=terminated, truncated=truncated)
    env.close()
    if outcome is Outcome.ABANDONED:
        return Played(outcome=outcome, decisions=decisions, guesses=(None, None))
    checked = [
        check_guesses(torch.stack(guessed[seat]), torch.stack(truths[seat])) if guessed[seat] else None
        for seat in range(SEATS)
    ]
    return Played(outcome=outcome, decisions=decisions, guesses=(checked[0], checked[1]))


def _held(hand: Mapping[str, int], *, cards_by_name: Mapping[str, int]) -> Tensor:
    # hand: copies of each card name in a seat's hand. Returns float [cards], 1 where the seat holds the card.
    held = torch.zeros(len(cards_by_name))
    held[[cards_by_name[name] for name in hand if name in cards_by_name]] = 1.0
    return held


def _outcome(info: Mapping[str, object], *, terminated: bool, truncated: bool) -> Outcome | None:
    # The outcome of the game after a step, or None while it goes on.
    if terminated:
        winner = info.get("winner_index")
        if not isinstance(winner, int):
            return Outcome.DRAW
        return Outcome.SEAT_1_WON if winner else Outcome.SEAT_0_WON
    if truncated or any(info.get(flag) for flag in _OVERFLOW_FLAGS):
        return Outcome.ABANDONED
    return None

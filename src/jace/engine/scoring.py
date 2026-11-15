"""Games between any two players on manabot's engine, each played to the end under a time limit.

Training steps games in batches through `GameBatch`. This module instead
plays one game at a time on manabot's `Env`, in child processes, because
scoring needs three things that a batch does not give.

- manabot's scripted players read manabot's `Env`, which plays a single game.
- Two different players each need a seat of their own, with their own
  memory of the game.
- The engine can enter a step that never returns, and only killing its
  process ends that step. `play_games` kills a child whose game runs past
  its time limit and records that game as hung.

Boards are encoded at the policy input version of `jace.engine.manabot`, so
a model reads here what it read in training. Decklists are closed, as in
training, unless the caller opens them for a search player. The search
player refuses to run without the opponent's list.

When the engine fails on a step, manabot's `Env` writes that game's inputs
to `.runs/game-failures` under the working directory. `fit_ratings` runs
manabot's rating fit on plain records.
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

# A seat makes one player's choices in one game. At each decision it is given the game and the
# acting seat's boards. It returns the index of its action and its guess of the opponent's hand,
# float [cards], or None in place of the guess when the seat makes none.
type Seat = Callable[[Env, Observation], tuple[int, Tensor | None]]

# A player takes a seed and returns a seat for one game. `play_games` calls it once for each game,
# in the process that plays the game, so a seat's memory of the game starts empty. `play_games`
# pickles players to send them to child processes, so a player must be a module-level function or
# a `functools.partial` of one.
type Player = Callable[[int], Seat]

# manabot's demo search tries each legal action in random games that start from sampled worlds. A
# world is one deal of the opponent's hand from the cards of the opponent's deck that are not in
# view. The search plays `_ROLLOUTS_PER_WORLD` games from each world, and stops each game after
# `_MAX_PLAYOUT_STEPS` steps.
_ROLLOUTS_PER_WORLD = 4
_MAX_PLAYOUT_STEPS = 2000
# manabot sets these on a step whose board has more actions, cards or permanents than the arrays have room for.
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
    # How many actions both seats took in all. It is 0 for a game that hung or whose process the
    # engine ended.
    decisions: int
    # What the caller's `check_guesses` returned for the hand guesses of seat 0's player, then of
    # seat 1's. An entry is None when that player made no guess or the game has no result.
    guesses: tuple[T | None, T | None]


@dataclass(frozen=True, slots=True)
class RatedGame:
    player_a: str
    player_b: str
    a_seat: int
    # Player A's result: 1 for a win, 0.5 for a draw, 0 for a loss.
    a_result: float
    # The deal the game belongs to. The games of every pair of players that played the same deal have
    # the same value.
    deal: int


@dataclass(frozen=True, slots=True)
class Rating:
    rating: float
    # The 2.5th and 97.5th percentiles of the ratings fitted on resampled deals, so the range between
    # them contains 95% of those ratings.
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

    The search deals the opponent a hand from the cards of the opponent's deck
    that are not in view. Finding those cards requires the opponent's decklist,
    so it raises unless `play_games` runs with open decklists. It reads the
    opponent's list but never the opponent's hand.

    Each decision costs `sims` games for each legal action, which takes
    seconds when `sims` is 256. On manabot 0.2.1, those random games often
    enter an engine step that does not return within minutes, so many of the
    games this seat plays end as hung.
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

    `names` lists the engine's name of each card, indexed by `CardId`. The
    games are split among `workers` child processes.

    A game that takes longer than `game_seconds` is recorded as
    `Outcome.HUNG`. Its process is killed, and a new process plays the games
    that the killed one had left. A game is recorded as `Outcome.ABANDONED`
    when the engine fails on it or cuts it short, when its board outgrows
    the board arrays, or when the engine ends its process during the game.
    No game is skipped or played twice.

    `check_guesses` is called in the child process, once for each seat that
    guessed in a game with a result. It receives two float
    `[decisions, cards]` tensors: the seat's hand guesses, and the
    opponent's true hand at each guess, with 1 for each card in the hand.
    What it returns comes back in `Played.guesses`. It must be a
    module-level function, because it is pickled to reach the child
    process. Its result is sent back through a pipe, so it should be small.

    Raises `WorkerDiedError` when a child process ends before it is ready to
    play.
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
    # Game i went to share i % workers as its game i // workers, so this restores the order of `games`.
    return tuple(parts[position % workers][position // workers] for position in range(len(games)))


def fit_ratings(games: Sequence[RatedGame], *, anchor: str, resamples: int, seed: int) -> dict[str, Rating]:
    """Rate every player in `games` with manabot's fit, on a scale where `anchor` is 1000.

    `games` must contain only games with a result.

    The fit is a Bradley-Terry model with a term for the advantage of seat 0
    and a prior that pulls each rating toward the others, so a player that
    wins every game still gets a finite rating. A difference of 400 points
    is ten-to-one odds. Each range comes from refitting on `resamples`
    resamples of whole deals, drawn with `seed`.

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
    # For each player, the 2.5th, 50th and 97.5th percentiles of its ratings over the resampled deals.
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
    # Plays `games` in order in one new process. Returns a `Played` for each game up to the first one
    # that hangs or ends the process, and records that last game as hung or abandoned. The time limit
    # starts when the process sends its ready message, so loading the players does not count against the
    # first game.
    context = multiprocessing.get_context("spawn")
    parent_end, child_end = context.Pipe()
    process = context.Process(
        target=_work,
        args=(child_end, games),
        kwargs={"check_guesses": check_guesses, "names": names, "decklists": decklists},
        daemon=True,
    )
    process.start()
    child_end.close()
    played: list[Played[T]] = []
    crashed = False
    try:
        try:
            parent_end.recv()
        except EOFError as error:
            # The process ended before it was ready, so its players or the engine failed to load.
            process.join()
            raise WorkerDiedError(process.exitcode) from error
        while len(played) < len(games) and parent_end.poll(game_seconds):
            played.append(parent_end.recv())
    except EOFError:
        # The engine can panic inside a game and end the process. That game has no result.
        crashed = True
    finally:
        process.kill()
        process.join()
        parent_end.close()
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
    # Runs in a child process. Sends None once it is ready, then one `Played` for each game.
    # One thread, because otherwise PyTorch in every process playing games would start a thread for each core.
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
    # For each seat, its hand guesses and the opponent's true hand at each guess, each [cards].
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
        # board after it has more cards or permanents than the board arrays have room for.
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
    # `hand` maps each card name in a seat's hand to its copies. Returns float [cards], with 1 for each
    # card in the hand and 0 for the rest.
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

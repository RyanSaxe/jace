"""Measure a player and a guess model from games that were played to the end.

A match between two players is a set of deals, and `legs` lists every deal
as four games: each player holds each deck, in each seat. Luck of the draw,
deck strength and going first then fall on both players alike, so
`score_match` reads the difference that is left as skill. Its range
resamples whole deals, because the four games of a deal share their decks
and shuffles and are not independent. A game with no result is counted
apart and never as a win or a loss. `tally_guess` reduces a guess model's
stated chances to sums that add across games, and `guess_report` turns the
sums into a loss and a table of stated against observed chances.
`model_seat` is a jace player as one seat of one game, with its own guess
model and its own earlier guesses, so that two different players can meet.
It either samples its action from its odds, as in training, or takes its
top action, and a score usually reports both.
Nothing here plays a game: the caller plays the games `legs` lists and
hands back their outcomes.
"""

import math
import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy.typing as npt
import torch
from torch import Tensor

from jace.cards import Deck
from jace.engine import SEATS, Outcome
from jace.play import Earlier, Guess, Guesser, Policy

__all__ = (
    "CHANCE_RANGES",
    "LEGS",
    "ActionChoice",
    "Deal",
    "Game",
    "GuessReport",
    "GuessTally",
    "MatchScore",
    "ModelSeat",
    "NoResultError",
    "WinRate",
    "a_results",
    "add_tallies",
    "check_guess",
    "guess_report",
    "legs",
    "model_seat",
    "score_match",
    "tally_guess",
    "win_rate",
)

# A guess is checked in this many equal ranges of stated chance: 0 to 10%, 10 to 20%, and so on.
CHANCE_RANGES = 10

# A model's choice at one decision of one game: (the game, which it ignores, and the acting
# seat's boards, each array without a batch axis) -> the index of its action and its guess of
# the opponent's hand, float [cards]. `jace.engine.scoring` seats any function of this shape.
type ModelSeat = Callable[[object, Mapping[str, npt.NDArray[Any]]], tuple[int, Tensor]]

# Every deal is played this many times: each player with each deck, in each seat.
LEGS = 2 * SEATS

# A win rate's range leaves this share of resampled rates out at each end, so it covers 95% of them.
_RANGE_TAIL = 0.025

# Chances are kept this far from 0 and 1 before taking a logarithm.
_CHANCE_MARGIN = 1e-6

# A player's seed for one game has this many bits, the most a PyTorch generator takes.
_SEED_BITS = 63


@dataclass(frozen=True, slots=True)
class WinRate:
    rate: float
    # low, high: the ends of the range that holds 95% of the rates seen when the results are resampled.
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class Deal:
    decks: tuple[Deck, Deck]
    # seed: the engine's seed for every game of the deal.
    seed: int


@dataclass(frozen=True, slots=True)
class Game:
    # deal: the position of this game's deal in the match.
    deal: int
    seed: int
    # decks: the deck of seat 0, then of seat 1.
    decks: tuple[Deck, Deck]
    # a_seat: the seat player A takes. Player B takes the other.
    a_seat: int
    # player_seeds: the seed of seat 0's player for this game, then of seat 1's.
    player_seeds: tuple[int, int]


@dataclass(frozen=True, slots=True)
class MatchScore:
    """Player A's result against player B over the deals of one match.

    `rate` is the mean over deals of A's mean result in the deal's finished
    games, where a win is 1, a draw 0.5 and a loss 0, and its range comes
    from resampling deals. `deals` counts the deals with at least one
    finished game and `games` the finished games. `abandoned` and `hung`
    count the games that have no result, which enter no rate.
    """

    rate: WinRate
    deals: int
    games: int
    abandoned: int
    hung: int


@dataclass(frozen=True, slots=True)
class GuessTally:
    """Sums over (decision, card) pairs that a `GuessReport` is computed from.

    Tallies of separate games add, field by field, into the tally of all
    their decisions, so a process that plays games can send one small
    record per game. `pairs` is how many pairs were scored and `loss` their
    total log loss. `held` has one entry per card: the decisions at which
    it was in hand. The other three tuples hold one entry per range of
    stated chance: how many pairs fell in it, the sum of their stated
    chances and how many had the card in hand.
    """

    pairs: float
    loss: float
    held: tuple[float, ...]
    counts: tuple[float, ...]
    stated: tuple[float, ...]
    observed: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class GuessReport:
    """How good and how honest the hand guess is.

    `loss` and `base_rate_loss` are the average log loss of the guess and of
    a guess that gives each card its own frequency over these decisions,
    whatever the board shows; the model is useful only when `loss` is
    lower. `stated` and `observed` hold, for each of the
    `CHANCE_RANGES` equal ranges of stated chance, the mean chance the model stated and how
    often the card was really in hand; they match when the model is honest.
    A range with no guesses in it is NaN in both.
    """

    loss: float
    base_rate_loss: float
    stated: tuple[float, ...]
    observed: tuple[float, ...]


class NoResultError(ValueError):
    def __init__(self) -> None:
        super().__init__("no game of the match finished with a result, so it has no win rate")


def legs(deals: Sequence[Deal]) -> tuple[Game, ...]:
    """List the four games of every deal, in deal order.

    The first two games seat the deal's decks in the order given and the
    last two in the other order. Within each pair player A takes seat 0 and
    then seat 1. A pair therefore starts from the same engine seed with the
    same deck in each seat, and only the players change places. Each seat's
    player gets its own seed in every game, drawn from the deal's seed, so a
    player facing itself does not replay one game from both sides.
    """
    games: list[Game] = []
    for index, deal in enumerate(deals):
        rng = random.Random(deal.seed)
        for decks in (deal.decks, (deal.decks[1], deal.decks[0])):
            games.extend(
                Game(
                    deal=index,
                    seed=deal.seed,
                    decks=decks,
                    a_seat=a_seat,
                    player_seeds=(rng.getrandbits(_SEED_BITS), rng.getrandbits(_SEED_BITS)),
                )
                for a_seat in range(SEATS)
            )
    return tuple(games)


# How a jace player picks its action from its odds: by sampling, as in training, or the top one.
type ActionChoice = Literal["sampled", "top"]


def model_seat(
    seed: int, *, policy: Policy, guesser: Guesser, cards: int, history: int, action: ActionChoice
) -> ModelSeat:
    """Seat a jace player for one game: its policy, its guess model and its own earlier guesses.

    At each decision the seat guesses the opponent's hand and shows the
    guess to the policy. With `action` "sampled" it samples an action from
    the policy's odds with a generator made from `seed`, as the training
    games do, and with "top" it takes the highest-scored action. It keeps its last `history` guesses of this
    game, newest first, and shows them to the guess model as
    `jace.train.play_games` does. `cards` is the number of cards in the pool.
    """
    generator = torch.Generator().manual_seed(seed)
    earlier = Guess(deck=torch.zeros(1, history, cards), hand=torch.zeros(1, history, cards))
    guesses_made = 0

    def seat(_game: object, observation: Mapping[str, npt.NDArray[Any]]) -> tuple[int, Tensor]:
        nonlocal earlier, guesses_made
        boards = {name: torch.from_numpy(array)[None] for name, array in observation.items()}
        with torch.no_grad():
            guess = guesser(boards, Earlier(guesses=earlier, mask=(torch.arange(history) < guesses_made)[None]))
            logits = policy(boards, guess)
        earlier = Guess(
            deck=torch.cat((guess.deck[:, None], earlier.deck), dim=1)[:, :history],
            hand=torch.cat((guess.hand[:, None], earlier.hand), dim=1)[:, :history],
        )
        guesses_made += 1
        if action == "top":
            return int(logits.argmax(dim=-1)), guess.hand[0]
        return int(torch.multinomial(logits.softmax(dim=-1), 1, generator=generator)), guess.hand[0]

    return seat


def a_results(games: Sequence[Game], outcomes: Sequence[Outcome]) -> Tensor:
    # Returns float64 [n]: player A's result in each game, 1 for a win, 0.5 for a draw, 0 for a
    # loss, and NaN for a game that was abandoned or hung.
    for_seat_0 = {Outcome.SEAT_0_WON: 1.0, Outcome.SEAT_1_WON: 0.0, Outcome.DRAW: 0.5}
    seat_0_results = torch.tensor([for_seat_0.get(outcome, math.nan) for outcome in outcomes], dtype=torch.float64)
    a_seats = torch.tensor([game.a_seat for game in games])
    return torch.where(a_seats == 0, seat_0_results, 1.0 - seat_0_results)


def score_match(
    games: Sequence[Game], outcomes: Sequence[Outcome], *, resamples: int, generator: torch.Generator
) -> MatchScore:
    """Summarize the outcomes of `games`, in the same order, as player A's `MatchScore`.

    Every finished game counts, however long it ran. A deal's result is the
    mean of its finished games, so a deal that lost a game to the engine
    still counts with the games it has, and a deal with no finished game
    drops out. Raises `NoResultError` when no game finished.
    """
    results = a_results(games, outcomes)
    finished = ~results.isnan()
    if not bool(finished.any()):
        raise NoResultError
    deals = torch.tensor([game.deal for game in games])[finished]
    empty = torch.zeros(int(deals.max()) + 1, dtype=torch.float64)
    played = empty.scatter_add(0, deals, torch.ones_like(results[finished]))
    totals = empty.scatter_add(0, deals, results[finished])
    deal_means = (totals / played)[played > 0]
    return MatchScore(
        rate=win_rate(deal_means.float(), resamples=resamples, generator=generator),
        deals=len(deal_means),
        games=int(finished.sum()),
        abandoned=sum(outcome is Outcome.ABANDONED for outcome in outcomes),
        hung=sum(outcome is Outcome.HUNG for outcome in outcomes),
    )


def win_rate(results: Tensor, *, resamples: int, generator: torch.Generator) -> WinRate:
    """Summarize results (1 win, 0.5 draw, 0 loss) as a rate with a range.

    The range comes from resampling the results with replacement `resamples`
    times, so it reflects how many there are. It treats them as
    independent. Single games that share a deck or a deal are not, and
    their range overstates certainty: pass one mean result per deal, as
    `score_match` does.
    """
    draws = torch.randint(0, len(results), (resamples, len(results)), generator=generator)
    rates = results[draws].mean(dim=1)
    low, high = rates.quantile(torch.tensor([_RANGE_TAIL, 1.0 - _RANGE_TAIL])).tolist()
    return WinRate(rate=float(results.mean()), low=low, high=high)


def tally_guess(hand_guess: Tensor, hand_truth: Tensor) -> GuessTally:
    # hand_guess, hand_truth: [n, cards], the stated chance that the opponent holds each card and
    # 1 where it truly did. Every (decision, card) pair is scored.
    stated = hand_guess.flatten().double()
    truth = hand_truth.flatten().double()
    safe = stated.clamp(_CHANCE_MARGIN, 1.0 - _CHANCE_MARGIN)
    # A stated chance of exactly 1 falls in the last range.
    chance_range = (stated * CHANCE_RANGES).long().clamp_max(CHANCE_RANGES - 1)
    empty = torch.zeros(CHANCE_RANGES, dtype=torch.float64)
    return GuessTally(
        pairs=float(stated.numel()),
        loss=float(-(truth * safe.log() + (1.0 - truth) * (1.0 - safe).log()).sum()),
        held=tuple(hand_truth.double().sum(dim=0).tolist()),
        counts=tuple(empty.scatter_add(0, chance_range, torch.ones_like(stated)).tolist()),
        stated=tuple(empty.scatter_add(0, chance_range, stated).tolist()),
        observed=tuple(empty.scatter_add(0, chance_range, truth).tolist()),
    )


def add_tallies(tallies: Iterable[GuessTally]) -> GuessTally:
    # The tally of every pair in `tallies`, which must all be over the same cards. No tallies
    # give the tally of no pairs.
    # Each tally becomes one row of pairs, loss, the three per-range tuples and then `held`, so
    # one sum adds every field.
    parts = [
        torch.tensor((tally.pairs, tally.loss, *tally.counts, *tally.stated, *tally.observed, *tally.held))
        for tally in tallies
    ]
    if not parts:
        nothing = (0.0,) * CHANCE_RANGES
        return GuessTally(pairs=0.0, loss=0.0, held=(), counts=nothing, stated=nothing, observed=nothing)
    total = torch.stack(parts).double().sum(dim=0)
    pairs, loss = total[:2].tolist()
    counts, stated, observed = (tuple(part.tolist()) for part in total[2 : 2 + 3 * CHANCE_RANGES].chunk(3))
    held = tuple(total[2 + 3 * CHANCE_RANGES :].tolist())
    return GuessTally(pairs=pairs, loss=loss, held=held, counts=counts, stated=stated, observed=observed)


def guess_report(tally: GuessTally) -> GuessReport:
    # The report of a tally with no pairs is NaN throughout.
    counts = torch.tensor(tally.counts, dtype=torch.float64)
    held = torch.tensor(tally.held, dtype=torch.float64)
    # base_rate: [cards], how often each card was in hand over the decisions scored.
    # `pairs` is decisions times cards, so `held * cards / pairs` is `held` over decisions.
    base_rate = held * len(held) / tally.pairs if tally.pairs else held
    safe = base_rate.clamp(_CHANCE_MARGIN, 1.0 - _CHANCE_MARGIN)
    return GuessReport(
        loss=tally.loss / tally.pairs if tally.pairs else math.nan,
        base_rate_loss=float(-(base_rate * safe.log() + (1.0 - base_rate) * (1.0 - safe).log()).mean()),
        stated=tuple((torch.tensor(tally.stated, dtype=torch.float64) / counts).tolist()),
        observed=tuple((torch.tensor(tally.observed, dtype=torch.float64) / counts).tolist()),
    )


def check_guess(guess: Guess, *, hand_truth: Tensor) -> GuessReport:
    # guess.hand, hand_truth: [n, cards]. Scores every (decision, card) pair.
    tally = tally_guess(guess.hand, hand_truth)
    return guess_report(tally)

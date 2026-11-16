"""Measure a player and a guess model from games that were played to the end.

A match between two players is a set of deals. A deal is a pair of decks
and an engine seed. `legs` lists every deal as four games, so that each
player plays each deck from each seat. Both players therefore get the same
luck of the draw, play the same decks and go first equally often.
`score_match` treats the difference that is left as skill. It computes its
range by resampling whole deals, because the four games of a deal share
their decks and shuffles and are not independent. A game with no result is
counted separately, never as a win or a loss.

`tally_guess` reduces a guess model's stated chances to sums that add
across games. `guess_report` turns those sums into a loss and a table of
stated against observed chances.

`model_seat` seats a jace player in one game, with its own guess model and
its own earlier guesses, so that two different jace players can play each
other. The seat either samples its action from the policy's odds, as
training does, or takes the top action. During training, `jace.loops`
scores the player both ways.

Nothing here plays a game. The caller plays the games that `legs` lists and
passes back their outcomes.
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

# A function that makes a jace player's choices in one seat of one game. At each decision it is
# given the game, which it ignores, and the acting seat's boards, each array without a batch axis.
# It returns the index of its action and its guess of the opponent's hand, float [cards].
# `jace.engine.scoring` accepts any function of this shape as a `Seat`.
type ModelSeat = Callable[[object, Mapping[str, npt.NDArray[Any]]], tuple[int, Tensor]]

# Every deal is played this many times: each player with each deck, in each seat.
LEGS = 2 * SEATS

# `win_rate` drops this share of the resampled rates at each end of its range, so the range
# contains the middle 95% of them.
_RANGE_TAIL = 0.025

# Chances are kept this far from 0 and 1 before taking a logarithm, so that a log loss stays finite.
_CHANCE_MARGIN = 1e-6

# A player's seed for one game has this many bits, which a PyTorch generator accepts.
_SEED_BITS = 63


@dataclass(frozen=True, slots=True)
class WinRate:
    rate: float
    # The ends of the range that contains the middle 95% of the rates computed from resampled results.
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class Deal:
    decks: tuple[Deck, Deck]
    # The engine's seed for every game of the deal.
    seed: int


@dataclass(frozen=True, slots=True)
class Game:
    # The position of this game's deal in the match.
    deal: int
    seed: int
    # The deck of seat 0, then the deck of seat 1.
    decks: tuple[Deck, Deck]
    # The seat that player A takes. Player B takes the other.
    a_seat: int
    # The seed of seat 0's player for this game, then the seed of seat 1's player.
    player_seeds: tuple[int, int]


@dataclass(frozen=True, slots=True)
class MatchScore:
    """Player A's result against player B over the deals of one match.

    A's result in a game is 1 for a win, 0.5 for a draw and 0 for a loss.
    `rate` is A's mean result in each deal's finished games, averaged over
    the deals, so every deal counts the same however many of its games
    finished. Its range comes from resampling deals.

    `deals` counts the deals with at least one finished game, and `games`
    counts the finished games. `abandoned` and `hung` count the games that
    have no result, and no rate includes them.
    """

    rate: WinRate
    deals: int
    games: int
    abandoned: int
    hung: int


@dataclass(frozen=True, slots=True)
class GuessTally:
    """Sums over (decision, card) pairs, from which a `GuessReport` is computed.

    A pair is one card at one decision: the chance the guess stated that the
    opponent's hand contained the card, and whether it did. `add_tallies`
    adds the tallies of separate games, field by field, into the tally of
    all their decisions. A process that plays games can therefore send one
    small record per game.

    `pairs` is how many pairs were scored, and `loss` is their total log
    loss. `held` has one entry per card: the number of decisions at which
    the card was in the opponent's hand. `counts`, `stated` and `observed`
    have one entry per range of stated chance: how many pairs fell in the
    range, the sum of their stated chances, and how many of them had the
    card in hand.
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

    `loss` is the guess's average log loss over the scored pairs.
    `base_rate_loss` is the average log loss of a guess that ignores the
    board and gives each card the frequency with which it was in the
    opponent's hand over these decisions. The guess model is useful only
    when `loss` is lower.

    `stated` and `observed` have one entry for each of the `CHANCE_RANGES`
    equal ranges of stated chance: the mean chance the model stated in that
    range, and how often the card was really in hand. They match when the
    model is calibrated, that is, when the cards it gives a 30% chance are in
    hand about 30% of the time. A range with no guesses in it is NaN in
    both.
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

    The first two games seat the deal's decks in the order given, and the
    last two in the other order. Within each pair, player A takes seat 0 and
    then seat 1. A pair therefore starts from the same engine seed with the
    same deck in each seat, and only the players change places.

    Each seat's player gets its own seed in every game, drawn from the
    deal's seed. A player facing itself therefore does not replay one game
    from both sides.
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

    At each decision, the guess model guesses the opponent's unseen cards
    and hand, and the policy scores the actions from the board and that
    guess. The guess model also reads the seat's last `history` guesses of
    this game, newest first, as in `jace.train.play_games`. `cards` is the
    number of cards in the pool.

    With `action` "sampled", the seat samples an action from the policy's
    odds with a generator seeded from `seed`, as the training games do. With
    "top", it takes the highest-scored action. Either way, the seat returns
    the hand part of its guess.
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
    """Summarize player A's results in `games` as a `MatchScore`.

    `outcomes[i]` is the outcome of `games[i]`. Every finished game counts,
    however long it ran. A deal's result is the mean over its finished
    games. So a deal with an abandoned or hung game still counts, with the
    games that finished, and a deal with no finished game drops out.

    Raises `NoResultError` when no game finished.
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
    times, so it is wider when there are fewer results. Resampling treats
    the results as independent. Single games that share a deck or a deal
    are not independent, and their range overstates certainty. Pass one
    mean result per deal instead, as `score_match` does.
    """
    draws = torch.randint(0, len(results), (resamples, len(results)), generator=generator)
    rates = results[draws].mean(dim=1)
    low, high = rates.quantile(torch.tensor([_RANGE_TAIL, 1.0 - _RANGE_TAIL])).tolist()
    return WinRate(rate=float(results.mean()), low=low, high=high)


def tally_guess(hand_guess: Tensor, hand_truth: Tensor) -> GuessTally:
    # `hand_guess` is float [n, cards], the stated chance that the opponent's hand contains each card
    # at each of n decisions. `hand_truth` has the same shape, with 1 where the hand did contain the
    # card. Every (decision, card) pair is scored.
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
    # The tally of every pair in `tallies`, which must all be over the same cards. An empty `tallies`
    # gives the tally of no pairs, over no cards.
    # Each tally becomes one row: `pairs`, `loss`, the three per-range tuples and then `held`. One
    # sum then adds every field at once.
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
    # A tally with no pairs gives NaN for `loss`, `stated` and `observed`. Its `base_rate_loss` is
    # close to 0, or NaN when the tally is over no cards, such as the tally `add_tallies` returns for
    # no tallies.
    counts = torch.tensor(tally.counts, dtype=torch.float64)
    held = torch.tensor(tally.held, dtype=torch.float64)
    # How often each card was in hand over the scored decisions, [cards]. `pairs` is decisions times
    # cards, so `held * cards / pairs` divides `held` by the number of decisions.
    base_rate = held * len(held) / tally.pairs if tally.pairs else held
    safe = base_rate.clamp(_CHANCE_MARGIN, 1.0 - _CHANCE_MARGIN)
    return GuessReport(
        loss=tally.loss / tally.pairs if tally.pairs else math.nan,
        base_rate_loss=float(-(base_rate * safe.log() + (1.0 - base_rate) * (1.0 - safe).log()).mean()),
        stated=tuple((torch.tensor(tally.stated, dtype=torch.float64) / counts).tolist()),
        observed=tuple((torch.tensor(tally.observed, dtype=torch.float64) / counts).tolist()),
    )


def check_guess(guess: Guess, *, hand_truth: Tensor) -> GuessReport:
    # `guess.hand` and `hand_truth` are [n, cards], as in `tally_guess`. Every (decision, card) pair
    # is scored.
    tally = tally_guess(guess.hand, hand_truth)
    return guess_report(tally)

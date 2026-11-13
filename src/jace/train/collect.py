"""Play a list of games to the end and record every decision.

A round of training is a list of matchups. `play_games` plays every game of
the list from its first decision to its last with the policies it is given,
and returns the decisions as a `Rollout`. No game is cut off for being long
and none carries over to a later call, so every row of a rollout was played
by one version of the weights, and long games count as much as short ones.
A game the engine abandons is counted and its decisions are left out.

A training loop depends on `PlayGames`, the one-argument form: matchups in,
`Collected` out. `play_games` with its other arguments bound is one, on a
single batch in this process. `jace.train.workers` builds another on worker
processes. Each model runs once per step over every game in play; the only
Python loop over games visits the few that ended on a step.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace

import numpy as np
import torch
from torch import Tensor

from jace.cards import Deck
from jace.engine import SEATS, Finished, GameBatch, Matchup, Outcome
from jace.play import Boards, Earlier, Guess, Guesser, Policy

__all__ = (
    "Collected",
    "GameResult",
    "NoGamesError",
    "PlayGames",
    "Rollout",
    "StartBatch",
    "join",
    "play_games",
    "select",
)


@dataclass(frozen=True, slots=True)
class Rollout:
    """Decisions in the order they were made, one row each.

    Every tensor has a leading axis of `n` rows. `boards` and `guesses` are
    what the actor saw. `logits` are the actor's action scores, with illegal
    actions at `-inf`, and `log_probs` and `values` are its log-probability
    of the action it took and its predicted win chance. `games` and `seats`
    identify the row's game and seat, and `learner` is False where the
    opponent policy acted. `results` is the game's final result from the
    row's seat: 1 for a win, 0.5 for a draw, 0 for a loss. `deck_truth` and
    `hand_truth` are the labels for the guess: the opponent's true unseen
    cards as shares, and 1 where the opponent truly held the card.
    """

    boards: Boards
    guesses: Guess
    logits: Tensor
    actions: Tensor
    log_probs: Tensor
    values: Tensor
    games: Tensor
    seats: Tensor
    learner: Tensor
    results: Tensor
    deck_truth: Tensor
    hand_truth: Tensor


@dataclass(frozen=True, slots=True)
class GameResult:
    # game: the game's position in the list of matchups that was played.
    game: int
    decks: tuple[Deck, Deck]
    learner_seat: int
    # The learner's result: 1 for a win, 0.5 for a draw, 0 for a loss.
    learner_result: float


@dataclass(frozen=True, slots=True)
class Collected:
    # rollout: the decisions of the games that finished, each row's game being its position in
    # the list of matchups. games: one entry per finished game.
    rollout: Rollout
    games: tuple[GameResult, ...]
    # abandoned: how many games the engine could not finish. Their decisions are not in the rollout.
    abandoned: int
    # lost: how many games of the list were not played to the end because the process playing
    # them was stopped. It is zero for games played in this process.
    lost: int


# Starts a batch that plays exactly the matchups it is given.
type StartBatch = Callable[[Sequence[Matchup]], GameBatch]

# Plays every game of a list of matchups to its end and returns their decisions.
type PlayGames = Callable[[Sequence[Matchup]], Collected]


class NoGamesError(ValueError):
    def __init__(self) -> None:
        super().__init__("a round needs at least one matchup to play")


def select(rollout: Rollout, rows: Tensor) -> Rollout:
    # rows: int64 [m] row indices, or bool [n]. Returns those rows as a rollout. A field on a GPU
    # is indexed with indices on that GPU: indexing it with CPU indices copies through the CPU.
    on_device: dict[torch.device, Tensor] = {}

    def rows_of(values: Tensor) -> Tensor:
        if values.device not in on_device:
            on_device[values.device] = rows.to(values.device)
        return values[on_device[values.device]]

    plain = {
        field.name: rows_of(getattr(rollout, field.name))
        for field in fields(Rollout)
        if field.name not in {"boards", "guesses"}
    }
    return Rollout(
        boards={name: rows_of(array) for name, array in rollout.boards.items()},
        guesses=Guess(deck=rows_of(rollout.guesses.deck), hand=rows_of(rollout.guesses.hand)),
        **plain,
    )


def join(parts: Sequence[Rollout]) -> Rollout:
    # The rows of every part, in the order given. `parts` must not be empty.
    plain = {
        field.name: torch.cat([getattr(part, field.name) for part in parts])
        for field in fields(Rollout)
        if field.name not in {"boards", "guesses"}
    }
    return Rollout(
        boards={name: torch.cat([part.boards[name] for part in parts]) for name in parts[0].boards},
        guesses=Guess(
            deck=torch.cat([part.guesses.deck for part in parts]),
            hand=torch.cat([part.guesses.hand for part in parts]),
        ),
        **plain,
    )


def _decide(
    boards: Boards, guess: Guess, *, is_learner: Tensor, learner: Policy, opponent: Policy
) -> tuple[Tensor, Tensor]:
    # Runs each policy once, on the rows it acts in. Returns (logits [n, k], values [n]) for every row.
    logits = torch.empty(0)
    values = torch.zeros(len(is_learner))
    for rows, policy in ((is_learner, learner), (~is_learner, opponent)):
        if not bool(rows.any()):
            continue
        chosen_boards = {name: array[rows] for name, array in boards.items()}
        policy_logits, policy_values = policy(chosen_boards, Guess(deck=guess.deck[rows], hand=guess.hand[rows]))
        if not logits.numel():
            logits = torch.full((len(is_learner), policy_logits.shape[1]), float("-inf"))
        logits[rows] = policy_logits
        values[rows] = policy_values
    return logits, values


def _truth(batch: GameBatch, seats: Tensor) -> tuple[Tensor, Tensor]:
    # Returns (deck_truth, hand_truth), each [n, cards], for the opponent of the seat acting in each game in play.
    hidden = batch.hidden_cards()
    in_play = torch.arange(len(seats))
    # `1 - seats` is the opponent of the seat acting in each game.
    hands = torch.from_numpy(hidden.hands)[in_play, 1 - seats].float()
    unseen = hands + torch.from_numpy(hidden.libraries)[in_play, 1 - seats].float()
    return unseen / unseen.sum(dim=-1, keepdim=True).clamp_min(1.0), (hands > 0).float()


def _finished_rows(parts: Sequence[Rollout], finished: Sequence[GameResult], *, abandoned: set[int]) -> Rollout:
    # Joins the steps of a round, drops the rows of abandoned games and fills in every row's result.
    joined = join(parts)
    kept = select(joined, ~torch.isin(joined.games, torch.tensor(sorted(abandoned), dtype=torch.int64)))
    seat_0_result = torch.zeros(int(joined.games.max()) + 1)
    for game in finished:
        seat_0_result[game.game] = game.learner_result if game.learner_seat == 0 else 1.0 - game.learner_result
    row_seat_0_result = seat_0_result[kept.games]
    return replace(kept, results=torch.where(kept.seats == 0, row_seat_0_result, 1.0 - row_seat_0_result))


def _guess(
    guesser: Guesser, boards: Boards, *, games: Tensor, seats: Tensor, history: Guess, history_count: Tensor
) -> Guess:
    # Makes the acting seat's guess in every game in play and puts it at the front of that seat's
    # history, in place. games, seats: int64 [n]. history: [games of the round, 2, h, cards]
    # each; history_count: int64 [games of the round, 2].
    history_length = history.deck.shape[2]
    earlier = Guess(deck=history.deck[games, seats], hand=history.hand[games, seats])
    mask = torch.arange(history_length) < history_count[games, seats][:, None]
    guess = guesser(boards, Earlier(guesses=earlier, mask=mask))
    # The new guess goes in front and the oldest falls off the end, so the history stays newest first.
    history.deck[games, seats] = torch.cat([guess.deck[:, None], earlier.deck], dim=1)[:, :history_length]
    history.hand[games, seats] = torch.cat([guess.hand[:, None], earlier.hand], dim=1)[:, :history_length]
    history_count[games, seats] = (history_count[games, seats] + 1).clamp_max(history_length)
    return guess


def _game_result(ended: Finished, *, learner_seat: int) -> GameResult:
    seat_0_result = {Outcome.SEAT_0_WON: 1.0, Outcome.SEAT_1_WON: 0.0, Outcome.DRAW: 0.5}[ended.outcome]
    return GameResult(
        game=ended.game,
        decks=ended.decks,
        learner_seat=learner_seat,
        learner_result=seat_0_result if learner_seat == 0 else 1.0 - seat_0_result,
    )


@torch.no_grad()
def play_games(
    start: StartBatch,
    matchups: Sequence[Matchup],
    *,
    learner: Policy,
    opponent: Policy,
    guesser: Guesser,
    cards: int,
    history: int,
    generator: torch.Generator,
) -> Collected:
    """Play every game of `matchups` to its end on one batch and return their decisions.

    `start` makes the batch for the list. `cards` is the number of cards in
    the pool and `history` is how many earlier guesses the guesser is
    shown. The learner takes seat 0 in every second game of the list and
    seat 1 in the others, starting from a seat drawn at random. When
    `opponent` is the same policy as `learner`, which is self-play, the
    decisions of both seats are marked as the learner's in the rollout, and
    `GameResult.learner_result` is still the result of the learner's seat.
    Raises `NoGamesError` when `matchups` is empty.
    """
    if not matchups:
        raise NoGamesError
    batch = start(matchups)
    empty = torch.zeros(len(matchups), SEATS, history, cards)
    guess_history = Guess(deck=empty, hand=empty.clone())
    history_count = torch.zeros(len(matchups), SEATS, dtype=torch.long)
    first_seat = int(torch.randint(0, SEATS, (1,), generator=generator))
    learner_seat = (torch.arange(len(matchups)) + first_seat) % SEATS
    # `==`, not `is`: a bound method is a new object each time it is read from its model.
    self_play = opponent == learner
    parts: list[Rollout] = []
    finished: list[GameResult] = []
    abandoned: set[int] = set()
    pending = batch.decisions()
    while len(pending.games):
        games = torch.from_numpy(pending.games.copy())
        seats = torch.from_numpy(pending.seats.copy())
        boards = {name: torch.from_numpy(np.array(array)) for name, array in pending.boards.items()}
        guess = _guess(guesser, boards, games=games, seats=seats, history=guess_history, history_count=history_count)
        is_learner = torch.ones_like(seats, dtype=torch.bool) if self_play else seats == learner_seat[games]
        logits, values = _decide(boards, guess, is_learner=is_learner, learner=learner, opponent=opponent)
        log_probs = logits.log_softmax(dim=-1)
        actions = torch.multinomial(log_probs.exp(), 1, generator=generator).squeeze(1)
        deck_truth, hand_truth = _truth(batch, seats)
        parts.append(
            Rollout(
                boards=boards,
                guesses=guess,
                logits=logits,
                actions=actions,
                log_probs=log_probs.gather(1, actions[:, None]).squeeze(1),
                values=values,
                games=games,
                seats=seats,
                learner=is_learner,
                results=torch.zeros(len(games)),
                deck_truth=deck_truth,
                hand_truth=hand_truth,
            )
        )
        for ended in batch.act(actions.numpy()):
            if ended.outcome is Outcome.ABANDONED:
                abandoned.add(ended.game)
            else:
                finished.append(_game_result(ended, learner_seat=int(learner_seat[ended.game])))
        pending = batch.decisions()
    rollout = _finished_rows(parts, finished, abandoned=abandoned)
    return Collected(rollout=rollout, games=tuple(finished), abandoned=len(abandoned), lost=0)

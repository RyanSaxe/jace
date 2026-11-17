"""Play a list of games to the end and record every decision.

A round of training is a list of matchups. `play_games` plays every game of the
list from its first decision to its last with the policies it is given, and
returns the decisions as a `Rollout`. No game is cut off for being long, and
none continues into a later call. A rollout therefore never mixes in decisions
made with an earlier version of the weights, and long games count as much as
short ones. When the engine abandons a game, the game is counted and its
decisions are left out.

A training loop plays its games through a `PlayGames`, a function that takes
a round's matchups and returns a `Collected`. Binding every argument of
`play_games` except the matchups gives one that plays on a single batch in
this process. `jace.train.workers` builds one that plays on worker processes.

Within a step, the guesser runs once over every game in play, and each
policy runs once over the games where it acts. The only Python loop over
games within a step visits the few games that ended on it.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace

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
    "to_device",
)


@dataclass(frozen=True, slots=True)
class Rollout:
    """Decisions in the order they were made, one row each.

    Every tensor has a leading axis of `n` rows. `boards` and `guesses` are
    the inputs the acting policy read. `logits` are that policy's action
    scores, with illegal actions at `-inf`. `actions` is the int64 index of
    the action it took, and `log_probs` is the log-probability it gave that
    action.

    `games` is each row's game, as its position in the list of matchups, and
    `seats` is the seat that acted. `learner` is False where the opponent
    policy acted. `results` is the game's final result from the acting seat:
    1 for a win, 0.5 for a draw and 0 for a loss.

    `deck_truth` and `hand_truth` are the opponent's true unseen cards, in
    the form of a `Guess`. `deck_truth` is each card's share of the
    opponent's hand and library, and `hand_truth` is 1 where the opponent's
    hand contains at least one copy of the card. They are the labels the
    guess model learns from, and the win chance reads them in place of a
    guess.
    """

    boards: Boards
    guesses: Guess
    logits: Tensor
    actions: Tensor
    log_probs: Tensor
    games: Tensor
    seats: Tensor
    learner: Tensor
    results: Tensor
    deck_truth: Tensor
    hand_truth: Tensor


@dataclass(frozen=True, slots=True)
class GameResult:
    # The game's position in the list of matchups that was played.
    game: int
    decks: tuple[Deck, Deck]
    learner_seat: int
    # The learner's result: 1 for a win, 0.5 for a draw, 0 for a loss.
    learner_result: float
    # True when the learner's policy played both seats.
    self_play: bool


@dataclass(frozen=True, slots=True)
class Collected:
    # The decisions of the games that finished. A row's `games` entry is its game's position in the
    # list of matchups.
    rollout: Rollout
    # One result for each finished game.
    games: tuple[GameResult, ...]
    # How many games the engine could not finish. Their decisions are not in the rollout.
    abandoned: int
    # How many games of the list were not played to the end because the process playing them was
    # stopped. It is zero for games played in this process.
    lost: int


# Starts a batch that plays exactly the matchups it is given.
type StartBatch = Callable[[Sequence[Matchup]], GameBatch]

# Plays every game of a list of matchups to its end and returns their decisions.
type PlayGames = Callable[[Sequence[Matchup]], Collected]


class NoGamesError(ValueError):
    def __init__(self) -> None:
        super().__init__("a round needs at least one matchup to play")


def select(rollout: Rollout, rows: Tensor) -> Rollout:
    # `rows` is either int64 row indices [m] or a bool mask [n]. Indexing a tensor on a GPU with
    # indices on the CPU makes PyTorch copy the indices to the GPU on every call. So `rows` is
    # copied to each device once, and every field on that device is indexed with that copy.
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


def to_device(rollout: Rollout, device: torch.device) -> Rollout:
    # A loop moves a rollout to the update's device once, before the update. Every slice of every
    # pass is then cut on that device, and no model input is copied from the CPU again.
    plain = {
        field.name: getattr(rollout, field.name).to(device)
        for field in fields(Rollout)
        if field.name not in {"boards", "guesses"}
    }
    return Rollout(
        boards={name: array.to(device) for name, array in rollout.boards.items()},
        guesses=Guess(deck=rollout.guesses.deck.to(device), hand=rollout.guesses.hand.to(device)),
        **plain,
    )


def join(parts: Sequence[Rollout]) -> Rollout:
    # The result has the rows of every part, in the order the parts are given. `parts` must not be
    # empty.
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


def _decide(boards: Boards, guess: Guess, *, is_learner: Tensor, learner: Policy, opponent: Policy) -> Tensor:
    # Runs each policy once, on the rows where it acts, and returns the logits [n, k] of every row.
    # When the learner acts in every row, as in self-play, it reads the boards whole and no board
    # is copied.
    if bool(is_learner.all()):
        return learner(boards, guess)
    logits = torch.empty(0)
    for rows, policy in ((is_learner, learner), (~is_learner, opponent)):
        if not bool(rows.any()):
            continue
        chosen_boards = {name: array[rows] for name, array in boards.items()}
        policy_logits = policy(chosen_boards, Guess(deck=guess.deck[rows], hand=guess.hand[rows]))
        if not logits.numel():
            logits = torch.full((len(is_learner), policy_logits.shape[1]), float("-inf"))
        logits[rows] = policy_logits
    return logits


def _truth(batch: GameBatch, seats: Tensor) -> tuple[Tensor, Tensor]:
    # Returns `deck_truth` and `hand_truth`, each [n, cards], for the opponent of the acting seat in each game in play.
    hidden = batch.hidden_cards()
    in_play = torch.arange(len(seats))
    # `1 - seats` is the opponent of the seat acting in each game.
    hands = torch.from_numpy(hidden.hands)[in_play, 1 - seats].float()
    unseen = hands + torch.from_numpy(hidden.libraries)[in_play, 1 - seats].float()
    return unseen / unseen.sum(dim=-1, keepdim=True).clamp_min(1.0), (hands > 0).float()


def _finished_rows(parts: Sequence[Rollout], finished: Sequence[GameResult], *, abandoned: set[int]) -> Rollout:
    # Joins the rows of every step of the round, leaving out the abandoned games, and fills in each
    # row's result from its game's result.
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
    # history, changing `history` and `history_count` in place. `games` and `seats` are int64 [n].
    # The two tensors of `history` are [matchups, 2, h, cards], one row per matchup of the round,
    # with each seat's guesses newest first. `history_count` is int64 [matchups, 2], how many of a
    # seat's h places contain a guess.
    history_length = history.deck.shape[2]
    earlier = Guess(deck=history.deck[games, seats], hand=history.hand[games, seats])
    mask = torch.arange(history_length) < history_count[games, seats][:, None]
    guess = guesser(boards, Earlier(guesses=earlier, mask=mask))
    # The new guess goes in front and the oldest falls off the end, so the history stays newest first.
    history.deck[games, seats] = torch.cat([guess.deck[:, None], earlier.deck], dim=1)[:, :history_length]
    history.hand[games, seats] = torch.cat([guess.hand[:, None], earlier.hand], dim=1)[:, :history_length]
    history_count[games, seats] = (history_count[games, seats] + 1).clamp_max(history_length)
    return guess


def _game_result(ended: Finished, *, learner_seat: int, self_play: bool) -> GameResult:
    seat_0_result = {Outcome.SEAT_0_WON: 1.0, Outcome.SEAT_1_WON: 0.0, Outcome.DRAW: 0.5}[ended.outcome]
    return GameResult(
        game=ended.game,
        decks=ended.decks,
        learner_seat=learner_seat,
        learner_result=seat_0_result if learner_seat == 0 else 1.0 - seat_0_result,
        self_play=self_play,
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
    the pool, and `history` is how many earlier guesses the guesser is
    shown. The learner's first seat and every action are drawn from
    `generator`.

    The learner's seat alternates down the list. It takes a seat drawn at
    random in the first game, the other seat in the second game, and so on.
    When `opponent` is the same policy as `learner`, which is self-play,
    `Rollout.learner` is True for the decisions of both seats.
    `GameResult.learner_result` is still the result of the seat the learner
    was given.

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
    # This compares with `==`, not `is`, because a bound method is a new object each time it is
    # read from its model.
    self_play = opponent == learner
    parts: list[Rollout] = []
    finished: list[GameResult] = []
    abandoned: set[int] = set()
    pending = batch.decisions()
    while len(pending.games):
        games = torch.from_numpy(pending.games)
        seats = torch.from_numpy(pending.seats)
        boards = {name: torch.from_numpy(array) for name, array in pending.boards.items()}
        guess = _guess(guesser, boards, games=games, seats=seats, history=guess_history, history_count=history_count)
        is_learner = torch.ones_like(seats, dtype=torch.bool) if self_play else seats == learner_seat[games]
        logits = _decide(boards, guess, is_learner=is_learner, learner=learner, opponent=opponent)
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
                finished.append(_game_result(ended, learner_seat=int(learner_seat[ended.game]), self_play=self_play))
        pending = batch.decisions()
    rollout = _finished_rows(parts, finished, abandoned=abandoned)
    return Collected(rollout=rollout, games=tuple(finished), abandoned=len(abandoned), lost=0)

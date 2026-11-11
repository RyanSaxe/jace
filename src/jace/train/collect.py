"""Play games in a batch and record every decision of the games that finish.

`collect` steps a `GameBatch` with a learner policy in one seat and an
opponent policy in the other, and returns the decisions of complete games
as a `Rollout`. Games still running when it stops are not dropped, which
would favour short games: they are handed back as a `Pending` value, and the
next call continues them. Each model runs once per step over every game
that needs it; the only Python loop over games visits the few that ended on
a step.
"""

from dataclasses import dataclass, fields

import numpy as np
import torch
from torch import Tensor

from jace.cards import Deck
from jace.engine import SEATS, Finished, GameBatch, Outcome
from jace.play import Boards, Earlier, Guess, Guesser, Policy

__all__ = ("Collected", "GameResult", "Pending", "Rollout", "collect", "select")


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
class Pending:
    """Everything `collect` must remember between calls on one batch.

    `rows` are the decisions of games that have not finished, with `results`
    not yet known. `history` holds each slot's and seat's most recent
    guesses, newest first, as `[slots, 2, history, cards]`, and
    `history_count` says how many are real. `learner_seat` is the learner's
    seat in each slot's current game, `game_ids` is that game's id, and
    `next_game` is the first unused id.
    """

    rows: Rollout | None
    history: Guess
    history_count: Tensor
    learner_seat: Tensor
    game_ids: Tensor
    next_game: int


@dataclass(frozen=True, slots=True)
class GameResult:
    game: int
    decks: tuple[Deck, Deck]
    learner_seat: int
    # The learner's result: 1 for a win, 0.5 for a draw, 0 for a loss.
    learner_result: float


@dataclass(frozen=True, slots=True)
class Collected:
    # rollout: the decisions of the games that finished. games: one entry per finished game.
    rollout: Rollout
    games: tuple[GameResult, ...]
    # pending: pass it to the next `collect` call on the same batch.
    pending: Pending


def select(rollout: Rollout, rows: Tensor) -> Rollout:
    # rows: int64 [m] row indices, or bool [n]. Returns those rows as a rollout.
    plain = {
        field.name: getattr(rollout, field.name)[rows]
        for field in fields(Rollout)
        if field.name not in {"boards", "guesses"}
    }
    return Rollout(
        boards={name: array[rows] for name, array in rollout.boards.items()},
        guesses=Guess(deck=rollout.guesses.deck[rows], hand=rollout.guesses.hand[rows]),
        **plain,
    )


def _join(parts: list[Rollout]) -> Rollout:
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


def _fresh(slots: int, *, cards: int, history: int, generator: torch.Generator) -> Pending:
    empty = torch.zeros(slots, SEATS, history, cards)
    return Pending(
        rows=None,
        history=Guess(deck=empty, hand=empty.clone()),
        history_count=torch.zeros(slots, SEATS, dtype=torch.long),
        learner_seat=torch.randint(0, SEATS, (slots,), generator=generator),
        game_ids=torch.arange(slots),
        next_game=slots,
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
    # Returns (deck_truth, hand_truth), each [n, cards], for the opponent of the seat acting in each slot.
    hidden = batch.hidden_cards()
    slot_index = torch.arange(len(seats))
    hands = torch.from_numpy(hidden.hands)[slot_index, 1 - seats].float()
    unseen = hands + torch.from_numpy(hidden.libraries)[slot_index, 1 - seats].float()
    return unseen / unseen.sum(dim=-1, keepdim=True).clamp_min(1.0), (hands > 0).float()


def _split(parts: list[Rollout], finished: list[GameResult]) -> tuple[Rollout, Rollout | None]:
    # Fills in results and returns (rows of finished games, rows of games still running or None).
    rows = _join(parts)
    result_for_seat_0 = {
        game.game: game.learner_result if game.learner_seat == 0 else 1.0 - game.learner_result for game in finished
    }
    seat_0_result = torch.tensor([result_for_seat_0.get(game, float("nan")) for game in rows.games.tolist()])
    results = torch.where(rows.seats == 0, seat_0_result, 1.0 - seat_0_result)
    done = ~results.isnan()
    plain = {
        field.name: getattr(rows, field.name)
        for field in fields(Rollout)
        if field.name not in {"boards", "guesses", "results"}
    }
    everything = Rollout(boards=rows.boards, guesses=rows.guesses, results=results, **plain)
    return select(everything, done), select(everything, ~done) if bool((~done).any()) else None


def _count_learner_rows(counts: dict[int, int], games: Tensor) -> None:
    # Adds one to `counts[game]` for each game id in `games`, int64 [m], in place.
    for game in games.tolist():
        counts[game] = counts.get(game, 0) + 1


def _guess(guesser: Guesser, boards: Boards, *, seats: Tensor, history: Guess, history_count: Tensor) -> Guess:
    # Makes the acting seat's guess in every slot and puts it at the front of that seat's
    # history, in place. history: [slots, 2, h, cards] each; history_count: int64 [slots, 2].
    slots = torch.arange(len(seats))
    shown = history.deck.shape[2]
    earlier = Guess(deck=history.deck[slots, seats], hand=history.hand[slots, seats])
    mask = torch.arange(shown) < history_count[slots, seats][:, None]
    guess = guesser(boards, Earlier(guesses=earlier, mask=mask))
    history.deck[slots, seats] = torch.cat([guess.deck[:, None], earlier.deck[:, :-1]], dim=1)
    history.hand[slots, seats] = torch.cat([guess.hand[:, None], earlier.hand[:, :-1]], dim=1)
    history_count[slots, seats] = (history_count[slots, seats] + 1).clamp_max(shown)
    return guess


def _game_result(ended: Finished, *, game: int, learner_seat: int) -> GameResult:
    seat_0_result = {Outcome.SEAT_0_WON: 1.0, Outcome.SEAT_1_WON: 0.0, Outcome.DRAW: 0.5}[ended.outcome]
    return GameResult(
        game=game,
        decks=ended.decks,
        learner_seat=learner_seat,
        learner_result=seat_0_result if learner_seat == 0 else 1.0 - seat_0_result,
    )


@torch.no_grad()
def collect(
    batch: GameBatch,
    pending: Pending | None,
    *,
    learner: Policy,
    opponent: Policy,
    guesser: Guesser,
    cards: int,
    history: int,
    decisions: int,
    generator: torch.Generator,
) -> Collected:
    """Step `batch` until finished games hold at least `decisions` learner decisions.

    Pass `pending=None` for a batch that has just been started and the
    `Pending` from the previous call afterwards. `cards` is the number of
    cards in the pool and `history` is how many earlier guesses the guesser
    is shown. The learner's seat is drawn at random for each new game.
    Returns the decisions and results of the finished games, and the state
    to continue from.
    """
    pending_decisions = batch.decisions()
    slots = len(pending_decisions.seats)
    state = pending if pending is not None else _fresh(slots, cards=cards, history=history, generator=generator)
    learner_seat, game_ids, next_game = state.learner_seat.clone(), state.game_ids.clone(), state.next_game
    guesses = Guess(deck=state.history.deck.clone(), hand=state.history.hand.clone())
    guess_count = state.history_count.clone()
    parts: list[Rollout] = [] if state.rows is None else [state.rows]
    finished: list[GameResult] = []
    learner_rows: dict[int, int] = {}
    if state.rows is not None:
        _count_learner_rows(learner_rows, state.rows.games[state.rows.learner])
    finished_learner_rows = 0
    while finished_learner_rows < decisions:
        seats = torch.from_numpy(pending_decisions.seats.copy())
        boards = {name: torch.from_numpy(np.array(array)) for name, array in pending_decisions.boards.items()}
        guess = _guess(guesser, boards, seats=seats, history=guesses, history_count=guess_count)
        is_learner = seats == learner_seat
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
                games=game_ids.clone(),
                seats=seats,
                learner=is_learner,
                results=torch.full((slots,), float("nan")),
                deck_truth=deck_truth,
                hand_truth=hand_truth,
            )
        )
        _count_learner_rows(learner_rows, game_ids[is_learner])

        for ended in batch.act(actions.numpy()):
            game = int(game_ids[ended.slot])
            finished.append(_game_result(ended, game=game, learner_seat=int(learner_seat[ended.slot])))
            finished_learner_rows += learner_rows.pop(game, 0)
            game_ids[ended.slot] = next_game
            next_game += 1
            learner_seat[ended.slot] = int(torch.randint(0, SEATS, (1,), generator=generator))
            guess_count[ended.slot] = 0
        pending_decisions = batch.decisions()

    rollout, unfinished = _split(parts, finished)
    carried = Pending(
        rows=unfinished,
        history=guesses,
        history_count=guess_count,
        learner_seat=learner_seat,
        game_ids=game_ids,
        next_game=next_game,
    )
    return Collected(rollout=rollout, games=tuple(finished), pending=carried)

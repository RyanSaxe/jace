"""Group a rollout's decisions into one sequence per game and seat.

A rollout lists decisions in the order they were made, with both seats of
many games interleaved. Computing advantages and gathering the guess model's
history both require each seat's earlier and later decisions in the same
game. `sequences` builds that grouping once, as an index table both can use.
Everything here runs as batched tensor operations over all the sequences at
once. The only Python loop, in `advantages`, runs over the positions of the
longest sequence.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from jace.engine import SEATS
from jace.play import Earlier, Guess

__all__ = ("Sequences", "advantages", "earlier_guesses", "sequences")


@dataclass(frozen=True, slots=True)
class Sequences:
    # The (game, seat) sequence each row belongs to, as int64 [n].
    sequence: Tensor
    # Each row's index within its sequence, as int64 [n]. Position 0 is the seat's first decision.
    position: Tensor
    # The row at each position of each sequence, as int64 [sequences, longest]. Places past the end
    # of a shorter sequence are -1.
    table: Tensor


def sequences(games: Tensor, seats: Tensor) -> Sequences:
    # `games` and `seats` are int64 [n], one entry per decision, in the order the decisions were made.
    rows = torch.arange(len(games), device=games.device)
    # One integer per (game, seat), so sorting by it gathers each sequence's rows together.
    keys = games * SEATS + seats
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys[order]
    starts = torch.ones_like(sorted_keys, dtype=torch.bool)
    starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
    sequence_sorted = starts.long().cumsum(0) - 1
    # The stable sort keeps each sequence's rows in time order. A row's position is therefore its
    # index in sorted order minus the sorted index where its sequence starts. `rows` serves here as
    # the sorted indices 0 to n - 1.
    sequence_start = torch.zeros_like(rows).scatter_reduce(0, sequence_sorted, rows, reduce="amin", include_self=False)
    position_sorted = rows - sequence_start[sequence_sorted]
    # Scattering through `order` puts both back from sorted order into the rollout's order.
    sequence = torch.empty_like(rows).scatter(0, order, sequence_sorted)
    position = torch.empty_like(rows).scatter(0, order, position_sorted)
    count = int(sequence_sorted[-1]) + 1 if len(rows) else 0
    longest = int(position.max()) + 1 if len(rows) else 0
    table = torch.full((count, longest), -1, dtype=torch.long, device=games.device)
    table[sequence, position] = rows
    return Sequences(sequence=sequence, position=position, table=table)


def advantages(values: Tensor, results: Tensor, *, layout: Sequences, lam: float) -> Tensor:
    """Compute generalized advantage estimates with no discount, as `[n]`.

    `values` is the win chance predicted at each row, and `results` is the
    game's final result from that row's seat: 1 for a win, 0.5 for a draw
    and 0 for a loss. The result is the only reward, and it comes after the
    seat's last decision.

    `lam`, between 0 and 1, sets how much a row's advantage depends on later
    decisions. At 0, the advantage is the change from the row's prediction to
    the seat's next prediction in the game, or to the result after the seat's
    last decision. At 1, it is the game's result minus the row's prediction. In
    between, a change k decisions ahead counts with weight `lam**k`.
    """
    valid = layout.table >= 0
    rows = layout.table.clamp_min(0)
    value = values[rows] * valid
    # `later_value` is `value` shifted one place left, so each position lines up with the next
    # decision's prediction. A sequence's last decision has no next prediction, and the game's
    # result takes its place.
    later_valid = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], dim=1)
    later_value = torch.cat([value[:, 1:], torch.zeros_like(value[:, :1])], dim=1)
    surprise = (torch.where(later_valid, later_value, results[rows]) - value) * valid
    estimate = torch.zeros_like(surprise)
    # `running` starts as zeros, one per sequence. With no rows there is no column 0 to take that
    # shape from, and a sum over the zero columns gives the same zeros.
    running = torch.zeros_like(surprise[:, 0]) if surprise.shape[1] else surprise.sum(dim=1)
    # `surprise` is zero past a sequence's end, so those positions add nothing to the estimates before them.
    for step in reversed(range(surprise.shape[1])):
        running = surprise[:, step] + lam * running
        estimate[:, step] = running
    return estimate[layout.sequence, layout.position]


def earlier_guesses(guesses: Guess, layout: Sequences, *, rows: Tensor, history: int) -> Earlier:
    """Gather, for each of `rows`, the same seat's last `history` guesses in that game.

    `guesses` has one guess per rollout row. The earlier guesses come back
    as `[m, history, cards]` tensors, newest first, with a bool mask
    `[m, history]` that is False where the seat had made fewer earlier
    decisions in the game. Where the mask is False, the guess is a
    placeholder.

    The earlier guesses are indexed out of `guesses` when they are needed,
    so no guess is stored twice.
    """
    steps_back = torch.arange(1, history + 1, device=rows.device)
    positions = layout.position[rows][:, None] - steps_back
    # A position before the seat's first decision is negative, and the mask marks it False. It is
    # clamped to 0 below so the lookup stays in range, and its guess is a placeholder.
    mask = positions >= 0
    earlier_rows = layout.table[layout.sequence[rows][:, None], positions.clamp_min(0)]
    return Earlier(guesses=Guess(deck=guesses.deck[earlier_rows], hand=guesses.hand[earlier_rows]), mask=mask)

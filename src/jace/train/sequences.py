"""Group a rollout's decisions into one sequence per game and seat.

A rollout stores decisions in the order they were made, with both seats of
many games interleaved. Advantages and the guess model's history both need
"the same seat's earlier and later decisions in this game", so this module
builds that grouping once as an index table. Everything here is a batched
tensor operation over all sequences; the only Python loop runs over the
positions of the longest sequence.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from jace.play import Earlier, Guess

__all__ = ("Sequences", "advantages", "earlier_guesses", "sequences")


@dataclass(frozen=True, slots=True)
class Sequences:
    # sequence: int64 [n], which (game, seat) sequence each row belongs to.
    sequence: Tensor
    # position: int64 [n], the row's index within its sequence, in time order.
    position: Tensor
    # table: int64 [sequences, longest], the row at each position, or -1 past a sequence's end.
    table: Tensor


def sequences(games: Tensor, seats: Tensor) -> Sequences:
    # games, seats: int64 [n], rows in the order the decisions were made.
    rows = torch.arange(len(games), device=games.device)
    keys = games * 2 + seats
    order = torch.argsort(keys, stable=True)
    sorted_keys = keys[order]
    starts = torch.ones_like(sorted_keys, dtype=torch.bool)
    starts[1:] = sorted_keys[1:] != sorted_keys[:-1]
    sequence_sorted = starts.long().cumsum(0) - 1
    first_row = torch.zeros_like(rows).scatter_reduce(0, sequence_sorted, rows, reduce="amin", include_self=False)
    position_sorted = rows - first_row[sequence_sorted]
    sequence = torch.empty_like(rows).scatter(0, order, sequence_sorted)
    position = torch.empty_like(rows).scatter(0, order, position_sorted)
    count = int(sequence_sorted[-1]) + 1 if len(rows) else 0
    longest = int(position.max()) + 1 if len(rows) else 0
    table = torch.full((count, longest), -1, dtype=torch.long, device=games.device)
    table[sequence, position] = rows
    return Sequences(sequence=sequence, position=position, table=table)


def advantages(values: Tensor, results: Tensor, *, layout: Sequences, lam: float) -> Tensor:
    """Compute generalized advantage estimates with no discount, as `[n]`.

    `values` is the win chance the actor predicted at each row and `results`
    is the game's final result from that row's seat (1, 0.5 or 0). The only
    reward is the result, so each step's surprise is the change in predicted
    win chance, and the last step's is the result minus the last prediction.
    `lam` in 0..1 sets how far a later surprise reaches back: 0 credits only
    the next step, 1 credits the whole rest of the game.
    """
    valid = layout.table >= 0
    rows = layout.table.clamp_min(0)
    value = values[rows] * valid
    later_valid = torch.cat([valid[:, 1:], torch.zeros_like(valid[:, :1])], dim=1)
    later_value = torch.cat([value[:, 1:], torch.zeros_like(value[:, :1])], dim=1)
    surprise = (torch.where(later_valid, later_value, results[rows]) - value) * valid
    estimate = torch.zeros_like(surprise)
    running = torch.zeros_like(surprise[:, 0]) if surprise.shape[1] else surprise.sum(dim=1)
    for step in reversed(range(surprise.shape[1])):
        running = surprise[:, step] + lam * running
        estimate[:, step] = running
    return estimate[layout.sequence, layout.position]


def earlier_guesses(guesses: Guess, layout: Sequences, *, rows: Tensor, history: int) -> Earlier:
    """Gather, for each of `rows`, the same seat's last `history` guesses in that game.

    The guesses come back as `[m, history, cards]` tensors, most recent
    first, with a bool mask `[m, history]` that is False where the game had
    fewer earlier decisions. They are looked up by index from
    `guesses`, which holds one guess per rollout row, so nothing is stored
    twice.
    """
    back = torch.arange(1, history + 1, device=rows.device)
    positions = layout.position[rows][:, None] - back
    mask = positions >= 0
    earlier = layout.table[layout.sequence[rows][:, None], positions.clamp_min(0)]
    return Earlier(guesses=Guess(deck=guesses.deck[earlier], hand=guesses.hand[earlier]), mask=mask)

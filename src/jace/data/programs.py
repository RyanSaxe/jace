"""Turn each card's engine program into the token ids the card encoder reads.

A program is the engine's own definition of a card: a flat run of (kind,
value) integer pairs, about fifty pairs per card. Rules text describes what
a card does, and the program defines it in the engine's terms. A card from
a set nobody has played has a program too.

This package does not import the engine, so the caller passes in the
engine's vocabulary of pairs. The vocabulary comes from the engine and not
from a pool, so a pair gets the same token id in every pool.
"""

from collections.abc import Mapping

import torch
from torch import Tensor

from jace.cards import (
    FIRST_PROGRAM_TOKEN,
    MAX_PROGRAM_TOKENS,
    NO_PROGRAM_TOKEN,
    PADDING_TOKEN,
    UNKNOWN_TOKEN,
    CardPool,
)

__all__ = ("ProgramTooLongError", "program_tokens")


class ProgramTooLongError(ValueError):
    def __init__(self, name: str, *, pairs: int) -> None:
        super().__init__(
            f"{name}'s engine program has {pairs} pairs, more than the {MAX_PROGRAM_TOKENS} the card encoder can read"
        )


def program_tokens(pool: CardPool, *, vocabulary: Mapping[tuple[int, int], int]) -> Tensor:
    """Return int64 `[cards, longest]`, each card's program as token ids, one row per card of `pool` in card id order.

    `vocabulary` gives each (kind, value) pair a number from zero, and the
    pair's token id is `FIRST_PROGRAM_TOKEN` plus that number. A pair
    missing from `vocabulary` becomes `UNKNOWN_TOKEN`. A card with an empty
    program, one the engine cannot play, gets the single token
    `NO_PROGRAM_TOKEN`. Every row therefore has at least one token that is
    not padding for the card encoder to average. Rows shorter than the
    longest are padded with `PADDING_TOKEN`.

    Raises `ProgramTooLongError` when a program has more than
    `MAX_PROGRAM_TOKENS` pairs.
    """
    rows: list[list[int]] = []
    for card in range(len(pool)):
        program = pool[card].program
        pairs = zip(program[0::2], program[1::2], strict=True)
        row = [FIRST_PROGRAM_TOKEN + vocabulary[pair] if pair in vocabulary else UNKNOWN_TOKEN for pair in pairs]
        if len(row) > MAX_PROGRAM_TOKENS:
            raise ProgramTooLongError(pool[card].name, pairs=len(row))
        rows.append(row or [NO_PROGRAM_TOKEN])
    tokens = torch.full((len(rows), max(map(len, rows))), PADDING_TOKEN, dtype=torch.int64)
    for card, row in enumerate(rows):
        tokens[card, : len(row)] = torch.tensor(row)
    return tokens

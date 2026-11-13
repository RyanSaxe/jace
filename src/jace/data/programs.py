"""Turn each card's engine program into token ids, or into a starting vector.

A program is the engine's own definition of a card: a flat run of (kind,
value) integer pairs, about fifty per card. It says what the card does in
the engine's terms, which rules text only describes, and it exists for a
card from a set nobody has played. `program_tokens` writes each pair as
the token id the card encoder reads, from a vocabulary the caller passes
in, because this package does not import the engine. `program_vectors` has no learned
parameters. It counts each pair and each two adjacent pairs, weights the
ones few cards share above the ones every card has, and keeps the
directions along which the cards differ most. Those directions are fitted
on the pool passed in, so vectors from two different pools do not share
axes.
"""

from collections import Counter
from collections.abc import Mapping
from itertools import pairwise

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
from jace.data.store import SMALLEST_SPREAD

__all__ = ("NoProgramsError", "ProgramTooLongError", "program_tokens", "program_vectors")


class NoProgramsError(ValueError):
    def __init__(self) -> None:
        super().__init__("no card in the pool has an engine program, so there is nothing to make program vectors from")


class ProgramTooLongError(ValueError):
    def __init__(self, name: str, *, pairs: int) -> None:
        super().__init__(
            f"{name}'s engine program has {pairs} pairs, more than the {MAX_PROGRAM_TOKENS} the card encoder can read"
        )


def program_tokens(pool: CardPool, *, vocabulary: Mapping[tuple[int, int], int]) -> Tensor:
    """Return int64 `[cards, longest]`, each card's program as token ids, one row per card of `pool` in card id order.

    `vocabulary` numbers (kind, value) pairs from zero, and a pair it holds
    becomes `FIRST_PROGRAM_TOKEN` plus that number. A pair it lacks becomes
    `UNKNOWN_TOKEN`. A card with an empty program, one the engine cannot
    play, gets the single token `NO_PROGRAM_TOKEN`, so every row has at
    least one token that is not padding. Rows shorter than the longest are
    padded with `PADDING_TOKEN`. Raises `ProgramTooLongError` when a program
    has more than `MAX_PROGRAM_TOKENS` pairs.
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


def program_vectors(pool: CardPool, *, width: int) -> Tensor:
    """Return float32 `[cards, columns + 1]`, one row per card of `pool` in card id order.

    The first columns are the card's place along the `width` strongest
    directions of the weighted counts, or along fewer when the cards with a
    program span fewer. Each has mean zero and spread one over the cards
    that have a program. A card with an empty program gets zero in those
    columns and one in the last column, which is zero for every other card.
    Raises `NoProgramsError` when every program is empty.
    """
    programs = [pool[card].program for card in range(len(pool))]
    known = [card for card, program in enumerate(programs) if program]
    if not known:
        raise NoProgramsError
    card_entries = [_pair_counts(programs[card]) for card in known]
    column_of = {
        entry: index for index, entry in enumerate(sorted({entry for entries in card_entries for entry in entries}))
    }
    table = torch.zeros(len(known), len(column_of))  # [known, vocabulary] how often each card has each entry
    for row, entries in enumerate(card_entries):
        table[row, [column_of[entry] for entry in entries]] = torch.tensor(list(entries.values()), dtype=torch.float32)
    # An entry every card has gets weight zero; one that a single card has gets the most.
    inverse_frequency = (len(known) / (table > 0).sum(dim=0)).log()  # [vocabulary]
    weighted = table.log1p() * inverse_frequency
    # Each card's row is scaled to length one, so a card with a long program does not outweigh the rest.
    weighted = weighted / weighted.pow(2).sum(dim=1, keepdim=True).sqrt().clamp_min(SMALLEST_SPREAD)
    # The SVD of the centred rows is principal component analysis: `left * strengths` is each
    # card's place along the directions the cards differ most in, strongest first.
    left, strengths, _ = torch.linalg.svd(weighted - weighted.mean(dim=0), full_matrices=False)
    reduced = (left * strengths)[:, :width]  # [known, columns]
    spread = reduced.std(dim=0, correction=0).clamp_min(SMALLEST_SPREAD)
    vectors = torch.zeros(len(pool), reduced.shape[1] + 1)
    vectors[:, -1] = 1.0
    vectors[known] = torch.cat([reduced / spread, torch.zeros(len(known), 1)], dim=1)
    return vectors


def _pair_counts(program: tuple[int, ...]) -> Counter[tuple[int, ...]]:
    # Counts every (kind, value) pair and every two adjacent pairs, the latter as (kind, value, kind, value).
    pairs = list(zip(program[0::2], program[1::2], strict=True))
    return Counter([*pairs, *[first + second for first, second in pairwise(pairs)]])

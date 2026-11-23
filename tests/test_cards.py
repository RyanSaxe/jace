"""Check the card encoder and the table frozen from it through `jace.cards`.

Each program is a short run of made-up token ids, padded the way real programs
are. The programs have different lengths, so some rows have more padding than
others. The encoder is tiny and has the weights PyTorch initialises it with. It
is never trained, because what these tests check is true for any weights.
"""

from collections.abc import Sequence

import torch

from jace.cards import (
    FIRST_PROGRAM_TOKEN,
    NO_PROGRAM_TOKEN,
    PADDING_TOKEN,
    UNKNOWN_TOKEN,
    CardEncoder,
    frozen_table,
)

__all__ = ()

_PROGRAMS = (
    (FIRST_PROGRAM_TOKEN, FIRST_PROGRAM_TOKEN + 1, FIRST_PROGRAM_TOKEN + 2),
    (NO_PROGRAM_TOKEN,),
    (FIRST_PROGRAM_TOKEN + 3, UNKNOWN_TOKEN),
    tuple(range(FIRST_PROGRAM_TOKEN, FIRST_PROGRAM_TOKEN + 6)),
    (FIRST_PROGRAM_TOKEN + 4,),
)
_VOCABULARY = FIRST_PROGRAM_TOKEN + 6


def _encoder(programs: Sequence[tuple[int, ...]]) -> CardEncoder:
    program = torch.full((len(programs), max(map(len, programs))), PADDING_TOKEN)
    for card, tokens in enumerate(programs):
        program[card, : len(tokens)] = torch.tensor(tokens)
    return CardEncoder(program, width=8, program_vocabulary=_VOCABULARY, program_layers=1, heads=2)


def test_a_cards_vector_depends_only_on_its_own_program_whatever_the_other_cards_are() -> None:
    torch.manual_seed(0)
    encoder = _encoder(_PROGRAMS)
    # Three of the cards, in another order. Card 3 has the longest program and is left out, so these rows
    # have less padding than before.
    chosen = [4, 0, 2]
    fewer = _encoder([_PROGRAMS[card] for card in chosen])
    fewer.load_state_dict(encoder.state_dict())

    with torch.no_grad():
        assert torch.allclose(fewer.table(), encoder.table()[chosen], atol=1e-6)


def test_a_frozen_table_gives_the_encoders_vectors_and_has_nothing_to_train() -> None:
    torch.manual_seed(0)
    encoder = _encoder(_PROGRAMS)
    cards = torch.tensor([[2, 0], [4, 4]])

    table = frozen_table(encoder)

    assert torch.allclose(table(cards), encoder(cards).detach())
    assert (table.cards, table.width) == (encoder.cards, encoder.width)
    assert not list(table.parameters())
    assert not table(cards).requires_grad

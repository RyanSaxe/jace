"""Check the card encoder and the table frozen from it through `jace.cards`.

The inputs are made up: random text vectors and printed facts, and short
programs of made-up token ids of different lengths, padded as the real ones
are. The encoder is tiny, starts from PyTorch's own weights and is never
trained, because what is checked holds for any weights.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from jace.cards import (
    FIRST_PROGRAM_TOKEN,
    NO_PROGRAM_TOKEN,
    PADDING_TOKEN,
    UNKNOWN_TOKEN,
    CardEncoder,
    CardInputs,
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


def _encoder(programs: Sequence[tuple[int, ...]], *, text: Tensor, printed: Tensor) -> CardEncoder:
    program = torch.full((len(programs), max(map(len, programs))), PADDING_TOKEN)
    for card, tokens in enumerate(programs):
        program[card, : len(tokens)] = torch.tensor(tokens)
    inputs = CardInputs(text=text, program=program, printed=printed)
    return CardEncoder(inputs, width=8, program_vocabulary=_VOCABULARY, program_layers=1, heads=2)


def test_a_cards_vector_depends_only_on_its_own_inputs_whatever_the_other_cards_are() -> None:
    torch.manual_seed(0)
    text, printed = torch.randn(len(_PROGRAMS), 6), torch.randn(len(_PROGRAMS), 4)
    encoder = _encoder(_PROGRAMS, text=text, printed=printed)
    # Fewer cards in another order, without card 3, whose program is the longest, so the rows are padded less.
    chosen = [4, 0, 2]
    fewer = _encoder([_PROGRAMS[card] for card in chosen], text=text[chosen], printed=printed[chosen])
    fewer.load_state_dict(encoder.state_dict())

    with torch.no_grad():
        assert torch.allclose(fewer.table(), encoder.table()[chosen], atol=1e-6)


def test_a_frozen_table_gives_the_encoders_vectors_and_has_nothing_to_train() -> None:
    torch.manual_seed(0)
    encoder = _encoder(_PROGRAMS, text=torch.randn(len(_PROGRAMS), 6), printed=torch.randn(len(_PROGRAMS), 4))
    cards = torch.tensor([[2, 0], [4, 4]])

    table = frozen_table(encoder)

    assert torch.allclose(table(cards), encoder(cards).detach())
    assert (table.cards, table.width) == (encoder.cards, encoder.width)
    assert not list(table.parameters())
    assert not table(cards).requires_grad

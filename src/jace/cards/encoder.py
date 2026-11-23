"""The card vectors every model reads: a program encoder, or a fixed table of its output.

`CardEncoder` makes a card's vector from its `managym` engine program alone,
which is known before anyone plays the card. A card's vector depends only on
its own program, so a card the encoder was not trained on still gets a vector
from its own mechanics.

The drafter's pretraining trains the encoder. Every later model reads
`frozen_table(encoder)`, a `CardTable` that contains the vectors and has
nothing to train, so deck building and play cannot change the vectors the
drafter learned. Models take either one as `CardVectors`.

The encoder reads a program as token ids, one for each (kind, value) pair.
The pairs of the engine's vocabulary take the ids from
`FIRST_PROGRAM_TOKEN` on, and the lower ids have special uses.
`PADDING_TOKEN` pads a row, `NO_PROGRAM_TOKEN` is the single token of a
card the engine cannot play, and `UNKNOWN_TOKEN` replaces a pair the
vocabulary lacks. The encoder learns a position for each of the first
`MAX_PROGRAM_TOKENS` tokens, so the engine's longest program, at 149 pairs,
fits.
"""

from typing import Protocol, override

import torch
from torch import Tensor, nn

from jace.modules import encode_real, transformer

__all__ = (
    "FIRST_PROGRAM_TOKEN",
    "MAX_PROGRAM_TOKENS",
    "NO_PROGRAM_TOKEN",
    "PADDING_TOKEN",
    "UNKNOWN_TOKEN",
    "CardEncoder",
    "CardTable",
    "CardVectors",
    "frozen_table",
    "own_parameters",
)

PADDING_TOKEN = 0
NO_PROGRAM_TOKEN = 1
UNKNOWN_TOKEN = 2
FIRST_PROGRAM_TOKEN = 3
MAX_PROGRAM_TOKENS = 256


class CardVectors(Protocol):
    # The interface through which a model reads its card vectors. `CardEncoder` and `CardTable` both provide it.
    # Calling it with int64 card ids of any shape [...] returns float [..., width].

    @property
    def cards(self) -> int: ...

    @property
    def width(self) -> int: ...

    @property
    def device(self) -> torch.device: ...

    def __call__(self, cards: Tensor) -> Tensor: ...


class CardEncoder(nn.Module):
    """Encode each card's engine program as one vector of `width` numbers.

    `program` is int64 `[cards, longest]`, one row per card in card id order,
    padded with `PADDING_TOKEN`, as `jace.data.program_tokens` returns it.
    `program_vocabulary` is the number of program token ids, which is
    `FIRST_PROGRAM_TOKEN` plus the size of the engine's vocabulary. A
    transformer encoder of `program_layers` layers, with `heads` attention
    heads, reads the program. Each token gets a learned position as well as
    its learned embedding, so a card's vector depends on the order of its
    program's pairs and not only on which pairs it contains.

    The program tensor is a buffer that is not saved in a checkpoint, because
    every run rebuilds it from the engine catalog. A checkpoint therefore
    contains the weights only.
    """

    def __init__(
        self, program: Tensor, *, width: int, program_vocabulary: int, program_layers: int, heads: int
    ) -> None:
        super().__init__()
        self.program: Tensor
        self.register_buffer("program", program.detach().clone(), persistent=False)
        self.width: int = width
        self.program_tokens: nn.Embedding = nn.Embedding(program_vocabulary, width, padding_idx=PADDING_TOKEN)
        self.program_positions: nn.Embedding = nn.Embedding(MAX_PROGRAM_TOKENS, width)
        self.program_encoder: nn.TransformerEncoder = transformer(width, layers=program_layers, heads=heads)
        self.norm: nn.LayerNorm = nn.LayerNorm(width)

    @property
    def cards(self) -> int:
        return self.program.shape[0]

    @property
    def device(self) -> torch.device:
        return self.program.device

    def table(self) -> Tensor:
        # Returns float [cards, width], every card's vector.
        # `program_tokens` gives every row at least one real token, so the mean over real tokens never divides by zero.
        real = self.program != PADDING_TOKEN  # bool [cards, longest]
        positions = torch.arange(self.program.shape[1], device=self.device)
        tokens = self.program_tokens(self.program) + self.program_positions(positions)
        encoded = encode_real(self.program_encoder, tokens, mask=real)  # [cards, longest, width], zero at padding
        program = encoded.sum(dim=1) / real.sum(dim=1, keepdim=True)
        return self.norm(program)

    @override
    def forward(self, cards: Tensor) -> Tensor:
        # `cards` is int64 card ids of any shape [...]. Returns float [..., width].
        return self.table()[cards]


class CardTable(nn.Module):
    """A fixed vector for every card, with nothing to train.

    `table` is float `[cards, width]` and is saved in a checkpoint, so a
    run that loads the checkpoint reads the same vectors without the
    encoder or its inputs.
    """

    def __init__(self, table: Tensor) -> None:
        super().__init__()
        self.table: Tensor
        self.register_buffer("table", table.detach().clone())

    @property
    def cards(self) -> int:
        return self.table.shape[0]

    @property
    def width(self) -> int:
        return self.table.shape[1]

    @property
    def device(self) -> torch.device:
        return self.table.device

    @override
    def forward(self, cards: Tensor) -> Tensor:
        # `cards` is int64 card ids of any shape [...]. Returns float [..., width].
        return self.table[cards]


def frozen_table(encoder: CardEncoder) -> CardTable:
    with torch.no_grad():
        return CardTable(encoder.table())


def own_parameters(model: nn.Module, embedding: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return the parameters of `model` that are not those of `embedding`, the card vectors it shares.

    A model that reads cards has its card vectors as a submodule, so
    `model.parameters()` includes the embedding's parameters. An embedding
    in training has its own optimizer, and a parameter in two optimizers is
    stepped twice for one loss. Build a model's optimizer from this function
    instead.

    The embedding's parameters are recognised by identity, so the embedding
    may sit at any depth in `model` and under any attribute name.
    """
    shared = {id(parameter) for parameter in embedding.parameters()}
    return tuple(parameter for parameter in model.parameters() if id(parameter) not in shared)

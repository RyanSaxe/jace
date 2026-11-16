"""The card vectors every model reads: an encoder of each card's content, or a fixed table of its output.

`CardEncoder` makes a card's vector from three inputs, all known before
anyone plays the card. They are the card's printed text as a language model
embeds it, its engine program, and a few facts printed on it. A card's
vector depends on that card's inputs alone, so a card the encoder was not
trained on still gets a vector from its own content.

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

from dataclasses import dataclass
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
    "CardInputs",
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

# The encoder joins three vectors of its width, one each for the text, the program and the printed facts.
_PARTS = 3


@dataclass(frozen=True, slots=True)
class CardInputs:
    """What the encoder reads about each card, one row per card in card id order.

    `text` is float32 `[cards, text width]`, each card's printed text
    embedded by a language model, as `jace.data.text_vectors` returns it.
    `program` is int64 `[cards, longest]`, the engine program as token ids
    padded with `PADDING_TOKEN`, as `jace.data.program_tokens` returns it.
    `printed` is float32 `[cards, printed facts]`, the `printed` tensor of
    `jace.data.SetCards`.
    """

    text: Tensor
    program: Tensor
    printed: Tensor


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
    """Encode each card's text, program and printed facts as one vector of `width` numbers.

    `program_vocabulary` is the number of program token ids, which is
    `FIRST_PROGRAM_TOKEN` plus the size of the engine's vocabulary. A
    transformer encoder of `program_layers` layers, with `heads` attention
    heads, reads the program. Each token gets a learned position as well as
    its learned embedding, so a card's vector depends on the order of its
    program's pairs and not only on which pairs it contains.

    The inputs are buffers that are not saved in a checkpoint, because every
    run rebuilds them from data. A checkpoint therefore contains the weights
    only.
    """

    def __init__(
        self, inputs: CardInputs, *, width: int, program_vocabulary: int, program_layers: int, heads: int
    ) -> None:
        super().__init__()
        self.text: Tensor
        self.program: Tensor
        self.printed: Tensor
        self.register_buffer("text", inputs.text.detach().clone(), persistent=False)
        self.register_buffer("program", inputs.program.detach().clone(), persistent=False)
        self.register_buffer("printed", inputs.printed.detach().clone(), persistent=False)
        self.width: int = width
        self.text_projection: nn.Linear = nn.Linear(inputs.text.shape[1], width)
        self.printed_projection: nn.Linear = nn.Linear(inputs.printed.shape[1], width)
        self.program_tokens: nn.Embedding = nn.Embedding(program_vocabulary, width, padding_idx=PADDING_TOKEN)
        self.program_positions: nn.Embedding = nn.Embedding(MAX_PROGRAM_TOKENS, width)
        self.program_encoder: nn.TransformerEncoder = transformer(width, layers=program_layers, heads=heads)
        self.join: nn.Sequential = nn.Sequential(nn.Linear(_PARTS * width, width), nn.GELU(), nn.Linear(width, width))
        self.norm: nn.LayerNorm = nn.LayerNorm(width)

    @property
    def cards(self) -> int:
        return self.text.shape[0]

    @property
    def device(self) -> torch.device:
        return self.text.device

    def table(self) -> Tensor:
        # Returns float [cards, width], every card's vector.
        # `program_tokens` gives every row at least one real token, so the mean over real tokens never divides by zero.
        real = self.program != PADDING_TOKEN  # bool [cards, longest]
        positions = torch.arange(self.program.shape[1], device=self.device)
        tokens = self.program_tokens(self.program) + self.program_positions(positions)
        encoded = encode_real(self.program_encoder, tokens, mask=real)  # [cards, longest, width], zero at padding
        program = encoded.sum(dim=1) / real.sum(dim=1, keepdim=True)
        parts = torch.cat([self.text_projection(self.text), program, self.printed_projection(self.printed)], dim=-1)
        return self.norm(self.join(parts))

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

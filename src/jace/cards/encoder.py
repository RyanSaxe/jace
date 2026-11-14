"""The card vectors every model reads: an encoder of each card's content, or a fixed table of its output.

`CardEncoder` makes a card's vector from three things known before anyone
plays the card: its printed text as a language model embeds it, its engine
program and a few printed facts. A card's vector depends on that card's
inputs alone, so a card the encoder was not trained on gets a vector from
its own content. The drafter's pretraining trains the encoder. Every later
model reads `frozen_table(encoder)`, a `CardTable` that holds the vectors
and has nothing to train, so deck building and play cannot move what the
drafter learned. Models take either one as `CardVectors`.

A program is read as tokens. The ids below are the ones an engine pair
cannot take: a row is padded with `PADDING_TOKEN`, a card the engine cannot
play has the single token `NO_PROGRAM_TOKEN`, and a pair the vocabulary
lacks becomes `UNKNOWN_TOKEN`. The vocabulary's pairs take the ids from
`FIRST_PROGRAM_TOKEN` on. The encoder learns one position for each of the
first `MAX_PROGRAM_TOKENS` pairs, and the engine's longest program has 149.
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

# The encoder joins three vectors of its width: the text's, the program's and the printed facts'.
_PARTS = 3


@dataclass(frozen=True, slots=True)
class CardInputs:
    """What the encoder reads about each card, one row per card in card id order.

    `text` is float32 `[cards, text width]`, the card's printed text as a
    language model embeds it. `program` is int64 `[cards, longest]`, the
    engine program as token ids padded with `PADDING_TOKEN`. `printed` is
    float32 `[cards, printed facts]`.
    """

    text: Tensor
    program: Tensor
    printed: Tensor


class CardVectors(Protocol):
    # What a model needs from its card vectors: a `CardEncoder` or a `CardTable`.
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

    The text and the printed facts each pass through a linear map. The
    program's tokens get a learned embedding and a learned position, pass
    through a transformer encoder of `program_layers` layers and are
    averaged over the real tokens. A small network joins the three vectors
    and a layer norm ends it. `program_vocabulary` is the number of token
    ids, `FIRST_PROGRAM_TOKEN` plus the engine vocabulary's size.

    The inputs are buffers that are not saved in a checkpoint: they are
    rebuilt from data in every run, so a checkpoint holds weights only.
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
        real = self.program != PADDING_TOKEN  # bool [cards, longest]; every row has at least one real token
        positions = torch.arange(self.program.shape[1], device=self.device)
        tokens = self.program_tokens(self.program) + self.program_positions(positions)
        encoded = encode_real(self.program_encoder, tokens, mask=real)  # [cards, longest, width], zero at padding
        program = encoded.sum(dim=1) / real.sum(dim=1, keepdim=True)
        parts = torch.cat([self.text_projection(self.text), program, self.printed_projection(self.printed)], dim=-1)
        return self.norm(self.join(parts))

    @override
    def forward(self, cards: Tensor) -> Tensor:
        # cards: int64 card ids of any shape [...]. Returns [..., width].
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
        # cards: int64 card ids of any shape [...]. Returns [..., width].
        return self.table[cards]


def frozen_table(encoder: CardEncoder) -> CardTable:
    with torch.no_grad():
        return CardTable(encoder.table())


def own_parameters(model: nn.Module, embedding: nn.Module) -> tuple[nn.Parameter, ...]:
    """Return the parameters of `model` that are not the shared embedding's.

    A model that reads cards holds its card vectors as a submodule, so
    `model.parameters()` includes the embedding's parameters. A trained
    embedding has its own optimizer, and a parameter held by two optimizers
    is stepped twice for one loss. Build a model's optimizer from this. The
    embedding is recognised by identity, so it may sit at any depth and
    under any attribute name.
    """
    shared = {id(parameter) for parameter in embedding.parameters()}
    return tuple(parameter for parameter in model.parameters() if id(parameter) not in shared)

"""Helpers every model shares: the standard transformer and encoding without padding.

`transformer` builds the one encoder shape the models use, so its settings
live in one place. `encode_real` runs an encoder on the real tokens only,
which matters because the engine pads every board to a fixed size and most
of its rows are empty.
"""

import torch
from torch import Tensor, nn

__all__ = ("DROPOUT", "FEEDFORWARD_EXPANSION", "encode_real", "transformer")

# A transformer layer's feed-forward block is this many times the model width,
# the ratio the original transformer used.
FEEDFORWARD_EXPANSION = 4
# Dropout is off, so a model gives the same output whether or not it is in training mode.
DROPOUT = 0.0


def transformer(width: int, *, layers: int, heads: int) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        width,
        heads,
        dim_feedforward=FEEDFORWARD_EXPANSION * width,
        dropout=DROPOUT,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)


def encode_real(encoder: nn.TransformerEncoder, sequence: Tensor, *, mask: Tensor) -> Tensor:
    """Encode each row's real tokens and return them at their original positions.

    `sequence` is float `[b, n, width]` and `mask` is bool `[b, n]`, true for
    a real token. The result is float `[b, n, width]`, zero wherever `mask`
    is false. Attention costs the square of the sequence length, so the real
    tokens are moved to the front and the columns no row uses are dropped
    before the encoder runs. The encoder has no position input, so moving
    tokens does not change what it computes for them.
    """
    longest = int(mask.sum(dim=1).max())
    order = (~mask).long().argsort(dim=1, stable=True)[:, :longest]  # each row's real tokens first
    index = order[..., None].expand(-1, -1, sequence.shape[-1])
    real = mask.gather(1, order)
    encoded = encoder.forward(sequence.gather(1, index), src_key_padding_mask=~real)
    return torch.zeros_like(sequence).scatter(1, index, encoded * real[..., None])

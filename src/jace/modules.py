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
# `encode_real` encodes a batch's rows in groups of at least this many, each padded only to its own longest row.
_GROUP_ROWS = 64


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
    is false. The encoder's cost follows the length it is given, so the real
    tokens are moved to the front of each row, and the rows are encoded in
    groups of similar length, each cut to its own longest row. The encoder
    has no position input and attends within a row, so neither the order of
    a row's tokens nor the rows grouped with it change what it computes.
    """
    counts = mask.sum(dim=1)
    rows_by_count = counts.argsort()
    groups = rows_by_count.tensor_split(max(len(counts) // _GROUP_ROWS, 1))
    # Each group's longest row is its last. One read from the device gives every group's length.
    lengths = counts[torch.stack([rows[-1] for rows in groups])].tolist()
    order = (~mask).long().argsort(dim=1, stable=True)[:, : max(lengths)]  # each row's real tokens first
    index = order[..., None].expand(-1, -1, sequence.shape[-1])
    real = mask.gather(1, order)
    compact = sequence.gather(1, index)
    encoded = torch.zeros_like(compact)
    for rows, length in zip(groups, lengths, strict=True):
        encoded[rows, :length] = encoder.forward(compact[rows, :length], src_key_padding_mask=~real[rows, :length])
    return torch.zeros_like(sequence).scatter(1, index, encoded * real[..., None])

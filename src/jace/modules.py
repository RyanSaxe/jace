"""Helpers every model shares: the standard transformer and encoding without padding.

`transformer` builds the one encoder shape the models use, so its settings
live in one place. `encode_real` runs an encoder on the real tokens only,
which matters because the engine pads every board to a fixed size and most
of its rows are empty.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from jace.metal_attention import attend_within_rows

__all__ = ("DROPOUT", "FEEDFORWARD_EXPANSION", "DropoutInEncodeRealError", "encode_real", "transformer")

# A transformer layer's feed-forward block is this many times the model width,
# the ratio the original transformer used.
FEEDFORWARD_EXPANSION = 4
# Dropout is off unless a model is built with its own, so a model gives the same output whether or not it is
# in training mode. The drafter's pretraining on human picks builds it with dropout and measures it in eval mode.
DROPOUT = 0.0
# encode_real adds dummy tokens to its packed tokens, at most 1 / 2 ** _PADDING_SHARE_BITS of their number, to
# round their count up to a multiple of a power of two, so the counts of later batches repeat earlier ones. MPS
# builds a graph for every new shape of an op such as a linear layer or a layer norm, and with the counts rounded
# a training block's update on MPS took about a tenth less time.
_PADDING_SHARE_BITS = 4


def transformer(width: int, *, layers: int, heads: int, dropout: float = DROPOUT) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        width,
        heads,
        dim_feedforward=FEEDFORWARD_EXPANSION * width,
        dropout=dropout,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)


class DropoutInEncodeRealError(ValueError):
    def __init__(self) -> None:
        super().__init__("encode_real runs an encoder without dropout: build it with the default DROPOUT")


def encode_real(encoder: nn.TransformerEncoder, sequence: Tensor, *, mask: Tensor) -> Tensor:
    """Encode each row's real tokens and return them at their original positions.

    `encoder` is one that `transformer` built, without dropout. `sequence`
    is float `[b, n, width]` and `mask` is bool `[b, n]`, true for a real
    token, with at least one in every row. The result is float
    `[b, n, width]`, what `encoder.forward(sequence, src_key_padding_mask=~mask)`
    gives at the real tokens and zero wherever `mask` is false. Most of an
    encoder's work is per token, so the real tokens of every row are packed
    into one `[tokens, width]` matrix, padded by a few dummy tokens to a
    rounder count, and the projections, feed-forward blocks and norms run on
    that alone. Only attention mixes the tokens of a row: on MPS it runs on
    the packed tokens as `jace.metal_attention` kernels, and elsewhere it
    reads each row padded to the longest row's length. The encoder has no
    position input, so the place of a token in its row changes nothing.
    Raises `DropoutInEncodeRealError` for an encoder with dropout in
    training mode.
    """
    if encoder.training and any(layer.dropout.p > 0 for layer in encoder.layers):
        raise DropoutInEncodeRealError
    batch, length, width = sequence.shape
    counts = mask.sum(dim=1)
    # One read from the device gives both sizes the packed layout needs.
    longest, total = torch.stack((counts.max(), counts.sum())).tolist()
    packing = _packing(mask, counts=counts, longest=longest, tokens=_rounded_up(total))
    # The dummy tokens read a zero row added after the last, and their outputs land there.
    with_zero_row = torch.cat((sequence.reshape(batch * length, width), sequence.new_zeros(1, width)))
    encoded = _encode_packed(encoder, with_zero_row[packing.sources], packing=packing)
    unpacked = sequence.new_zeros(batch * length + 1, width).index_put((packing.sources,), encoded)
    return unpacked[:-1].view(batch, length, width)


@dataclass(frozen=True, slots=True)
class _Packing:
    # sources: int64 [tokens], each packed token's place among the batch's flattened tokens, row after row,
    # and for a dummy token the place after the last. real: bool [tokens], false for a dummy token.
    sources: Tensor
    real: Tensor
    # rows: int64 [tokens], each token's row, the last row for a dummy. bounds: int32 [tokens, 2], the
    # packed index of the first token of each token's row and of the token after its last; a dummy token
    # makes a row of its own.
    rows: Tensor
    bounds: Tensor
    # filled: bool [b, longest], true for as many of a row's first places as it has real tokens.
    filled: Tensor


def _rounded_up(count: int) -> int:
    # The least multiple of a power of two that is at least `count` and at most 1 / 2 ** _PADDING_SHARE_BITS above it.
    step = 1 << max(count.bit_length() - 1 - _PADDING_SHARE_BITS, 0)
    return -(-count // step) * step


def _packing(mask: Tensor, *, counts: Tensor, longest: int, tokens: int) -> _Packing:
    batch, length = mask.shape
    sources = mask.flatten().nonzero_static(size=tokens, fill_value=batch * length).squeeze(1)
    real = sources < batch * length
    rows = (sources // length).clamp_max(batch - 1)
    ends = counts.cumsum(0)
    packed = torch.arange(tokens, device=mask.device)
    starts = torch.where(real, (ends - counts)[rows], packed)
    stops = torch.where(real, ends[rows], packed + 1)
    return _Packing(
        sources=sources,
        real=real,
        rows=rows,
        bounds=torch.stack((starts, stops), dim=1).int(),
        filled=torch.arange(longest, device=mask.device) < counts[:, None],
    )


def _encode_packed(encoder: nn.TransformerEncoder, packed: Tensor, *, packing: _Packing) -> Tensor:
    # packed: float [tokens, width]. Each step is the one in `nn.TransformerEncoder.forward` with pre-norm
    # layers and no dropout, which `transformer` builds.
    encoded = packed
    for layer in encoder.layers:
        encoded = encoded + _attention(layer.self_attn, layer.norm1(encoded), packing=packing)
        encoded = encoded + layer.linear2(layer.activation(layer.linear1(layer.norm2(encoded))))
    return encoded if encoder.norm is None else encoder.norm(encoded)


def _attention(attention: nn.MultiheadAttention, packed: Tensor, *, packing: _Packing) -> Tensor:
    # What `attention` gives each packed token when it attends to the tokens of its own row.
    tokens, width = packed.shape
    heads = attention.num_heads
    projected = functional.linear(packed, attention.in_proj_weight, attention.in_proj_bias)
    projected = projected.view(tokens, 3, heads, width // heads)  # each token's query, key and value heads
    if projected.device.type == "mps":
        mixed, _ = attend_within_rows(projected, packing.bounds)
    else:
        mixed = _padded_attention(projected, packing=packing)
    return attention.out_proj(mixed.reshape(tokens, width))


def _padded_attention(projected: Tensor, *, packing: _Packing) -> Tensor:
    # projected: float [tokens, 3, heads, depth]. Returns float [tokens, heads, depth]. Queries, keys and
    # values are each laid out as a contiguous [b, heads, longest, depth] block, a row's tokens in its first
    # places, because scaled_dot_product_attention is slower on permuted views of one tensor.
    tokens, _, heads, depth = projected.shape
    batch, longest = packing.filled.shape
    block = batch * heads * longest
    packed = torch.arange(tokens, device=projected.device)
    ranks = packed - packing.bounds[:, 0]
    # places: int64 [tokens, heads], where each of a token's heads goes in its block. A dummy token's go to a
    # spare place after the block, which attention never reads.
    places = ((packing.rows * heads)[:, None] + torch.arange(heads, device=projected.device)) * longest
    places = torch.where(packing.real[:, None], places + ranks[:, None], block)
    parts = torch.arange(3, device=projected.device)[None, :, None]
    laid_out = projected.new_zeros(3, block + 1, depth).index_put((parts, places[:, None]), projected)
    query, key, value = (part[:block].view(batch, heads, longest, depth) for part in laid_out)
    mixed = functional.scaled_dot_product_attention(query, key, value, attn_mask=packing.filled[:, None, None])
    return torch.cat((mixed.reshape(block, depth), mixed.new_zeros(1, depth)))[places]

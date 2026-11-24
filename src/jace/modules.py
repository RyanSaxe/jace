"""Helpers the models share: standard transformers, unpadded encoding, and device selection.

`transformer` builds the encoder that most of the models use. The deck
models' `CardSetEncoder` builds its own from the same constants, so the
encoder's settings are chosen in one place. `encode_real` runs an encoder
on the real tokens only, because the engine pads every board to a fixed size.
`default_device_name` and `device_named` resolve accelerator selection across
pretraining and self-play entry points.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

__all__ = (
    "DROPOUT",
    "FEEDFORWARD_EXPANSION",
    "DropoutInEncodeRealError",
    "default_device_name",
    "device_named",
    "encode_real",
    "transformer",
)

# A transformer layer's feed-forward block is this many times the model width,
# the ratio the original transformer used.
FEEDFORWARD_EXPANSION = 4
# Dropout is off unless a model is built with its own, so a model gives the same output whether or not it is
# in training mode. The drafter's pretraining on human picks builds it with dropout and measures it in eval mode.
DROPOUT = 0.0
# encode_real adds dummy tokens to round its count of packed tokens up to a multiple of a power of two. The rounded
# counts repeat from batch to batch, which matters on MPS, because MPS builds a graph for every new shape of an op such
# as a linear layer or a layer norm. With the counts rounded, a training block's update on MPS took about a tenth less
# time. The dummy tokens add at most 1 / 2 ** _PADDING_SHARE_BITS of the count.
_PADDING_SHARE_BITS = 4


def default_device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_named(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device(default_device_name())


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

    `encoder` must be one that `transformer` built, and its dropout must be
    off: either it was built without dropout, or it is in eval mode.
    `sequence` is float `[b, n, width]`, and `mask` is bool `[b, n]`, true
    for a real token. Every row of `mask` must have at least one real token.

    The result is float `[b, n, width]`. At the real tokens it is what
    `encoder.forward(sequence, src_key_padding_mask=~mask)` gives, and
    wherever `mask` is false it is zero.

    Most of an encoder's work is done one token at a time, so the real
    tokens of every row are packed into one `[tokens, width]` matrix, with a
    few dummy tokens to round the count up. The projections, feed-forward
    blocks and norms run on that matrix alone. Only attention mixes the
    tokens of a row, padding each row to the longest row's length for
    attention. Packing changes nothing in the result, because the encoder has
    no position input.

    Raises `DropoutInEncodeRealError` for an encoder with dropout that is in
    training mode.
    """
    if encoder.training and any(layer.dropout.p > 0 for layer in encoder.layers):
        raise DropoutInEncodeRealError
    batch, length, width = sequence.shape
    counts = mask.sum(dim=1)
    # One read from the device gives both sizes the packed layout needs.
    longest, total = torch.stack((counts.max(), counts.sum())).tolist()
    packing = _packing(mask, counts=counts, longest=longest, tokens=_rounded_up(total))
    # The dummy tokens take their input from a zero row appended after the batch's b * n rows. Their outputs
    # are written back to that row, which is then dropped.
    with_zero_row = torch.cat((sequence.reshape(batch * length, width), sequence.new_zeros(1, width)))
    encoded = _encode_packed(encoder, with_zero_row[packing.sources], packing=packing)
    unpacked = sequence.new_zeros(batch * length + 1, width).index_put((packing.sources,), encoded)
    return unpacked[:-1].view(batch, length, width)


@dataclass(frozen=True, slots=True)
class _Packing:
    # sources: int64 [tokens], where each packed token comes from among the batch's tokens, flattened row
    # after row into b * n rows. Every dummy token comes from row b * n, the zero row appended after them.
    sources: Tensor
    # real: bool [tokens], false for a dummy token.
    real: Tensor
    # rows: int64 [tokens], the row of each token, and the last row for a dummy token.
    rows: Tensor
    # bounds: int32 [tokens, 2]. For each token, the packed index of the first token of its row and of the
    # token just after the row's last. A dummy token forms a row of its own.
    bounds: Tensor
    # filled: bool [b, longest], true at a row's first k places when the row has k real tokens.
    filled: Tensor


def _rounded_up(count: int) -> int:
    # Rounds `count` up to a multiple of `step`, the largest power of two that is at most
    # count / 2 ** _PADDING_SHARE_BITS, or 1 for a small count. So rounding adds at most that many.
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
    # packed: float [tokens, width]. The loop computes what `nn.TransformerEncoder.forward` would for the
    # pre-norm layers that `transformer` builds, with dropout off, but on the packed tokens.
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
    # places: int64 [tokens, heads], where each of a token's heads goes in its block. A dummy token's heads go
    # to a spare place after the block, which attention never reads.
    places = ((packing.rows * heads)[:, None] + torch.arange(heads, device=projected.device)) * longest
    places = torch.where(packing.real[:, None], places + ranks[:, None], block)
    parts = torch.arange(3, device=projected.device)[None, :, None]
    laid_out = projected.new_zeros(3, block + 1, depth).index_put((parts, places[:, None]), projected)
    query, key, value = (part[:block].view(batch, heads, longest, depth) for part in laid_out)
    mixed = functional.scaled_dot_product_attention(query, key, value, attn_mask=packing.filled[:, None, None])
    return torch.cat((mixed.reshape(block, depth), mixed.new_zeros(1, depth)))[places]

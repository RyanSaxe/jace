"""Helpers every model shares: the standard transformer and encoding without padding.

`transformer` builds the one encoder shape the models use, so its settings
live in one place. `encode_real` runs an encoder on the real tokens only,
which matters because the engine pads every board to a fixed size and most
of its rows are empty.
"""

import torch
from torch import Tensor, nn
from torch.nn import functional

__all__ = ("DROPOUT", "FEEDFORWARD_EXPANSION", "encode_real", "transformer")

# A transformer layer's feed-forward block is this many times the model width,
# the ratio the original transformer used.
FEEDFORWARD_EXPANSION = 4
# Dropout is off unless a model is built with its own, so a model gives the same output whether or not it is
# in training mode. The drafter's pretraining on human picks builds it with dropout and measures it in eval mode.
DROPOUT = 0.0


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


def encode_real(encoder: nn.TransformerEncoder, sequence: Tensor, *, mask: Tensor) -> Tensor:
    """Encode each row's real tokens and return them at their original positions.

    `encoder` is one that `transformer` built. `sequence` is float
    `[b, n, width]` and `mask` is bool `[b, n]`, true for a real token, with
    at least one in every row. The result is float `[b, n, width]`, what
    `encoder.forward(sequence, src_key_padding_mask=~mask)` gives at the
    real tokens and zero wherever `mask` is false. Most of an encoder's
    work is per token, so the real tokens of every row are packed into one
    `[tokens, width]` matrix and the projections, feed-forward blocks and
    norms run on that alone. Only attention mixes the tokens of a row, and
    it reads each row padded to the longest row's length. The encoder has
    no position input, so the place of a token in its row changes nothing.
    """
    batch, length, width = sequence.shape
    counts = mask.sum(dim=1)
    # One read from the device gives both sizes the packed layout needs.
    longest, total = torch.stack((counts.max(), counts.sum())).tolist()
    # Each real token's place in `sequence`, row after row, its row, and its place among its row's real tokens.
    sources = mask.flatten().nonzero_static(size=total).squeeze(1)
    rows = sources // length
    ranks = mask.cumsum(dim=1).flatten()[sources] - 1
    filled = torch.arange(longest, device=mask.device) < counts[:, None]
    packed = sequence.reshape(batch * length, width)[sources]
    encoded = _encode_packed(encoder, packed, rows=rows, ranks=ranks, filled=filled)
    return sequence.new_zeros(batch * length, width).index_put((sources,), encoded).view(batch, length, width)


def _encode_packed(
    encoder: nn.TransformerEncoder, packed: Tensor, *, rows: Tensor, ranks: Tensor, filled: Tensor
) -> Tensor:
    # packed: float [tokens, width]. rows, ranks: int64 [tokens], each token's row and its place among
    # the row's real tokens. filled: bool [b, longest], true for a place that holds a token. Each step is
    # the one in `nn.TransformerEncoder.forward` with pre-norm layers, which `transformer` builds.
    encoded = packed
    for layer in encoder.layers:
        attended = _attention(layer.self_attn, layer.norm1(encoded), rows=rows, ranks=ranks, filled=filled)
        encoded = encoded + layer.dropout1(attended)
        expanded = layer.dropout(layer.activation(layer.linear1(layer.norm2(encoded))))
        encoded = encoded + layer.dropout2(layer.linear2(expanded))
    return encoded if encoder.norm is None else encoder.norm(encoded)


def _attention(
    attention: nn.MultiheadAttention, packed: Tensor, *, rows: Tensor, ranks: Tensor, filled: Tensor
) -> Tensor:
    # What `attention` gives each packed token when it attends to the tokens of its own row. Its queries,
    # keys and values are each laid out as a contiguous [b, heads, longest, depth] block, a row's tokens in
    # its first places, because scaled_dot_product_attention on MPS is slower on permuted views of one tensor.
    batch, longest = filled.shape
    tokens, width = packed.shape
    heads = attention.num_heads
    depth = width // heads
    block = batch * heads * longest
    # places: int64 [tokens, heads], where each of a token's heads goes in a block.
    places = ((rows * heads)[:, None] + torch.arange(heads, device=rows.device)) * longest + ranks[:, None]
    # The projection holds each token's query heads, then its key heads, then its value heads.
    blocks = torch.arange(3, device=rows.device)[:, None] * block
    projected = functional.linear(packed, attention.in_proj_weight, attention.in_proj_bias)
    laid_out = projected.new_zeros(3 * block, depth).index_copy(
        0, (places[:, None] + blocks).flatten(), projected.view(tokens * 3 * heads, depth)
    )
    query, key, value = laid_out.view(3, batch, heads, longest, depth)
    mixed = functional.scaled_dot_product_attention(
        query, key, value, attn_mask=filled[:, None, None], dropout_p=attention.dropout if attention.training else 0.0
    )  # [b, heads, longest, depth]
    return attention.out_proj(mixed.reshape(block, depth)[places.flatten()].view(tokens, width))

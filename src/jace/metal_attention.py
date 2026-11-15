"""Attention within the rows of packed tokens, as Metal kernels for Apple GPUs.

`jace.modules.encode_real` packs the real tokens of many rows into one
matrix, and each token attends only to the tokens of its own row. On MPS,
`scaled_dot_product_attention` would need those rows padded to one length,
and MPS builds a graph for every new length. `attend_within_rows` avoids
both by working on the packed tokens directly. One SIMD group of 32 lanes
computes the attention of one token and one head, with an online softmax
over the keys of that token's row. The backward pass recomputes the
attention weights from the log-sum-exp that the forward pass saved.
Nothing is padded and no shape is special, so every batch runs the same
three kernels.

The kernels run on MPS only. On every other device, `encode_real` uses
`scaled_dot_product_attention`.
"""

import functools
import math
from typing import Protocol

import torch
from torch import Tensor

__all__ = ("attend_within_rows",)

# A SIMD group on Apple GPUs has 32 lanes. Lane l works on dimensions l, l + 32, ... of a head. A dot product
# of two heads is then one simd_sum over the lanes' partial sums, and the group reads a key's or value's head
# together.
_SOURCE = """
#include <metal_stdlib>
using namespace metal;
constant constexpr int DEPTH = {depth};
constant constexpr int PER_LANE = (DEPTH + 31) / 32;
constant constexpr float SCALE = {scale};

inline float lane_value(device const float* head, uint lane, int r) {{
    const int d = lane + 32 * r;
    return d < DEPTH ? head[d] : 0.0f;
}}

inline void write_lanes(device float* head, uint lane, thread const float* values) {{
    for (int r = 0; r < PER_LANE; ++r) {{
        const int d = lane + 32 * r;
        if (d < DEPTH) head[d] = values[r];
    }}
}}

// projected: [tokens, 3, heads, DEPTH], each token's query, key and value heads.
// bounds: [tokens, 2], the first token of a token's row and the token after its last.
// mixed: [tokens, heads, DEPTH]. lse: [tokens, heads], the log-sum-exp of each query's scores.
kernel void attend_forward(
    device const float* projected [[buffer(0)]],
    device const int* bounds [[buffer(1)]],
    device float* mixed [[buffer(2)]],
    device float* lse [[buffer(3)]],
    device const int* sizes [[buffer(4)]],
    uint thread_id [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const int tokens = sizes[0], heads = sizes[1];
    const int pair = thread_id / 32, head = pair / tokens, i = pair % tokens;
    if (head >= heads) return;
    const int stride = 3 * heads * DEPTH;
    float query[PER_LANE], out[PER_LANE];
    for (int r = 0; r < PER_LANE; ++r) {{
        query[r] = lane_value(projected + i * stride + head * DEPTH, lane, r) * SCALE;
        out[r] = 0.0f;
    }}
    float top = -INFINITY, total = 0.0f;
    for (int j = bounds[2 * i]; j < bounds[2 * i + 1]; ++j) {{
        device const float* key = projected + j * stride + (heads + head) * DEPTH;
        device const float* value = key + heads * DEPTH;
        float score = 0.0f;
        for (int r = 0; r < PER_LANE; ++r) score += query[r] * lane_value(key, lane, r);
        score = simd_sum(score);
        const float new_top = max(top, score);
        const float shrink = exp(top - new_top), weight = exp(score - new_top);
        total = total * shrink + weight;
        for (int r = 0; r < PER_LANE; ++r) out[r] = out[r] * shrink + weight * lane_value(value, lane, r);
        top = new_top;
    }}
    for (int r = 0; r < PER_LANE; ++r) out[r] /= total;
    write_lanes(mixed + (i * heads + head) * DEPTH, lane, out);
    if (lane == 0) lse[i * heads + head] = top + log(total);
}}

// delta: [tokens, heads], each query's output head dotted with its gradient.
kernel void attend_backward_queries(
    device const float* projected [[buffer(0)]],
    device const int* bounds [[buffer(1)]],
    device const float* mixed_grad [[buffer(2)]],
    device const float* lse [[buffer(3)]],
    device const float* delta [[buffer(4)]],
    device float* query_grad [[buffer(5)]],
    device const int* sizes [[buffer(6)]],
    uint thread_id [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const int tokens = sizes[0], heads = sizes[1];
    const int pair = thread_id / 32, head = pair / tokens, i = pair % tokens;
    if (head >= heads) return;
    const int stride = 3 * heads * DEPTH;
    float query[PER_LANE], grad[PER_LANE], out[PER_LANE];
    for (int r = 0; r < PER_LANE; ++r) {{
        query[r] = lane_value(projected + i * stride + head * DEPTH, lane, r) * SCALE;
        grad[r] = lane_value(mixed_grad + (i * heads + head) * DEPTH, lane, r);
        out[r] = 0.0f;
    }}
    const float lse_i = lse[i * heads + head], delta_i = delta[i * heads + head];
    for (int j = bounds[2 * i]; j < bounds[2 * i + 1]; ++j) {{
        device const float* key = projected + j * stride + (heads + head) * DEPTH;
        device const float* value = key + heads * DEPTH;
        float score = 0.0f, weight_grad = 0.0f;
        for (int r = 0; r < PER_LANE; ++r) {{
            score += query[r] * lane_value(key, lane, r);
            weight_grad += grad[r] * lane_value(value, lane, r);
        }}
        const float score_grad = exp(simd_sum(score) - lse_i) * (simd_sum(weight_grad) - delta_i);
        for (int r = 0; r < PER_LANE; ++r) out[r] += score_grad * lane_value(key, lane, r) * SCALE;
    }}
    write_lanes(query_grad + (i * heads + head) * DEPTH, lane, out);
}}

kernel void attend_backward_keys_values(
    device const float* projected [[buffer(0)]],
    device const int* bounds [[buffer(1)]],
    device const float* mixed_grad [[buffer(2)]],
    device const float* lse [[buffer(3)]],
    device const float* delta [[buffer(4)]],
    device float* key_grad [[buffer(5)]],
    device float* value_grad [[buffer(6)]],
    device const int* sizes [[buffer(7)]],
    uint thread_id [[thread_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]])
{{
    const int tokens = sizes[0], heads = sizes[1];
    const int pair = thread_id / 32, head = pair / tokens, j = pair % tokens;
    if (head >= heads) return;
    const int stride = 3 * heads * DEPTH;
    device const float* own_key = projected + j * stride + (heads + head) * DEPTH;
    float key[PER_LANE], value[PER_LANE], key_out[PER_LANE], value_out[PER_LANE];
    for (int r = 0; r < PER_LANE; ++r) {{
        key[r] = lane_value(own_key, lane, r);
        value[r] = lane_value(own_key + heads * DEPTH, lane, r);
        key_out[r] = 0.0f;
        value_out[r] = 0.0f;
    }}
    // A key is read by the queries of its own row, which are the tokens its bounds name.
    for (int i = bounds[2 * j]; i < bounds[2 * j + 1]; ++i) {{
        device const float* query_head = projected + i * stride + head * DEPTH;
        device const float* grad_head = mixed_grad + (i * heads + head) * DEPTH;
        float score = 0.0f, weight_grad = 0.0f;
        for (int r = 0; r < PER_LANE; ++r) {{
            score += lane_value(query_head, lane, r) * key[r];
            weight_grad += lane_value(grad_head, lane, r) * value[r];
        }}
        const float weight = exp(simd_sum(score) * SCALE - lse[i * heads + head]);
        const float score_grad = weight * (simd_sum(weight_grad) - delta[i * heads + head]);
        for (int r = 0; r < PER_LANE; ++r) {{
            value_out[r] += weight * lane_value(grad_head, lane, r);
            key_out[r] += score_grad * lane_value(query_head, lane, r) * SCALE;
        }}
    }}
    write_lanes(key_grad + (j * heads + head) * DEPTH, lane, key_out);
    write_lanes(value_grad + (j * heads + head) * DEPTH, lane, value_out);
}}
"""

# _LANES is the number of lanes in a SIMD group. _THREADGROUP, the number of threads in a threadgroup, is a
# multiple of _LANES, so no SIMD group is split between two threadgroups.
_LANES = 32
_THREADGROUP = 256


class _Kernels(Protocol):
    # The kernels of `_SOURCE`, as the library from torch.mps.compile_shader exposes them. That library is
    # untyped, so this protocol gives the type checker their signatures. Each kernel takes its buffers in
    # order, then the number of threads and the threadgroup size.
    def attend_forward(self, *buffers: Tensor, threads: int, group_size: int) -> None: ...
    def attend_backward_queries(self, *buffers: Tensor, threads: int, group_size: int) -> None: ...
    def attend_backward_keys_values(self, *buffers: Tensor, threads: int, group_size: int) -> None: ...


class _Saved(Protocol):
    # The part of torch.library's context object that the backward pass uses.
    saved_tensors: tuple[Tensor, ...]

    def save_for_backward(self, *tensors: Tensor) -> None: ...


@functools.cache
def _kernels(depth: int) -> _Kernels:
    # The kernels for heads `depth` wide, compiled once.
    scale = f"{1.0 / math.sqrt(depth)!r}f"
    return torch.mps.compile_shader(_SOURCE.format(depth=depth, scale=scale))


def _launch(projected: Tensor) -> tuple[Tensor, int]:
    # The sizes the kernels read, and the threads to run them on: one SIMD group for each token and head.
    tokens, _, heads, _ = projected.shape
    sizes = torch.tensor([tokens, heads], dtype=torch.int32, device=projected.device)
    return sizes, tokens * heads * _LANES


@torch.library.custom_op("jace::attend_within_rows", mutates_args=())
def attend_within_rows(projected: Tensor, bounds: Tensor) -> tuple[Tensor, Tensor]:
    """Return each packed token's attention over the tokens of its own row, and the log-sum-exp of its scores.

    `projected` is float `[tokens, 3, heads, depth]` on MPS: each token's
    query, key and value heads, split the way `nn.MultiheadAttention`
    splits its in-projection. `bounds` is int32 `[tokens, 2]`: for each
    token, the packed index of the first token of its row and of the token
    just after the row's last. A row's tokens must therefore be consecutive.

    Returns the attention, float `[tokens, heads, depth]`, and the
    log-sum-exp of each query's scaled scores, float `[tokens, heads]`. The
    backward pass reads the log-sum-exp to recompute the attention weights,
    and no gradient flows through it.
    """
    tokens, _, heads, depth = projected.shape
    mixed = projected.new_empty(tokens, heads, depth)
    lse = projected.new_empty(tokens, heads)
    sizes, threads = _launch(projected)
    _kernels(depth).attend_forward(
        projected.contiguous(), bounds.contiguous(), mixed, lse, sizes, threads=threads, group_size=_THREADGROUP
    )
    return mixed, lse


@attend_within_rows.register_fake
def _shapes(projected: Tensor, _bounds: Tensor) -> tuple[Tensor, Tensor]:
    tokens, _, heads, depth = projected.shape
    return projected.new_empty(tokens, heads, depth), projected.new_empty(tokens, heads)


def _keep_for_backward(ctx: _Saved, *, inputs: tuple[Tensor, Tensor], output: tuple[Tensor, Tensor]) -> None:
    # torch.library calls this with keywords. The backward pass needs the inputs and both outputs.
    projected, bounds = inputs
    mixed, lse = output
    ctx.save_for_backward(projected, bounds, mixed, lse)


def _backward(ctx: _Saved, mixed_grad: Tensor, *_lse_grad: Tensor) -> tuple[Tensor, None]:
    # torch.library passes a gradient for each output; the log-sum-exp is not differentiated.
    projected, bounds, mixed, lse = ctx.saved_tensors
    mixed_grad = mixed_grad.contiguous()
    delta = (mixed_grad * mixed).sum(dim=-1)  # [tokens, heads]
    query_grad, key_grad, value_grad = (torch.empty_like(mixed) for _ in range(3))
    sizes, threads = _launch(projected)
    kernels = _kernels(projected.shape[-1])
    shared = (projected, bounds, mixed_grad, lse, delta)
    kernels.attend_backward_queries(*shared, query_grad, sizes, threads=threads, group_size=_THREADGROUP)
    kernels.attend_backward_keys_values(*shared, key_grad, value_grad, sizes, threads=threads, group_size=_THREADGROUP)
    return torch.stack((query_grad, key_grad, value_grad), dim=1), None


attend_within_rows.register_autograd(_backward, setup_context=_keep_for_backward)

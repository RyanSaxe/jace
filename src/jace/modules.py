"""Helpers every model shares: a checked module call and the standard transformer.

PyTorch types a module call as returning anything, which the strict type
checker rejects wherever a tensor is expected. `run` makes the call and
verifies the result at run time, so the rest of the package can rely on
tensor types without casts. It goes through the module's normal call, so
hooks still fire. `transformer` builds the one encoder shape the models use,
so its settings live in one place. It raises `TypeError` when a module returns anything but
one tensor; a module with several outputs exposes its own typed method.
"""

from torch import Tensor, nn

__all__ = ("DROPOUT", "FEEDFORWARD_EXPANSION", "run", "transformer")

# A transformer layer's feed-forward block is this many times the model width,
# the ratio the original transformer used.
FEEDFORWARD_EXPANSION = 4
# Dropout is off, so a model gives the same output whether or not it is in training mode.
DROPOUT = 0.0


def run(module: nn.Module, *inputs: Tensor) -> Tensor:
    result = module(*inputs)
    if not isinstance(result, Tensor):
        message = f"{type(module).__name__} returned {type(result).__name__}, expected a tensor"
        raise TypeError(message)
    return result


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

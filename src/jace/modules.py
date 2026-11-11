"""Call PyTorch modules with a checked tensor result.

PyTorch types a module call as returning anything, which the strict type
checker rejects wherever a tensor is expected. `run` makes the call and
verifies the result at run time, so the rest of the package can rely on
tensor types without casts. It goes through the module's normal call, so
hooks still fire. It raises `TypeError` when a module returns anything but
one tensor; a module with several outputs exposes its own typed method.
"""

from torch import Tensor, nn

__all__ = ("run",)


def run(module: nn.Module, *inputs: Tensor) -> Tensor:
    result = module(*inputs)
    if not isinstance(result, Tensor):
        message = f"{type(module).__name__} returned {type(result).__name__}, expected a tensor"
        raise TypeError(message)
    return result

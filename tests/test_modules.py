"""Check that encoding only the real tokens computes what the padded encoder computes.

`encode_real` packs the real tokens of every row and runs the encoder's layers
on them by hand, not through `nn.TransformerEncoder.forward`. These tests are
the only check that the two compute the same. The rows have every length from
one token to the full row, and each row's real tokens are at random positions.
The first test runs in float64, so any difference larger than rounding fails
it. On MPS, attention runs as Metal kernels that work in float32, so the MPS
test allows float32's rounding.
"""

import pytest
import torch
from torch import Tensor, nn

from jace.modules import encode_real, transformer

__all__ = ()

ROWS = 12
LENGTH = 12
WIDTH = 16


def _rows() -> tuple[nn.TransformerEncoder, Tensor, Tensor, Tensor]:
    # Returns a float64 encoder, its input, the mask of real tokens and the weights of a loss on the output.
    generator = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    encoder = transformer(WIDTH, layers=2, heads=4).double()
    sequence = torch.randn(ROWS, LENGTH, WIDTH, dtype=torch.float64, generator=generator, requires_grad=True)
    counts = torch.arange(1, ROWS + 1)
    mask = torch.rand(ROWS, LENGTH, generator=generator).argsort(dim=1) < counts[:, None]
    weights = torch.randn(ROWS, LENGTH, WIDTH, dtype=torch.float64, generator=generator)
    return encoder, sequence, mask, weights


def test_encoding_the_real_tokens_matches_the_padded_encoder_and_its_gradients() -> None:
    encoder, sequence, mask, weights = _rows()
    inputs = (sequence, *encoder.parameters())

    padded = encoder.forward(sequence, src_key_padding_mask=~mask) * mask[..., None]
    padded_gradients = torch.autograd.grad((padded * weights).sum(), inputs)
    real = encode_real(encoder, sequence, mask=mask)
    real_gradients = torch.autograd.grad((real * weights).sum(), inputs)

    torch.testing.assert_close(real, padded, rtol=0, atol=1e-12)
    assert bool((real[~mask] == 0).all())
    for real_gradient, padded_gradient in zip(real_gradients, padded_gradients, strict=True):
        torch.testing.assert_close(real_gradient, padded_gradient, rtol=0, atol=1e-10)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="the Metal kernels run on MPS only")
def test_encoding_on_mps_with_the_metal_kernels_matches_the_padded_encoder() -> None:
    encoder, sequence, mask, weights = _rows()
    padded = encoder.forward(sequence, src_key_padding_mask=~mask) * mask[..., None]
    padded_gradients = torch.autograd.grad((padded * weights).sum(), (sequence, *encoder.parameters()))

    device = torch.device("mps")
    on_mps = encoder.float().to(device)
    sequence_on_mps = sequence.detach().float().to(device).requires_grad_()
    real = encode_real(on_mps, sequence_on_mps, mask=mask.to(device))
    loss = (real * weights.float().to(device)).sum()
    real_gradients = torch.autograd.grad(loss, (sequence_on_mps, *on_mps.parameters()))

    torch.testing.assert_close(real.cpu().double(), padded, rtol=0, atol=1e-4)
    for real_gradient, padded_gradient in zip(real_gradients, padded_gradients, strict=True):
        torch.testing.assert_close(real_gradient.cpu().double(), padded_gradient, rtol=1e-3, atol=1e-4)

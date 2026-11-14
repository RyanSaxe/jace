"""Check that encoding only the real tokens computes what the padded encoder computes.

`encode_real` packs the real tokens of every row and runs the encoder's
layers on them by hand, so nothing but this check ties it to
`nn.TransformerEncoder.forward`. The rows have every length from one token
to the full row, with their real tokens at scattered positions, and the
check runs in float64 so that any difference beyond rounding shows.
"""

import torch

from jace.modules import encode_real, transformer

__all__ = ()

ROWS = 12
LENGTH = 12
WIDTH = 16


def test_encoding_the_real_tokens_matches_the_padded_encoder_and_its_gradients() -> None:
    generator = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    encoder = transformer(WIDTH, layers=2, heads=4).double()
    sequence = torch.randn(ROWS, LENGTH, WIDTH, dtype=torch.float64, generator=generator, requires_grad=True)
    counts = torch.arange(1, ROWS + 1)
    mask = torch.rand(ROWS, LENGTH, generator=generator).argsort(dim=1) < counts[:, None]
    weights = torch.randn(ROWS, LENGTH, WIDTH, dtype=torch.float64, generator=generator)
    inputs = (sequence, *encoder.parameters())

    padded = encoder.forward(sequence, src_key_padding_mask=~mask) * mask[..., None]
    padded_gradients = torch.autograd.grad((padded * weights).sum(), inputs)
    real = encode_real(encoder, sequence, mask=mask)
    real_gradients = torch.autograd.grad((real * weights).sum(), inputs)

    torch.testing.assert_close(real, padded, rtol=0, atol=1e-12)
    assert bool((real[~mask] == 0).all())
    for real_gradient, padded_gradient in zip(real_gradients, padded_gradients, strict=True):
        torch.testing.assert_close(real_gradient, padded_gradient, rtol=0, atol=1e-10)

"""Run a Hugging Face text-embedding model as the encoder `text_vectors` takes.

This is the only module that imports `transformers`. Loading that package
takes about half a second, and only a run that embeds card text uses it, so
`jace.data` does not import this module. Import `hugging_face_encoder` from
`jace.data.hugging_face` by its full name.
"""

from collections.abc import Sequence

import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from jace.data.text import TextEncoder

__all__ = ("hugging_face_encoder",)

_TEXTS_PER_BATCH = 64


def hugging_face_encoder(model_name: str, *, device: torch.device) -> TextEncoder:
    """Return an encoder that runs the Hugging Face model `model_name` on `device`.

    The encoder loads the model and its tokenizer each time it is called,
    and not before, so a run that finds its vectors in the cache never loads
    them. If this machine has no copy, `transformers` downloads them first.

    A text's vector is the mean of the model's last hidden state over the
    text's tokens. A text longer than the model's limit is cut off at the
    limit. The encoder raises `ValueError` when Hugging Face has no
    tokenizer for `model_name`.
    """

    def encode(texts: Sequence[str]) -> Tensor:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer is None:
            message = f"Hugging Face has no tokenizer for {model_name}"
            raise ValueError(message)
        model = AutoModel.from_pretrained(model_name).to(device).eval()
        rows: list[Tensor] = []
        with torch.no_grad():
            for start in range(0, len(texts), _TEXTS_PER_BATCH):
                batch = tokenizer(
                    list(texts[start : start + _TEXTS_PER_BATCH]), padding=True, truncation=True, return_tensors="pt"
                ).to(device)
                hidden = model(**batch).last_hidden_state  # [b, tokens, width]
                real = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [b, tokens, 1]
                rows.append(((hidden * real).sum(dim=1) / real.sum(dim=1)).cpu())
        return torch.cat(rows)

    return encode

"""Make each card's starting vector from its printed text with a frozen language model.

A card from a new set has no play statistics, but its mana cost, type line,
power, toughness, rarity and rules text are printed on it. `card_texts`
writes those as one string per card and leaves the card's name out, so two
cards that do the same thing get nearby vectors whatever they are called.
`text_vectors` passes the strings to a text-embedding model once and caches
the result, because the model is frozen and its output never changes. The
columns are standardised over the cards passed in, so vectors made for two
different card lists are not on the same scale.
"""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl
import torch
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

__all__ = ("TEXT_MODEL", "TextEncoder", "card_texts", "hugging_face_encoder", "text_vectors")

# texts -> float [texts, width], one row per text in the order given.
type TextEncoder = Callable[[Sequence[str]], Tensor]

# A 22.7 million parameter sentence-embedding model under the Apache 2.0 licence. It gives 384
# numbers per text and is trained to be pooled by the mean over tokens, as `hugging_face_encoder` does.
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_OWN_NAME = "this card"
_TEXTS_PER_BATCH = 64
# A column whose values are all equal has a spread of zero. Dividing by this instead leaves it at zero.
_SMALLEST_SPREAD = 1e-6
_KEY_LENGTH = 16


def card_texts(attributes: pl.DataFrame) -> tuple[str, ...]:
    """Write each card as the text printed on it, without its name, in row order.

    `attributes` has the columns `name`, `mana_cost`, `card_type`,
    `subtype`, `power`, `toughness`, `rarity` and `oracle_text`. A line is
    left out when the card has nothing printed there, such as a land's mana
    cost. Inside the rules text, the card's full name and the part of it
    before a comma ("Aang" for "Aang, the Last Airbender") become "this
    card". Rarity is included because it is printed on the card and is
    known for a set nobody has played.
    """
    columns = ("name", "mana_cost", "card_type", "subtype", "power", "toughness", "rarity", "oracle_text")
    texts: list[str] = []
    for name, mana_cost, card_type, subtype, power, toughness, rarity, rules in attributes.select(columns).iter_rows():
        lines = [
            f"Mana cost: {mana_cost}" if mana_cost else "",
            f"Type: {card_type} - {subtype}" if subtype else f"Type: {card_type}",
            f"Power/toughness: {power}/{toughness}" if power is not None and toughness is not None else "",
            f"Rarity: {rarity}",
            (rules or "").replace(name, _OWN_NAME).replace(name.split(",")[0], _OWN_NAME),
        ]
        texts.append("\n".join(line for line in lines if line))
    return tuple(texts)


def text_vectors(texts: Sequence[str], *, encode: TextEncoder, model_name: str, cache_dir: Path) -> Tensor:
    """Embed `texts` and return float32 `[texts, width]` with mean zero and spread one in every column.

    The result is saved in `cache_dir` in a file named by a hash of
    `model_name` and the texts, and `encode` runs only when that file is
    missing. Changing a text, their order or the model name therefore makes
    a new file, and nothing checks that `encode` really is `model_name`.
    The mean and spread are taken over these texts.
    """
    key = hashlib.sha256("\x00".join([model_name, *texts]).encode()).hexdigest()[:_KEY_LENGTH]
    path = cache_dir / f"text_vectors_{key}.pt"
    if not path.exists():
        raw = encode(texts).detach().float().cpu()  # [texts, width]
        spread = raw.std(dim=0, correction=0).clamp_min(_SMALLEST_SPREAD)
        path.parent.mkdir(parents=True, exist_ok=True)
        unfinished = path.with_suffix(".partial")
        torch.save((raw - raw.mean(dim=0)) / spread, unfinished)
        unfinished.rename(path)
    vectors = torch.load(path, weights_only=True)
    if not isinstance(vectors, Tensor):
        message = f"{path} does not hold one tensor; delete it to rebuild"
        raise TypeError(message)
    return vectors


def hugging_face_encoder(model_name: str, *, device: torch.device) -> TextEncoder:
    """Return an encoder that runs the Hugging Face model `model_name` on `device`.

    The model and its tokenizer are loaded, and downloaded if this machine
    has no copy, when the encoder is first called and not before, so a run
    that finds its vectors in the cache never loads them. A text's vector is
    the mean of the model's last hidden state over the text's tokens. A text
    longer than the model's limit is cut off at the limit.
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

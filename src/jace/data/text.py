"""Make a vector of each card's printed text with a frozen language model, for the card encoder.

A card from a new set has no play statistics, but its mana cost, type line,
power, toughness, rarity and rules text are printed on it. `card_texts`
writes those as one string per card and leaves the card's name out, so two
cards that do the same thing get nearby vectors whatever they are called.

`text_vectors` passes the strings to a text-embedding model once and caches
the result, because the model is frozen and its output never changes. It
standardises each column of the vectors over the cards passed in, so
vectors made for two different card lists are not on the same scale.
"""

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import polars as pl
import torch
from torch import Tensor

from jace.data.store import SMALLEST_SPREAD

__all__ = ("TEXT_MODEL", "TextEncoder", "card_texts", "text_vectors")

# A text encoder takes texts and returns float [texts, width], one row per text in the order given.
type TextEncoder = Callable[[Sequence[str]], Tensor]

# A 22.7 million parameter sentence-embedding model under the Apache 2.0 licence. It gives 384
# numbers per text and is trained to be pooled by the mean over tokens, as
# `jace.data.hugging_face.hugging_face_encoder` does.
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_OWN_NAME = "this card"
_KEY_LENGTH = 16


def card_texts(attributes: pl.DataFrame) -> tuple[str, ...]:
    """Write each card as the text printed on it, without its name, in row order.

    `attributes` must have the columns `name`, `mana_cost`, `card_type`,
    `subtype`, `power`, `toughness`, `rarity` and `oracle_text`. A line is
    left out when the card has nothing printed there, such as a land's mana
    cost. Inside the rules text, the card's full name becomes "this card",
    and so does the part of the name before a comma, such as "Aang" for
    "Aang, the Last Airbender". Rarity is included because it is printed on
    the card and is known for a set nobody has played.
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

    The result is saved in `cache_dir`, in a file named by a hash of
    `model_name` and the texts, and `encode` runs only when that file is
    missing. Changing a text, the order of the texts or the model name
    therefore makes a new file. Nothing checks that `encode` really runs
    `model_name`, so the vectors of an encoder passed with the wrong name
    are saved under that name. The mean and spread are taken over these
    texts.

    Raises `TypeError` when the cached file contains something other than a
    tensor.
    """
    key = hashlib.sha256("\x00".join([model_name, *texts]).encode()).hexdigest()[:_KEY_LENGTH]
    path = cache_dir / f"text_vectors_{key}.pt"
    if not path.exists():
        raw = encode(texts).detach().float().cpu()  # [texts, width]
        spread = raw.std(dim=0, correction=0).clamp_min(SMALLEST_SPREAD)
        path.parent.mkdir(parents=True, exist_ok=True)
        unfinished = path.with_suffix(".partial")
        torch.save((raw - raw.mean(dim=0)) / spread, unfinished)
        unfinished.rename(path)
    vectors = torch.load(path, weights_only=True)
    if not isinstance(vectors, Tensor):
        message = f"{path} does not hold one tensor; delete it to rebuild"
        raise TypeError(message)
    return vectors

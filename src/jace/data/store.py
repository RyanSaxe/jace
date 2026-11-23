"""Helpers the modules of `jace.data` share: the on-disk cache, the card columns, and the split.

The 17Lands files have one column per card, so reading them takes seconds
and gigabytes of memory. Each loader therefore reads the files once and
saves plain tensors, and later runs load the tensors. A cache file also
contains the names of the cards it was built for. Loading it for a pool
with other names raises `StaleCacheError`, because its card ids would refer
to other cards.

A draft's split key depends only on its id. A draft is therefore on the
same side of the held-out split in the pick data, in the deck data and in
every run.
"""

import zlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch
from torch import Tensor

from jace.cards import CardPool

__all__ = (
    "SPLIT_KEYS",
    "CardColumnsError",
    "StaleCacheError",
    "cached",
    "card_columns",
    "split_keys",
)

# A draft's split key is in [0, SPLIT_KEYS). A draft is held out when its key is below
# `round(held_out_share * SPLIT_KEYS)`.
SPLIT_KEYS = 1000

type Tensors = Mapping[str, Tensor]


class StaleCacheError(ValueError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"{path} was built for a different card pool; delete it to rebuild")


class CardColumnsError(ValueError):
    def __init__(self, prefix: str, *, absent: Sequence[str], unknown: Sequence[str]) -> None:
        super().__init__(
            f"the {prefix}* columns and the card pool differ: "
            f"pool cards with no column {list(absent)}, columns with no pool card {list(unknown)}"
        )


def cached(path: Path, build: Callable[[], Tensors], *, pool: CardPool) -> Tensors:
    """Load the tensors saved at `path`, or call `build` and save what it returns.

    The file contains the tensors and the names of the pool's cards in id
    order. `build` runs only when the file is missing.

    Raises `StaleCacheError` when the file was saved for a pool with other
    names, because its card ids would refer to the wrong cards.
    """
    names = [pool[card].name for card in range(len(pool))]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # The tensors are saved to a .partial file, which is renamed once the save is complete. An interrupted save
        # therefore leaves nothing at `path`, and the next call builds again.
        unfinished = path.with_suffix(".partial")
        torch.save({"names": names, "tensors": dict(build())}, unfinished)
        unfinished.rename(path)
    saved = torch.load(path, weights_only=True)
    if saved["names"] != names:
        raise StaleCacheError(path)
    return {name: tensor for name, tensor in saved["tensors"].items() if isinstance(tensor, Tensor)}


def card_columns(columns: Sequence[str], prefix: str, *, pool: CardPool) -> list[str]:
    # Returns the `prefix` column of every pool card, in card id order. Raises `CardColumnsError`
    # unless the columns that start with `prefix` name exactly the pool's cards.
    wanted = [prefix + pool[card].name for card in range(len(pool))]
    found = {column for column in columns if column.startswith(prefix)}
    if found != set(wanted):
        absent = sorted(column.removeprefix(prefix) for column in set(wanted) - found)
        unknown = sorted(column.removeprefix(prefix) for column in found - set(wanted))
        raise CardColumnsError(prefix, absent=absent, unknown=unknown)
    return wanted


def split_keys(draft_ids: Sequence[str]) -> Tensor:
    # Returns int16 [drafts], each draft's key in [0, SPLIT_KEYS). The key comes from a CRC-32 checksum of the id,
    # which is the same in every process. Python's `hash` of a string changes from one process to the next.
    keys = [zlib.crc32(draft_id.encode()) % SPLIT_KEYS for draft_id in draft_ids]
    return torch.tensor(keys, dtype=torch.int16)

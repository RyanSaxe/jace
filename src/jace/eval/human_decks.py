"""Human 17Lands Limited deck loading, card-pool filtering, and eval deal builders.

Reads the cached `decks.pt` tensor archive produced by `jace.data.decks` so self-play
workers and evaluation scripts never re-parse raw 17Lands Parquet files. Applies the
exact deterministic CRC-32 held-out split from `jace.data.store`, and samples compact
`UniformDecks` shards so spawned workers validate hundreds of decks rather than all
91,359 on startup.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import managym
import numpy as np
import torch
from manabot.training.decks import UniformDecks
from torch import Tensor

from jace.data.store import SPLIT_KEYS
from jace.eval.deals import Deal

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from managym.disclosure import DecklistDisclosure

__all__ = (
    "HumanDeckCatalog",
    "build_human_deals",
    "build_human_deck_source",
    "load_human_deck_catalog",
    "row_to_deck",
    "sample_human_decks",
    "top_card_pool",
)

_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})


@dataclass(frozen=True, slots=True)
class HumanDeckCatalog:
    card_names: tuple[str, ...]
    train_decks: Tensor
    train_sideboards: Tensor
    held_out_decks: Tensor
    held_out_sideboards: Tensor


def _require_tensor(tensors: Mapping[str, object], key: str, *, path: Path) -> Tensor:
    value = tensors.get(key)
    if not isinstance(value, Tensor):
        msg = f"expected tensor {key!r} in {path}"
        raise TypeError(msg)
    return value


def load_human_deck_catalog(path: Path, *, held_out_share: float = 0.1) -> HumanDeckCatalog:
    """Load `decks.pt` and split rows into training and held-out uint8 deck matrices.

    Uses the exact same `split_key < round(held_out_share * SPLIT_KEYS)` partition as
    `jace.data.decks.load_decks` so held-out decks stay identical across pretraining,
    self-play, and evaluation. Raises `FileNotFoundError` when `path` is missing and
    `ValueError` when required archive keys or `held_out_share` are invalid.
    """
    if not 0.0 < held_out_share < 1.0:
        msg = f"held_out_share must be in (0, 1), got {held_out_share}"
        raise ValueError(msg)
    if not path.exists():
        msg = f"human deck cache not found: {path}"
        raise FileNotFoundError(msg)
    raw = torch.load(path, weights_only=True)
    if not isinstance(raw, Mapping) or "names" not in raw or "tensors" not in raw:
        msg = f"unexpected deck cache layout in {path}"
        raise ValueError(msg)
    tensors = raw["tensors"]
    if not isinstance(tensors, Mapping):
        msg = f"unexpected tensors mapping in {path}"
        raise TypeError(msg)
    names = tuple(str(name) for name in raw["names"])
    decks = _require_tensor(tensors, "decks", path=path).to(torch.uint8)
    pools = _require_tensor(tensors, "pools", path=path).to(torch.uint8)
    split_key = _require_tensor(tensors, "split_key", path=path)
    sideboards = torch.clamp(pools.to(torch.int16) - decks.to(torch.int16), min=0).to(torch.uint8)
    held_mask = split_key < round(held_out_share * SPLIT_KEYS)
    train_mask = ~held_mask
    if int(train_mask.sum().item()) == 0 or int(held_mask.sum().item()) == 0:
        msg = f"deck split produced an empty partition for held_out_share={held_out_share}"
        raise ValueError(msg)
    return HumanDeckCatalog(
        card_names=names,
        train_decks=decks[train_mask].contiguous(),
        train_sideboards=sideboards[train_mask].contiguous(),
        held_out_decks=decks[held_mask].contiguous(),
        held_out_sideboards=sideboards[held_mask].contiguous(),
    )


def row_to_deck(main_row: Tensor, sideboard_row: Tensor, *, card_names: Sequence[str]) -> managym.Deck:
    main_idx = torch.nonzero(main_row, as_tuple=False).flatten().tolist()
    sb_idx = torch.nonzero(sideboard_row, as_tuple=False).flatten().tolist()
    main = {card_names[i]: int(main_row[i].item()) for i in main_idx}
    sideboard = {card_names[i]: int(sideboard_row[i].item()) for i in sb_idx}
    return managym.Deck(main, sideboard)


def top_card_pool(catalog: HumanDeckCatalog, *, card_count: int) -> frozenset[str]:
    """Select `card_count` cards maximizing the number of fitting human training decks.

    Keeps every basic land present in `catalog` and greedily peels non-basic cards one
    at a time, removing whichever card drops the fewest surviving training decks (with
    total deck frequency breaking ties). This preserves thousands of complete human decks
    at small pool sizes (e.g. ~4,081 decks at 90 cards) instead of scattering across rares.
    """
    if card_count < 1:
        msg = f"card_count must be >= 1, got {card_count}"
        raise ValueError(msg)
    if card_count >= len(catalog.card_names):
        return frozenset(catalog.card_names)
    basics = [idx for idx, name in enumerate(catalog.card_names) if name in _BASIC_LANDS]
    non_basics = [idx for idx, name in enumerate(catalog.card_names) if name not in _BASIC_LANDS]
    keep_non_basics = max(0, card_count - len(basics))
    if keep_non_basics == 0:
        return frozenset(catalog.card_names[idx] for idx in basics[:card_count])
    present = (catalog.train_decks[:, non_basics] > 0).float()
    usage = present.sum(dim=0)
    tie_break = 1e-6 * usage / torch.clamp(usage.max(), min=1.0)
    in_pool = torch.ones(len(non_basics), dtype=torch.bool)
    alive = torch.ones(present.shape[0], dtype=torch.float32)
    for _ in range(len(non_basics) - keep_non_basics):
        lost = torch.where(in_pool, present.T @ alive + tie_break, float("inf"))
        drop = int(lost.argmin().item())
        in_pool[drop] = False
        alive = alive * (1.0 - present[:, drop])
    kept = [non_basics[pos] for pos, active in enumerate(in_pool.tolist()) if active]
    return frozenset(catalog.card_names[idx] for idx in (*basics, *kept))


def _eligible_indices(
    decks: Tensor,
    *,
    card_names: Sequence[str],
    allowed_cards: frozenset[str] | None,
) -> np.ndarray:
    total = int(decks.shape[0])
    if allowed_cards is None or len(allowed_cards) >= len(card_names):
        return np.arange(total, dtype=np.int64)
    disallowed = [idx for idx, name in enumerate(card_names) if name not in allowed_cards]
    if not disallowed:
        return np.arange(total, dtype=np.int64)
    banned_counts = decks[:, disallowed].to(torch.int32).sum(dim=1)
    eligible = torch.nonzero(banned_counts == 0, as_tuple=False).flatten().numpy().astype(np.int64)
    if eligible.size == 0:
        msg = f"no decks matched allowed_cards subset of size {len(allowed_cards)}"
        raise ValueError(msg)
    return eligible


def sample_human_decks(
    catalog: HumanDeckCatalog,
    *,
    split: Literal["train", "held_out"] = "train",
    count: int,
    seed: int,
    allowed_cards: frozenset[str] | None = None,
) -> list[tuple[str, managym.Deck]]:
    if count < 1:
        msg = f"count must be >= 1, got {count}"
        raise ValueError(msg)
    decks_t = catalog.train_decks if split == "train" else catalog.held_out_decks
    sbs_t = catalog.train_sideboards if split == "train" else catalog.held_out_sideboards
    eligible = _eligible_indices(decks_t, card_names=catalog.card_names, allowed_cards=allowed_cards)
    rng = np.random.default_rng(seed)
    replace = count > eligible.size
    chosen = rng.choice(eligible, size=count, replace=replace)
    return [
        (
            f"tla-{split}-{int(row_idx)}-{pos}",
            row_to_deck(decks_t[int(row_idx)], sbs_t[int(row_idx)], card_names=catalog.card_names),
        )
        for pos, row_idx in enumerate(chosen.tolist())
    ]


def build_human_deals(
    catalog: HumanDeckCatalog,
    *,
    split: Literal["train", "held_out"] = "held_out",
    count: int,
    seed: int = 20_000,
    decklists: DecklistDisclosure = "closed",
    allowed_cards: frozenset[str] | None = None,
) -> list[Deal]:
    sampled = sample_human_decks(
        catalog,
        split=split,
        count=count * 2,
        seed=seed,
        allowed_cards=allowed_cards,
    )
    deals: list[Deal] = []
    for index in range(count):
        name_a, deck_a = sampled[2 * index]
        name_b, deck_b = sampled[2 * index + 1]
        deals.append(
            Deal(
                seed=seed + index * 1009,
                deck_a_name=name_a,
                deck_b_name=name_b,
                deck_a=dict(deck_a.main),
                deck_b=dict(deck_b.main),
                sideboard_a=dict(deck_a.sideboard),
                sideboard_b=dict(deck_b.sideboard),
                decklists=decklists,
            )
        )
    return deals


def build_human_deck_source(
    catalog: HumanDeckCatalog,
    *,
    count: int,
    seed: int,
    allowed_cards: frozenset[str] | None = None,
    decklists: DecklistDisclosure = "closed",
) -> UniformDecks:
    sampled = sample_human_decks(
        catalog,
        split="train",
        count=count,
        seed=seed,
        allowed_cards=allowed_cards,
    )
    return UniformDecks(
        decks=tuple(deck for _, deck in sampled),
        entry_ids=tuple(name for name, _ in sampled),
        decklists=decklists,
        sampling="uniform-independent-v1",
    )

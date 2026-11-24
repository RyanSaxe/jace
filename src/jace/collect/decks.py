"""Human 17Lands Limited deck loading, built-in deck packs, and card-pool curricula.

Reads the cached `decks.pt` tensor archive produced by `jace.data.decks` so self-play
workers and evaluation scripts never re-parse raw 17Lands Parquet files. Applies the
exact deterministic CRC-32 held-out split from `jace.data.store`, builds color-balanced
nested card-pool curricula, and samples compact `UniformDecks` shards so spawned workers
validate hundreds of decks rather than all 91,359 on startup.
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

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from managym.disclosure import DecklistDisclosure

__all__ = (
    "HumanDeckCatalog",
    "build_curriculum_deck_source",
    "build_curriculum_stage_pools",
    "build_human_deck_source",
    "curriculum_blend_spec",
    "elemental_decks",
    "load_human_deck_catalog",
    "needs_curriculum_refresh",
    "nested_card_pools",
    "parse_curriculum_stages",
    "row_to_deck",
    "sample_human_decks",
    "top_card_pool",
)

_BASIC_LAND_ORDER = ("Plains", "Island", "Swamp", "Mountain", "Forest")
_BASIC_LANDS = frozenset(_BASIC_LAND_ORDER)
_MIN_SECOND_COLOR_LANDS = 3
_NUM_COLOR_PAIRS = 10
_MONO_GROUP = 10
_CANDIDATES_PER_PAIR = 250
_BALANCE_SMOOTH = 1.0


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


def _deck_color_pair_groups(decks: Tensor, *, card_names: Sequence[str]) -> Tensor:
    name_to_idx = {name: idx for idx, name in enumerate(card_names)}
    basic_counts = torch.stack(
        [
            decks[:, name_to_idx[land]].float()
            if land in name_to_idx
            else torch.zeros(decks.shape[0], dtype=torch.float32)
            for land in _BASIC_LAND_ORDER
        ],
        dim=1,
    )
    order = basic_counts.argsort(dim=1, descending=True, stable=True)
    first, second = order[:, 0], order[:, 1]
    low, high = torch.minimum(first, second), torch.maximum(first, second)
    pair_map = torch.full((len(_BASIC_LAND_ORDER), len(_BASIC_LAND_ORDER)), _MONO_GROUP, dtype=torch.long)
    color_pairs = [(i, j) for i in range(len(_BASIC_LAND_ORDER)) for j in range(i + 1, len(_BASIC_LAND_ORDER))]
    for pair_id, (first_c, second_c) in enumerate(color_pairs):
        pair_map[first_c, second_c] = pair_id
    second_lands = basic_counts.gather(1, second[:, None]).squeeze(1)
    return torch.where(second_lands >= _MIN_SECOND_COLOR_LANDS, pair_map[low, high], _MONO_GROUP)


def _stratified_candidates(possible: Tensor, *, missing_counts: Tensor, arch: Tensor) -> Tensor:
    buckets: list[Tensor] = []
    for pair_id in range(_NUM_COLOR_PAIRS):
        subset = possible[arch[possible] == pair_id]
        if subset.numel() > 0:
            buckets.append(subset[missing_counts[subset].argsort()[:_CANDIDATES_PER_PAIR]])
    if buckets:
        return torch.cat(buckets)
    return possible[missing_counts[possible].argsort()[:1000]]


def _grow_balanced_pool(
    present: Tensor,
    *,
    arch: Tensor,
    target: int,
    start: Tensor | None = None,
) -> Tensor:
    groups = torch.nn.functional.one_hot(arch, _NUM_COLOR_PAIRS + 1).float()
    in_pool = torch.zeros(present.shape[1], dtype=torch.bool) if start is None else start.clone()
    while int(in_pool.sum().item()) < target:
        missing = present * (~in_pool).float()[None, :]
        counts = missing.sum(dim=1)
        room = target - int(in_pool.sum().item())
        possible = torch.nonzero((counts > 0) & (counts <= room), as_tuple=False).flatten()
        if possible.numel() == 0:
            break
        candidates = _stratified_candidates(possible, missing_counts=counts, arch=arch)
        relevant = torch.nonzero((counts > 0) & (counts <= counts[candidates].max()), as_tuple=False).flatten()
        completes = ((missing[relevant] @ (1.0 - missing[candidates]).T) == 0).float()
        fitting = (groups * (counts == 0).float()[:, None]).sum(dim=0)[:_NUM_COLOR_PAIRS]
        gain = (completes.T @ groups[relevant])[:, :_NUM_COLOR_PAIRS]
        balance = (fitting[None, :] + gain + _BALANCE_SMOOTH).log() - (fitting[None, :] + _BALANCE_SMOOTH).log()
        best = candidates[int((balance.sum(dim=1) / counts[candidates]).argmax().item())]
        in_pool |= present[best] > 0
    return in_pool


def nested_card_pools(catalog: HumanDeckCatalog, *, sizes: Sequence[int]) -> dict[int, frozenset[str]]:
    if not sizes:
        return {}
    ordered = sorted(set(sizes))
    if ordered[0] < 1:
        msg = f"pool sizes must be >= 1, got {ordered[0]}"
        raise ValueError(msg)
    basics = [idx for idx, name in enumerate(catalog.card_names) if name in _BASIC_LANDS]
    non_basics = [idx for idx, name in enumerate(catalog.card_names) if name not in _BASIC_LANDS]
    present = (catalog.train_decks[:, non_basics] > 0).float()
    arch = _deck_color_pair_groups(catalog.train_decks, card_names=catalog.card_names)
    in_pool = torch.zeros(len(non_basics), dtype=torch.bool)
    pools: dict[int, frozenset[str]] = {}
    for size in ordered:
        if size >= len(catalog.card_names):
            pools[size] = frozenset(catalog.card_names)
            continue
        target = max(0, size - len(basics))
        if target == 0:
            pools[size] = frozenset(catalog.card_names[idx] for idx in basics[:size])
            continue
        in_pool = _grow_balanced_pool(present, arch=arch, target=target, start=in_pool)
        kept = [non_basics[pos] for pos, active in enumerate(in_pool.tolist()) if active]
        pools[size] = frozenset(catalog.card_names[idx] for idx in (*basics, *kept))
    return pools


def top_card_pool(catalog: HumanDeckCatalog, *, card_count: int) -> frozenset[str]:
    """Select `card_count` cards via color-pair-balanced deck-completing growth.

    Keeps every basic land in `catalog` and grows the non-basic pool one fitting deck at
    a time, choosing whichever deck maximizes per-card log-balanced gain across the ten
    two-color pairs (`v1` `nested_pools` objective with per-pair candidate stratification).
    This keeps all five colors represented at small pool sizes (e.g. 915 decks across
    W/U/B/R/G inside 90 cards) instead of collapsing into a single two-color pair.
    """
    return nested_card_pools(catalog, sizes=(card_count,))[card_count]


def parse_curriculum_stages(spec: str) -> tuple[tuple[int, int], ...]:
    stages: list[tuple[int, int]] = []
    for part in spec.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        cards_str, start_str = cleaned.split(":", maxsplit=1)
        stages.append((int(cards_str), int(start_str)))
    if not stages:
        msg = f"empty curriculum stages spec: {spec!r}"
        raise ValueError(msg)
    return tuple(sorted(stages, key=lambda item: item[1]))


def curriculum_blend_spec(
    update_idx: int,
    *,
    stages: Sequence[tuple[int, int]],
    blend_updates: int = 10,
) -> tuple[int, int, float]:
    active_idx = 0
    for idx, (_, start_u) in enumerate(stages):
        if update_idx > start_u or idx == 0:
            active_idx = idx
    next_size, start_u = stages[active_idx]
    if active_idx == 0 or blend_updates <= 0:
        return next_size, next_size, 1.0
    prev_size = stages[active_idx - 1][0]
    progress = min(1.0, max(0.0, float(update_idx - start_u) / float(blend_updates)))
    return prev_size, next_size, progress


def needs_curriculum_refresh(
    update_idx: int,
    *,
    stages: Sequence[tuple[int, int]],
    blend_updates: int = 10,
) -> bool:
    return any(start_u < update_idx <= start_u + max(1, blend_updates) for _, start_u in stages[1:])


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


def elemental_decks() -> list[tuple[str, managym.Deck]]:
    pack = "elemental-world-v2"
    result: list[tuple[str, managym.Deck]] = []
    for key in managym.authored_deck_keys(pack):
        setup = managym.authored_deck_setup(pack, key)
        result.append((key, managym.Deck(dict(setup.decklist), dict(setup.sideboard))))
    return result


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


def build_curriculum_stage_pools(
    catalog: HumanDeckCatalog, *, spec: str | None
) -> tuple[tuple[int, int, frozenset[str]], ...]:
    if spec is None or not spec.strip():
        return ()
    stages = parse_curriculum_stages(spec)
    pools = nested_card_pools(catalog, sizes=[size for size, _ in stages])
    return tuple((size, start_u, pools[size]) for size, start_u in stages)


def build_curriculum_deck_source(
    catalog: HumanDeckCatalog,
    *,
    count: int,
    seed: int,
    update_idx: int,
    stage_pools: Sequence[tuple[int, int, frozenset[str]]],
    blend_updates: int = 10,
    decklists: DecklistDisclosure = "closed",
) -> UniformDecks:
    stages = [(size, start_u) for size, start_u, _ in stage_pools]
    pool_map = {size: cards for size, _, cards in stage_pools}
    prev_size, next_size, next_share = curriculum_blend_spec(update_idx, stages=stages, blend_updates=blend_updates)
    next_count = round(count * next_share)
    prev_count = count - next_count
    sampled: list[tuple[str, managym.Deck]] = []
    if prev_count > 0:
        sampled.extend(
            sample_human_decks(catalog, split="train", count=prev_count, seed=seed, allowed_cards=pool_map[prev_size])
        )
    if next_count > 0:
        sampled.extend(
            sample_human_decks(
                catalog, split="train", count=next_count, seed=seed + 17, allowed_cards=pool_map[next_size]
            )
        )
    return UniformDecks(
        decks=tuple(deck for _, deck in sampled),
        entry_ids=tuple(f"{name}-c{pos}" for pos, (name, _) in enumerate(sampled)),
        decklists=decklists,
        sampling="uniform-independent-v1",
    )

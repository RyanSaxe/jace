"""Human drafts from 17Lands pick rows, as tensors a drafter can train on.

17Lands publishes one row per pick with one column per card for the pack
and for the pool so far. A draft becomes one sequence of steps, numbered
across rounds, so a model can read the whole draft. Some sets, Avatar among
them, lack every draft's first pick because the game client never logged
it. The pool at the second pick still names the card taken, so the first
step is rebuilt as a pack that holds only that card, with weight zero: the
model reads it as history and is never trained or scored on it. Drafts that
lack any later pick are dropped.

The sample weights follow an earlier human-trained drafter by the same
author: picks by stronger players and from events that went well count for
more, and the first picks of each round count for less, because those are
where humans most often take a rare card over a better one.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
from torch import Tensor

from jace.cards import CardPool
from jace.data.store import SPLIT_KEYS, Tensors, cached, card_columns, split_keys

__all__ = ("RESULT_FLOOR", "HumanDrafts", "PoolAtSecondPickError", "load_drafts")

_CACHE_FILE = "drafts.pt"
# Pick rows read into memory at a time. Each row holds one int8 per card for the pack.
_ROWS_PER_READ = 500_000

# How much a player's rank says about their picks, from 0 to 1, as the earlier drafter set it.
_RANK_SCORES = {"bronze": 0.01, "silver": 0.1, "gold": 0.25, "platinum": 0.5, "diamond": 0.75, "mythic": 1.0}
_RANKS = tuple(_RANK_SCORES)
# Used for a player with no recorded rank, and for one with no recorded win rate.
_UNKNOWN_RANK_SCORE = 0.5
_UNKNOWN_WIN_RATE = 0.5
# The skill weight is win rate ** (_SKILL_POWER - rank score), kept within these bounds, so
# rank and win rate count together and no player's picks are ignored.
_SKILL_POWER = 2.0
_SKILL_FLOOR = 0.1
# An event's result weighs its picks and its deck by the share it won, kept at or above this.
RESULT_FLOOR = 0.5
# Pick x of a round weighs (ln x + 1) / x ** _POSITION_POWER: low for the first pick, highest
# for the second, then falling, with the eighth pick about equal to the first.
_POSITION_POWER = math.e / 5


@dataclass(frozen=True, slots=True)
class HumanDrafts:
    """Complete human drafts, one row per draft and one step per pick.

    `packs` is int16 `[drafts, steps, pack_size]`: the card ids in the pack
    at each step, then -1. A card with two copies in a pack appears twice.
    `picks` is int16 `[drafts, steps]`, the card taken. `weights` is float32
    `[drafts, steps]`, the sample weight of each pick; zero marks a first
    pick rebuilt from the pool, whose pack holds only the card taken.
    `held_out` is bool `[drafts]`, True for drafts kept out of training.
    """

    packs: Tensor
    picks: Tensor
    weights: Tensor
    held_out: Tensor


class PoolAtSecondPickError(ValueError):
    def __init__(self, drafts: int) -> None:
        super().__init__(f"{drafts} drafts lack a first pick and do not hold exactly one card at their second pick")


def load_drafts(rows: pl.LazyFrame, *, pool: CardPool, cache_dir: Path, held_out_share: float) -> HumanDrafts:
    """Turn 17Lands pick rows into tensors, reading `rows` only when `cache_dir` has no copy.

    `rows` needs the columns `draft_id`, `rank`, `user_game_win_rate_bucket`,
    `event_match_wins`, `event_match_losses`, `pack_number`, `pick_number`
    and `pick`, plus `pack_card_<name>` and `pool_<name>` for every card of
    `pool`. The first call saves `drafts.pt` in `cache_dir` and later calls
    load it, so use one directory per card set. Drafts are ordered by id.
    A draft is held out when a checksum of its id falls in the lowest
    `held_out_share` of values, so the split is by draft and never changes.
    """
    saved = cached(cache_dir / _CACHE_FILE, lambda: _read(rows, pool=pool), pool=pool)
    packs = saved["packs"]
    win_rate = torch.where(saved["win_rate"].isnan(), _UNKNOWN_WIN_RATE, saved["win_rate"])
    rank_scores = torch.tensor([*_RANK_SCORES.values(), _UNKNOWN_RANK_SCORE])
    skill = win_rate.pow(_SKILL_POWER - rank_scores[saved["rank"].long()]).clamp(_SKILL_FLOOR, 1.0)
    matches = (saved["wins"] + saved["losses"]).float()
    result = (saved["wins"] / matches.clamp_min(1.0)).clamp(RESULT_FLOOR, 1.0)
    picks_per_round = packs.shape[2]
    pick_in_round = torch.arange(packs.shape[1]) % picks_per_round + 1
    position_weight = (pick_in_round.log() + 1) / pick_in_round.pow(_POSITION_POWER)
    weights = (skill * result)[:, None] * position_weight[None, :]
    # A pack that opens a round with one card is a first pick rebuilt from the pool.
    rebuilt = packs[:, 0, 1] < 0 if picks_per_round > 1 else torch.zeros(len(packs), dtype=torch.bool)
    weights[rebuilt, 0] = 0.0
    return HumanDrafts(
        packs=packs,
        picks=saved["picks"],
        weights=weights,
        held_out=saved["split_key"] < round(held_out_share * SPLIT_KEYS),
    )


def _read(rows: pl.LazyFrame, *, pool: CardPool) -> Tensors:
    columns = rows.collect_schema().names()
    pack_columns = card_columns(columns, "pack_card_", pool=pool)
    pool_columns = card_columns(columns, "pool_", pool=pool)
    card_ids = {pool[card].name: card for card in range(len(pool))}
    picks_per_round = int(rows.select(pl.col("pick_number").max() + 1).collect().item())
    steps = picks_per_round * int(rows.select(pl.col("pack_number").max() + 1).collect().item())
    step = pl.col("pack_number").cast(pl.Int32) * picks_per_round + pl.col("pick_number")
    described = rows.select(
        "draft_id",
        step.alias("step"),
        pl.col("pick").replace_strict(card_ids, return_dtype=pl.Int16),
        pl.col("rank").replace_strict({rank: index for index, rank in enumerate(_RANKS)}, default=len(_RANKS)),
        pl.col("user_game_win_rate_bucket").alias("win_rate"),
        pl.col("event_match_wins").alias("wins"),
        pl.col("event_match_losses").alias("losses"),
    ).collect()
    # A complete draft has every step, or every step but the first.
    drafts = (
        described.filter(pl.col("step") > 0)
        .group_by("draft_id")
        .agg(pl.col("step").n_unique().alias("steps"), pl.col("rank", "win_rate", "wins", "losses").first())
        .filter(pl.col("steps") == steps - 1)
        .sort("draft_id")
        .with_row_index("draft")
    )
    placed = described.join(drafts.select("draft_id", "draft"), on="draft_id", how="left", maintain_order="left")
    draft_of_row = torch.from_numpy(
        placed["draft"].fill_null(-1).cast(pl.Int64).to_numpy(writable=True)
    )  # [rows], -1 to skip
    step_of_row = torch.tensor(placed["step"].to_list(), dtype=torch.int64)  # [rows]
    kept = draft_of_row >= 0

    picks = torch.zeros(len(drafts), steps, dtype=torch.int16)
    picks[draft_of_row[kept], step_of_row[kept]] = torch.from_numpy(placed["pick"].to_numpy(writable=True))[kept]
    packs = torch.full((len(drafts), steps, picks_per_round), -1, dtype=torch.int16)
    for start in range(0, len(placed), _ROWS_PER_READ):
        pack_counts = torch.from_numpy(rows.slice(start, _ROWS_PER_READ).select(pack_columns).collect().to_numpy())
        chunk = slice(start, start + len(pack_counts))
        row, slot, card = _pack_slots(pack_counts[kept[chunk]])
        packs[draft_of_row[chunk][kept[chunk]][row], step_of_row[chunk][kept[chunk]][row], slot] = card.to(torch.int16)

    missing_first = packs[:, 0, 0] < 0
    if missing_first.any():
        second = rows.filter(step == 1).select("draft_id", *pool_columns).collect()
        second = second.join(drafts.select("draft_id", "draft"), on="draft_id").sort("draft")
        second_pool = torch.from_numpy(second.select(pool_columns).to_numpy())  # int8 [drafts, cards]
        if len(second_pool) != len(drafts) or bool((second_pool.sum(dim=1) != 1)[missing_first].any()):
            raise PoolAtSecondPickError(int(missing_first.sum()))
        first_picks = second_pool.argmax(dim=1).to(torch.int16)
        picks[missing_first, 0] = first_picks[missing_first]
        packs[missing_first, 0, 0] = first_picks[missing_first]
    return {
        "packs": packs,
        "picks": picks,
        "rank": torch.tensor(drafts["rank"].to_list(), dtype=torch.int8),
        "win_rate": torch.tensor(drafts["win_rate"].fill_null(float("nan")).to_list(), dtype=torch.float32),
        "wins": torch.tensor(drafts["wins"].to_list(), dtype=torch.int16),
        "losses": torch.tensor(drafts["losses"].to_list(), dtype=torch.int16),
        "split_key": split_keys(drafts["draft_id"].to_list()),
    }


def _pack_slots(counts: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    # counts: int8 [rows, cards], copies of each card in each row's pack. Returns int64 (row, slot,
    # card), one entry per copy: the card sits in `slot` of its row's pack, counted from 0.
    row, card = counts.nonzero(as_tuple=True)
    copies = counts[row, card].long()
    row, card = row.repeat_interleave(copies), card.repeat_interleave(copies)
    sizes = counts.sum(dim=1, dtype=torch.int64)
    first_entry = sizes.cumsum(dim=0) - sizes  # [rows], where each row's entries start
    return row, torch.arange(len(row)) - first_entry[row], card

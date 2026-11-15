"""Human decks from 17Lands game rows: one deck and pool per draft, with the event's result.

17Lands publishes one row per game, and a player may rebuild the deck
between games, so a draft can have several decks. `load_decks` keeps the
build the player used in the most games, and on a tie the one with the
lowest `build_index`, so each draft gives one example. The wins and losses
include every game of the event, whichever build the player used in it.

The deck score and the deck builder train on these decks in
`jace.loops.pretrain_decks`. Nothing in this module trains a model.
"""

from dataclasses import dataclass
from pathlib import Path

import polars as pl
import torch
from torch import Tensor

from jace.cards import CardPool
from jace.data.store import SPLIT_KEYS, Tensors, cached, card_columns, split_keys

__all__ = ("HumanDecks", "load_decks")

_CACHE_FILE = "decks.pt"


@dataclass(frozen=True, slots=True)
class HumanDecks:
    """One human deck per draft that has game data, ordered by draft id.

    `decks` is uint8 `[drafts, cards]`, the copies of each card in the main
    deck, basic lands included. `pools` is uint8 `[drafts, cards]`, the deck
    plus its sideboard, which are the cards the player could build from.

    `wins` and `losses` are int16 `[drafts]`, the games the player won and
    lost in the event. `held_out` is bool `[drafts]`, split the same way as
    the pick data in `HumanDrafts`.
    """

    decks: Tensor
    pools: Tensor
    wins: Tensor
    losses: Tensor
    held_out: Tensor


def load_decks(games: pl.LazyFrame, *, pool: CardPool, cache_dir: Path, held_out_share: float) -> HumanDecks:
    """Turn 17Lands game rows into tensors, reading `games` only when `cache_dir` has no copy.

    `games` must have the columns `draft_id`, `build_index` and `won`, plus
    `deck_<name>` and `sideboard_<name>` for every card of `pool`. The first
    call saves `decks.pt` in `cache_dir` and later calls load it, so use one
    directory per card set. A draft is held out when a checksum of its id
    falls in the lowest `held_out_share` of values, as in `load_drafts`.

    Raises `StaleCacheError` when `decks.pt` was saved for a pool with other
    card names, and `CardColumnsError` when the card columns of `games` do
    not match the cards of `pool`.
    """
    saved = cached(cache_dir / _CACHE_FILE, lambda: _read(games, pool=pool), pool=pool)
    return HumanDecks(
        decks=saved["decks"],
        pools=saved["pools"],
        wins=saved["wins"],
        losses=saved["losses"],
        held_out=saved["split_key"] < round(held_out_share * SPLIT_KEYS),
    )


def _read(games: pl.LazyFrame, *, pool: CardPool) -> Tensors:
    columns = games.collect_schema().names()
    deck_columns = card_columns(columns, "deck_", pool=pool)
    sideboard_columns = card_columns(columns, "sideboard_", pool=pool)
    numbered = games.with_row_index("row")
    described = numbered.select("row", "draft_id", "build_index", "won").collect()
    events = described.group_by("draft_id").agg(
        pl.col("won").sum().alias("wins"), pl.col("won").not_().sum().alias("losses")
    )
    builds = (
        described.group_by("draft_id", "build_index")
        .agg(pl.len().alias("games"), pl.col("row").min())
        .sort("draft_id", "games", "build_index", descending=[False, True, False])
        .unique("draft_id", keep="first", maintain_order=True)
        .join(events, on="draft_id", maintain_order="left")
    )
    chosen = numbered.filter(pl.col("row").is_in(builds["row"])).select("draft_id", *deck_columns, *sideboard_columns)
    cards = builds.select("draft_id").join(chosen.collect(), on="draft_id", maintain_order="left")
    decks = torch.from_numpy(cards.select(deck_columns).to_numpy()).to(torch.uint8)
    sideboards = torch.from_numpy(cards.select(sideboard_columns).to_numpy()).to(torch.uint8)
    return {
        "decks": decks,
        "pools": decks + sideboards,
        "wins": torch.tensor(builds["wins"].to_list(), dtype=torch.int16),
        "losses": torch.tensor(builds["losses"].to_list(), dtype=torch.int16),
        "split_key": split_keys(builds["draft_id"].to_list()),
    }

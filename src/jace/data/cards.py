"""Build a set's card pool and per-card metadata from MTGJSON and 17Lands tables.

`card_table` aligns MTGJSON card attributes, 17Lands average pick position
(`ata`), and `managym` engine programs into one `SetCards` record in card id
order. `is_land` and `mana_value` support deck-building rules and draft priors,
and `ata` supports the average-pick baseline. No card vector includes `ata`.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl
import torch
from torch import Tensor

from jace.cards import CardPool, CardRecord, Rarity

__all__ = ("SetCards", "card_table")

# MTGJSON gives basic lands the rarity "common", so `card_table` recognises basics by name.
_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})


@dataclass(frozen=True, slots=True)
class SetCards:
    """One card set: its pool, and per-card tensors in card id order.

    `is_land` is bool `[cards]`, true for every land card in the set. `ata` is
    float32 `[cards]`, the average pick number at which humans took each card,
    and NaN for a card nobody took. `mana_value` is float32 `[cards]`, the
    printed mana value.
    """

    pool: CardPool
    is_land: Tensor
    ata: Tensor
    mana_value: Tensor


def card_table(
    attributes: pl.DataFrame, statistics: pl.DataFrame, *, programs: Mapping[str, tuple[int, ...]]
) -> SetCards:
    """Read a set's card pool, land flags, mana values, and average pick position.

    `attributes` has one row per card, and the card ids follow its row
    order. It must have the columns `name`, `set_code`, `rarity`, `card_type`,
    `mana_value`, and `oracle_text`. `statistics` must have a `name` column
    and an `ata` column.

    `programs` maps a printed name to the engine's program for that card, a
    flat run of (kind, value) integer pairs. A card missing from `programs`
    gets an empty program, which marks it as a card the engine cannot play.

    `read_set` returns statistics that spells aggregated over every draft of
    the set, held-out drafts included. A baseline that ranks cards by `ata`
    therefore ranks them by statistics that include the held-out picks, and
    its held-out score is a little too high.
    """
    cards = attributes.join(statistics, on="name", how="left", maintain_order="left")
    records = [
        CardRecord(
            name=name,
            set_code=set_code,
            rarity=Rarity.BASIC if name in _BASIC_LANDS else Rarity(rarity),
            text=text,
            program=programs.get(name, ()),
        )
        for name, set_code, rarity, text in cards.select("name", "set_code", "rarity", "oracle_text").iter_rows()
    ]
    is_land = torch.tensor(cards["card_type"].str.contains("Land").fill_null(value=False).to_list(), dtype=torch.bool)
    mana_value = torch.tensor(cards["mana_value"].cast(pl.Float64, strict=False).to_numpy(), dtype=torch.float32)
    return SetCards(
        pool=CardPool(records),
        is_land=is_land,
        ata=torch.tensor(cards["ata"].cast(pl.Float64).to_numpy(), dtype=torch.float32),
        mana_value=mana_value,
    )

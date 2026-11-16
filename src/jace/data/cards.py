"""Build the card pool and the facts printed on each card from MTGJSON and 17Lands tables.

Besides a card's text and engine program, the card encoder reads a few
facts printed on the card. They are its mana value, power, toughness,
colours, types and rarity, and whether it is on the bonus sheet. All of
them are known before anyone plays the card, so a set with no human data
has them too.

Of the 17Lands statistics, only each card's average pick position is read,
for a baseline that ranks cards by it. No card vector includes it.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl
import torch
from torch import Tensor

from jace.cards import CardPool, CardRecord, Rarity
from jace.data.store import SMALLEST_SPREAD

__all__ = ("SetCards", "card_table")

# MTGJSON gives basic lands the rarity "common", so `card_table` recognises basics by name.
_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
_COLOURS = ("W", "U", "B", "R", "G")
_TYPES = ("Creature", "Instant", "Sorcery", "Enchantment", "Artifact", "Land")
# The printed numbers, which `card_table` standardises. Every card has a mana value, so only power and toughness
# get a column that marks them missing.
_NUMBERS = ("mana_value", "power", "toughness")
_SOMETIMES_MISSING = ("power", "toughness")


@dataclass(frozen=True, slots=True)
class SetCards:
    """One card set: its pool, and per-card tensors in card id order.

    `printed` is float32 `[cards, printed facts]`, and `printed_names`
    gives each column's name. Mana value, power and toughness have mean
    zero and spread one over the cards that have a value. A card with no
    power or toughness, or one printed as `*`, gets zero in that column and
    one in the `no_power` or `no_toughness` column. The other columns are
    0/1 flags for the colours, the types, the bonus sheet and the rarity.

    `ata` is float32 `[cards]`, the average pick number at which humans
    took each card, and NaN for a card nobody took. `mana_value` is float32
    `[cards]`, the mana value as printed, not standardised.
    """

    pool: CardPool
    printed_names: tuple[str, ...]
    printed: Tensor
    ata: Tensor
    mana_value: Tensor


def card_table(
    attributes: pl.DataFrame, statistics: pl.DataFrame, *, programs: Mapping[str, tuple[int, ...]]
) -> SetCards:
    """Read a set's printed card attributes, and the average pick position from its 17Lands statistics.

    `attributes` has one row per card, and the card ids follow its row
    order. It must have the columns `name`, `set_code`, `rarity`, `color`,
    `card_type`, `mana_value`, `power`, `toughness`, `is_bonus_sheet` and
    `oracle_text`. `statistics` must have a `name` column and an `ata`
    column.

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
    numbers = cards.select(*[pl.col(number).cast(pl.Float64, strict=False) for number in _NUMBERS])
    flags = cards.select(
        *[pl.col("color").str.contains(colour).alias(f"colour_{colour}") for colour in _COLOURS],
        *[pl.col("card_type").str.contains(kind).alias(f"type_{kind.lower()}") for kind in _TYPES],
        pl.col("is_bonus_sheet"),
    )
    rarities = torch.tensor([[record.rarity is rarity for rarity in Rarity] for record in records])
    values = torch.from_numpy(numbers.to_numpy()).float()  # [cards, numbers], NaN where a card has no value
    absent = values.isnan()
    mean = values.nanmean(dim=0)
    spread = (values - mean).pow(2).nanmean(dim=0).sqrt().clamp_min(SMALLEST_SPREAD)
    missing = [_NUMBERS.index(number) for number in _SOMETIMES_MISSING]
    printed = torch.cat(
        [
            torch.where(absent, 0.0, (values - mean) / spread),
            absent[:, missing].float(),
            torch.from_numpy(flags.to_numpy()).float(),
            rarities.float(),
        ],
        dim=1,
    )
    return SetCards(
        pool=CardPool(records),
        printed_names=(
            *_NUMBERS,
            *[f"no_{number}" for number in _SOMETIMES_MISSING],
            *flags.columns,
            *[f"rarity_{rarity.value}" for rarity in Rarity],
        ),
        printed=printed,
        ata=torch.tensor(cards["ata"].cast(pl.Float64).to_numpy(), dtype=torch.float32),
        mana_value=values[:, _NUMBERS.index("mana_value")].clone(),
    )

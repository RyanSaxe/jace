"""Build the card pool and the facts printed on each card from MTGJSON and 17Lands tables.

The printed facts are what the card encoder reads besides a card's text and
engine program: mana value, power, toughness, colours, types, rarity and
whether the card is on the bonus sheet. All of them are known before anyone
plays a card, so they exist for a set with no human data. The 17Lands
statistics are read only for each card's average pick position, which a
baseline ranks by; no card vector holds them. MTGJSON gives basic lands the
rarity "common", so basics are recognised by name.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl
import torch
from torch import Tensor

from jace.cards import CardPool, CardRecord, Rarity
from jace.data.store import SMALLEST_SPREAD

__all__ = ("SetCards", "card_table")

_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
_COLOURS = ("W", "U", "B", "R", "G")
_TYPES = ("Creature", "Instant", "Sorcery", "Enchantment", "Artifact", "Land")
# The printed numbers, standardised. Every card has a mana value, so only power and toughness
# get a column that marks them missing.
_NUMBERS = ("mana_value", "power", "toughness")
_SOMETIMES_MISSING = ("power", "toughness")


@dataclass(frozen=True, slots=True)
class SetCards:
    """One card set: its pool, and per-card tensors in card id order.

    `printed` is float32 `[cards, printed facts]`, with one name per column
    in `printed_names`. Mana value, power and toughness have mean zero and
    spread one over the cards that have a value. A card with no power or
    toughness, or one printed as `*`, gets zero there and one in the
    `no_power` or `no_toughness` column. The other columns are 0/1 flags
    for the colours, the types, the bonus sheet and the rarity. `ata` is
    float32 `[cards]`, the average pick number a card was taken at, NaN for
    a card nobody took. `mana_value` is float32 `[cards]`, as printed.
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

    `attributes` has one row per card with the columns `name`, `set_code`,
    `rarity`, `color`, `card_type`, `mana_value`, `power`, `toughness`,
    `is_bonus_sheet` and `oracle_text`, and its row order becomes the card
    ids. `statistics` has a `name` column and an `ata` column. `programs`
    maps a printed name to the engine's tokens for that card. A card it
    lacks gets an empty program, which marks it as one the engine cannot
    play.

    `read_set` returns the statistics aggregated over every draft of the
    set, held-out drafts included, so a baseline that ranks by `ata` has
    seen a summary of the held-out picks and its held-out score is a little
    too high.
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

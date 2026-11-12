"""Build the card pool and each card's starting vector from 17Lands and MTGJSON tables.

The starting vectors here stand in for the planned language-model card
encoder. Until that exists, a card is described by how humans drafted and
played it (17Lands statistics such as win rate in hand and average pick
position) and by a few printed facts: mana value, colours, types, rarity,
power and toughness. Statistics come from the same drafts the drafter is
trained on, held-out drafts included, so they describe this card set and
say nothing about a set with no human data. MTGJSON gives basic lands the
rarity "common", so basics are recognised by name.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl
import torch
from torch import Tensor

from jace.cards import CardPool, CardRecord, Rarity

__all__ = ("SetCards", "card_table")

_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
_COLOURS = ("W", "U", "B", "R", "G")
_TYPES = ("Creature", "Instant", "Sorcery", "Enchantment", "Artifact", "Land")
# 17Lands statistics used as features, by their spells column names: win rate with the card in
# hand, in the opening hand and in the deck, the win rate gained by drawing it, the pick number
# it was last seen at, the pick number it was taken at, and the share of its copies that made a deck.
_STATISTICS = ("gih_wr", "oh_wr", "gp_wr", "iwd", "alsa", "ata", "pct_gp")
# A column whose values are all equal has a spread of zero. Dividing by this instead leaves it at zero.
_SMALLEST_SPREAD = 1e-6


@dataclass(frozen=True, slots=True)
class SetCards:
    """One card set: its pool, and per-card tensors in card id order.

    `features` is float32 `[cards, features]`, the starting vectors for a
    `CardEmbedding`, with one name per column in `feature_names`. Measured
    columns have mean zero and spread one over the cards that have a value.
    A card with no value gets zero there and one in that column's
    `no_<name>` column. `ata` is float32 `[cards]`, the average pick number
    a card was taken at, NaN for a card nobody took. `mana_value` is
    float32 `[cards]`.
    """

    pool: CardPool
    feature_names: tuple[str, ...]
    features: Tensor
    ata: Tensor
    mana_value: Tensor


def card_table(
    attributes: pl.DataFrame, statistics: pl.DataFrame, *, programs: Mapping[str, tuple[int, ...]]
) -> SetCards:
    """Join a set's printed card attributes with its 17Lands statistics.

    `attributes` has one row per card with the columns `name`, `set_code`,
    `rarity`, `color`, `card_type`, `mana_value`, `power`, `toughness`,
    `is_bonus_sheet` and `oracle_text`, and its row order becomes the card
    ids. `statistics` has a `name` column, `num_gih` and the 17Lands
    columns this module reads. `programs` maps a printed name to the
    engine's tokens for that card. A card it lacks gets an empty program,
    which marks it as one the engine cannot play.
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
    measured = cards.select(
        *[pl.col(column).cast(pl.Float64) for column in _STATISTICS],
        pl.col("num_gih").cast(pl.Float64).log1p().alias("log_games_in_hand"),
        pl.col("mana_value").cast(pl.Float64),
        pl.col("power").cast(pl.Float64, strict=False),
        pl.col("toughness").cast(pl.Float64, strict=False),
    )
    flags = cards.select(
        *[pl.col("color").str.contains(colour).alias(f"colour_{colour}") for colour in _COLOURS],
        *[pl.col("card_type").str.contains(kind).alias(f"type_{kind.lower()}") for kind in _TYPES],
        pl.col("is_bonus_sheet"),
    )
    rarities = torch.tensor([[record.rarity is rarity for rarity in Rarity] for record in records])
    values = torch.from_numpy(measured.to_numpy()).float()  # [cards, measured], NaN where a card has no value
    absent = values.isnan()
    mean = values.nanmean(dim=0)
    spread = (values - mean).pow(2).nanmean(dim=0).sqrt().clamp_min(_SMALLEST_SPREAD)
    standardised = torch.where(absent, 0.0, (values - mean) / spread)
    sometimes_absent = absent.any(dim=0)
    features = torch.cat(
        [
            standardised,
            absent[:, sometimes_absent].float(),
            torch.from_numpy(flags.to_numpy()).float(),
            rarities.float(),
        ],
        dim=1,
    )
    names = (
        *measured.columns,
        *[f"no_{name}" for name, some in zip(measured.columns, sometimes_absent.tolist(), strict=True) if some],
        *flags.columns,
        *[f"rarity_{rarity.value}" for rarity in Rarity],
    )
    return SetCards(
        pool=CardPool(records),
        feature_names=names,
        features=features,
        ata=values[:, measured.columns.index("ata")].clone(),
        mana_value=values[:, measured.columns.index("mana_value")].clone(),
    )

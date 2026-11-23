"""Check the human-data loaders through `jace.data`, on small hand-made tables.

No test downloads anything or reads a set through the `spells` package.
Most tests build the few pick or game rows they need in the shape 17Lands
publishes them, with one column per card, and pass them to the same
functions that read the real files. The card-table test builds a small
table of card attributes and one of card statistics instead. The cache
files go to pytest's temporary folder.
"""

from pathlib import Path

import polars as pl
import pytest
import torch

from jace.cards import (
    FIRST_PROGRAM_TOKEN,
    NO_PROGRAM_TOKEN,
    PADDING_TOKEN,
    UNKNOWN_TOKEN,
    CardPool,
    CardRecord,
    Rarity,
)
from jace.data import StaleCacheError, card_table, load_decks, load_drafts, program_tokens

__all__ = ()

_NAMES = ("Ant", "Bat", "Cat")


def _pool(names: tuple[str, ...]) -> CardPool:
    records = [CardRecord(name=name, set_code="TST", rarity=Rarity.COMMON, text="", program=()) for name in names]
    return CardPool(records)


def _pick_row(
    draft: str, *, pack_number: int, pick_number: int, pick: str, pack: tuple[int, ...], held: tuple[int, ...]
) -> dict[str, object]:
    # `pack` and `held` give the copies of Ant, Bat and Cat in the pack and in the pool so far.
    return {
        "draft_id": draft,
        "rank": "gold",
        "user_game_win_rate_bucket": 0.6,
        "event_match_wins": 3,
        "event_match_losses": 3,
        "pack_number": pack_number,
        "pick_number": pick_number,
        "pick": pick,
        **{f"pack_card_{name}": copies for name, copies in zip(_NAMES, pack, strict=True)},
        **{f"pool_{name}": copies for name, copies in zip(_NAMES, held, strict=True)},
    }


def _two_round_drafts() -> pl.LazyFrame:
    # Two rounds of two picks each. Draft "full" has a row for every pick except its first. Draft "cut" also has
    # no row for its last pick.
    rows = [
        _pick_row("full", pack_number=0, pick_number=1, pick="Bat", pack=(0, 1, 0), held=(0, 0, 1)),
        _pick_row("full", pack_number=1, pick_number=0, pick="Ant", pack=(1, 0, 1), held=(0, 1, 1)),
        _pick_row("full", pack_number=1, pick_number=1, pick="Cat", pack=(0, 0, 1), held=(1, 1, 1)),
        _pick_row("cut", pack_number=0, pick_number=1, pick="Ant", pack=(1, 0, 0), held=(0, 1, 0)),
        _pick_row("cut", pack_number=1, pick_number=0, pick="Cat", pack=(0, 1, 1), held=(1, 1, 0)),
    ]
    return pl.LazyFrame(rows)


def test_a_draft_missing_only_its_first_pick_gets_it_back_from_the_pool_and_never_trains_on_it(tmp_path: Path) -> None:
    drafts = load_drafts(_two_round_drafts(), pool=_pool(_NAMES), cache_dir=tmp_path, held_out_share=0.0)

    assert drafts.picks.tolist() == [[2, 1, 0, 2]]
    assert drafts.packs.tolist() == [[[2, -1], [1, -1], [0, 2], [2, -1]]]
    assert drafts.weights[0, 0] == 0
    assert bool((drafts.weights[0, 1:] > 0).all())


def test_a_pack_with_two_copies_of_a_card_lists_it_twice(tmp_path: Path) -> None:
    rows = [
        _pick_row("twin", pack_number=0, pick_number=0, pick="Ant", pack=(2, 0, 1), held=(0, 0, 0)),
        _pick_row("twin", pack_number=0, pick_number=1, pick="Cat", pack=(1, 0, 1), held=(1, 0, 0)),
        _pick_row("twin", pack_number=0, pick_number=2, pick="Ant", pack=(1, 0, 0), held=(1, 0, 1)),
    ]

    drafts = load_drafts(pl.LazyFrame(rows), pool=_pool(_NAMES), cache_dir=tmp_path, held_out_share=0.0)

    assert drafts.packs.tolist() == [[[0, 0, 2], [0, 2, -1], [0, -1, -1]]]
    assert drafts.picks.tolist() == [[0, 2, 0]]


def test_a_draft_is_held_out_of_its_picks_and_of_its_deck_alike(tmp_path: Path) -> None:
    ids = [f"draft-{number}" for number in range(12)]
    picks = [
        _pick_row(draft, pack_number=0, pick_number=pick, pick="Ant", pack=(2 - pick, 0, 0), held=(pick, 0, 0))
        for draft in ids
        for pick in range(2)
    ]
    games = [
        {"draft_id": draft, "build_index": 0, "won": True}
        | {f"deck_{name}": 1 for name in _NAMES}
        | {f"sideboard_{name}": 0 for name in _NAMES}
        for draft in reversed(ids)
    ]

    drafts = load_drafts(pl.LazyFrame(picks), pool=_pool(_NAMES), cache_dir=tmp_path, held_out_share=0.5)
    decks = load_decks(pl.LazyFrame(games), pool=_pool(_NAMES), cache_dir=tmp_path, held_out_share=0.5)

    assert 0 < int(drafts.held_out.sum()) < len(ids)
    assert torch.equal(drafts.held_out, decks.held_out)


def test_a_second_load_returns_the_cached_drafts_without_reading_the_rows(tmp_path: Path) -> None:
    first = load_drafts(_two_round_drafts(), pool=_pool(_NAMES), cache_dir=tmp_path, held_out_share=0.0)

    second = load_drafts(pl.LazyFrame(), pool=_pool(_NAMES), cache_dir=tmp_path, held_out_share=0.0)

    assert torch.equal(first.packs, second.packs)
    assert torch.equal(first.picks, second.picks)


def test_a_cache_built_for_other_cards_is_rejected(tmp_path: Path) -> None:
    load_drafts(_two_round_drafts(), pool=_pool(_NAMES), cache_dir=tmp_path, held_out_share=0.0)

    with pytest.raises(StaleCacheError, match="different card pool"):
        load_drafts(pl.LazyFrame(), pool=_pool(("Ant", "Bat", "Dog")), cache_dir=tmp_path, held_out_share=0.0)


def test_a_draft_with_two_builds_gives_the_build_played_most_and_the_whole_events_result(tmp_path: Path) -> None:
    def game(build: int, *, won: bool, deck: tuple[int, ...], sideboard: tuple[int, ...]) -> dict[str, object]:
        return {
            "draft_id": "one",
            "build_index": build,
            "won": won,
            **{f"deck_{name}": copies for name, copies in zip(_NAMES, deck, strict=True)},
            **{f"sideboard_{name}": copies for name, copies in zip(_NAMES, sideboard, strict=True)},
        }

    games = pl.LazyFrame(
        [
            game(0, won=False, deck=(2, 0, 0), sideboard=(0, 1, 1)),
            game(1, won=True, deck=(1, 1, 0), sideboard=(1, 0, 1)),
            game(1, won=False, deck=(1, 1, 0), sideboard=(1, 0, 1)),
        ]
    )

    decks = load_decks(games, pool=_pool(_NAMES), cache_dir=tmp_path, held_out_share=0.0)

    assert decks.decks.tolist() == [[1, 1, 0]]
    assert decks.pools.tolist() == [[2, 1, 1]]
    assert (decks.wins.tolist(), decks.losses.tolist()) == ([1], [2])


def test_card_table_aligns_pool_lands_mana_values_programs_and_ata() -> None:
    attributes = pl.DataFrame(
        {
            "name": ["Ant", "Bat", "Plains"],
            "set_code": ["TST", "TST", "TST"],
            "rarity": ["rare", "common", "common"],
            "card_type": ["Creature", "Creature", "Land"],
            "mana_value": [2.0, 4.0, 0.0],
            "oracle_text": ["", "", ""],
        }
    )
    statistics = pl.DataFrame({"name": ["Plains", "Ant", "Bat"], "ata": [10.0, 2.0, 6.0]})

    cards = card_table(attributes, statistics, programs={"Ant": (7, 7)})

    assert cards.is_land.tolist() == [False, False, True]
    assert cards.mana_value.tolist() == [2.0, 4.0, 0.0]
    assert [cards.pool[card].rarity for card in range(3)] == [Rarity.RARE, Rarity.COMMON, Rarity.BASIC]
    assert [cards.pool[card].program for card in range(3)] == [(7, 7), (), ()]
    assert cards.ata.tolist() == [2.0, 6.0, 10.0]


def test_program_tokens_mark_a_pair_the_vocabulary_lacks_and_a_card_without_a_program() -> None:
    programs = {"Ant": (1, 5, 2, 9), "Bat": (1, 5, 3, 4, 2, 9), "Dog": ()}
    records = [
        CardRecord(name=name, set_code="TST", rarity=Rarity.COMMON, text="", program=program)
        for name, program in programs.items()
    ]

    ant, bat, dog = program_tokens(CardPool(records), vocabulary={(1, 5): 0, (2, 9): 1}).tolist()

    assert ant == [FIRST_PROGRAM_TOKEN, FIRST_PROGRAM_TOKEN + 1, PADDING_TOKEN]
    assert bat == [FIRST_PROGRAM_TOKEN, UNKNOWN_TOKEN, FIRST_PROGRAM_TOKEN + 1]
    assert dog == [NO_PROGRAM_TOKEN, PADDING_TOKEN, PADDING_TOKEN]

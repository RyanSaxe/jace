"""Check boosters, whole drafts and the drafter model through `jace.draft`.

The tests run small real drafts with real drafters and a tiny real model.
They use only the names `jace.draft` exports, so the code behind those names
can be rearranged without touching them.
"""

import random
from collections.abc import Sequence

import pytest
import torch

from jace.cards import CardEmbedding, CardId, CardPool, CardRecord, Rarity
from jace.draft import (
    TLA_PLAY_BOOSTER,
    BoosterSpecError,
    DrafterModel,
    DraftFormat,
    SeatView,
    model_drafter,
    open_booster,
    run_drafts,
)

__all__ = ()


def _pool(counts: dict[Rarity, int]) -> CardPool:
    records = [
        CardRecord(name=f"{rarity.value} {number}", set_code="TST", rarity=rarity, text="", program=())
        for rarity, count in counts.items()
        for number in range(count)
    ]
    return CardPool(records)


def _first_card(views: Sequence[SeatView]) -> tuple[CardId, ...]:
    return tuple(view.pack[0] for view in views)


def _model(cards: int) -> DrafterModel:
    embedding = CardEmbedding(torch.randn(cards, 8), hidden=8, offset_bound=None)
    return DrafterModel(embedding, width=16, layers=1, heads=2, picks=4)


def test_a_drafter_filling_several_seats_gets_all_of_them_across_pods_in_one_call_per_pick() -> None:
    draft_format = DraftFormat(seats=3, rounds=1, picks_per_round=2)
    boosters = [
        [[(1, 2), (3, 4), (5, 6)]],
        [[(7, 8), (9, 10), (11, 12)]],
    ]
    calls: list[tuple[CardId, ...]] = []

    def two_seats(views: Sequence[SeatView]) -> tuple[CardId, ...]:
        calls.append(tuple(view.pack[0] for view in views))
        return _first_card(views)

    records = run_drafts(draft_format, boosters, drafters=[two_seats, _first_card, two_seats])

    assert calls == [(1, 5, 7, 11), (6, 4, 12, 10)]
    assert [record.pools for record in records] == [((1, 6), (3, 2), (5, 4)), ((7, 12), (9, 8), (11, 10))]


def test_a_seats_view_holds_each_pack_as_it_reached_the_seat() -> None:
    draft_format = DraftFormat(seats=3, rounds=1, picks_per_round=3)
    boosters = [[[(1, 2, 3), (4, 5, 6), (7, 8, 9)]]]

    (record,) = run_drafts(draft_format, boosters, drafters=[_first_card] * 3)

    assert record.views[0][2] == SeatView(pick_number=2, pack=(6,), seen=((1, 2, 3), (8, 9)), picks=(1, 8))


def test_every_slot_of_a_booster_holds_a_distinct_card_of_a_rarity_its_odds_allow() -> None:
    pool = _pool({Rarity.BASIC: 5, Rarity.COMMON: 10, Rarity.UNCOMMON: 5, Rarity.RARE: 3, Rarity.MYTHIC: 3})
    rng = random.Random(0)

    for _ in range(200):
        booster = open_booster(TLA_PLAY_BOOSTER, pool, rng=rng)

        assert len(set(booster)) == len(TLA_PLAY_BOOSTER)
        for card, odds in zip(booster, TLA_PLAY_BOOSTER, strict=True):
            assert odds[pool[card].rarity] > 0


def test_a_spec_with_a_slot_for_a_rarity_the_pool_lacks_is_rejected() -> None:
    pool = _pool({Rarity.COMMON: 4})

    with pytest.raises(BoosterSpecError, match="1 slots can draw a rare card, but the pool holds 0"):
        open_booster([{Rarity.COMMON: 0.9, Rarity.RARE: 0.1}], pool, rng=random.Random(0))


def test_a_model_drafter_finishes_a_draft_over_a_larger_card_pool_than_the_model_was_built_with() -> None:
    model = _model(cards=4)
    model.embedding = CardEmbedding(torch.randn(30, 8), hidden=8, offset_bound=None)
    draft_format = DraftFormat(seats=2, rounds=2, picks_per_round=2)
    boosters = [
        [[(4, 5, 6), (7, 8, 9)], [(10, 11, 12), (13, 14, 15)]],
        [[(16, 17, 18), (19, 20, 21)], [(22, 23, 24), (25, 26, 27)]],
    ]
    sampling = model_drafter(model, generator=torch.Generator().manual_seed(0))
    greedy = model_drafter(model, generator=None)

    # A pick outside the seat's pack would raise `IllegalPickError` here.
    records = run_drafts(draft_format, boosters, drafters=[sampling, greedy])

    assert [[len(pool) for pool in record.pools] for record in records] == [[4, 4], [4, 4]]


def test_padding_a_shorter_draft_history_leaves_its_logits_unchanged() -> None:
    model = _model(cards=6)
    seen = torch.tensor([[[1.0, 1.0, 1.0, 0.0, 0.0, 0.0]]])
    picks = torch.tensor([[2]])
    pack = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 0.0]])
    longer_seen = torch.tensor([[[0.0, 1.0, 1.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 1.0, 1.0]]])
    padded_seen = torch.cat([seen, torch.zeros(1, 1, 6)], dim=1)

    alone = model.forward(seen, picks, pack=pack)
    batched = model.forward(
        torch.cat([padded_seen, longer_seen]), torch.tensor([[2, 0], [1, 5]]), pack=torch.cat([pack, pack])
    )

    assert torch.allclose(batched[0], alone[0], atol=1e-5)
    assert torch.isinf(alone[0]).tolist() == [True, True, True, False, False, True]

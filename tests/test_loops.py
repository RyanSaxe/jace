"""Run the limited loop end to end on the toy engine with tiny models.

One test drives a whole iteration through `limited_iteration`, the way a
user would, and checks that every model was updated. It proves the pieces
fit together; it says nothing about how well anything learns, which the
tests of each piece cover.
"""

import random

import torch
from torch import Tensor, nn

from jace.cards import CardEmbedding, CardPool, CardRecord, Rarity
from jace.decks import BuildRules, DeckBuilderModel, DeckScore
from jace.draft import DrafterModel, DraftFormat
from jace.engine.toy import HAND_SIZE, TOY_CARDS, ToyBatch
from jace.loops import Learners, LimitedSetup, limited_iteration, phased_schedule
from jace.modules import run
from jace.play import Boards, Earlier, Guess
from jace.train import PlayerRule

__all__ = ()


def test_one_limited_iteration_updates_every_model() -> None:
    torch.manual_seed(0)
    records = [CardRecord(name="Island", set_code="toy", rarity=Rarity.BASIC, text="", program=())]
    records += [
        CardRecord(name=f"Card {card}", set_code="toy", rarity=Rarity.COMMON, text="", program=())
        for card in range(1, TOY_CARDS)
    ]
    pool = CardPool(records)
    embedding = CardEmbedding(torch.randn(TOY_CARDS, 8), hidden=16, offset_bound=None)
    player = nn.Sequential(nn.Linear(HAND_SIZE + 1, 16), nn.Tanh(), nn.Linear(16, HAND_SIZE + 1))
    guess = nn.Sequential(nn.Linear(HAND_SIZE + 1, 16), nn.Tanh(), nn.Linear(16, 2 * TOY_CARDS))
    drafter = DrafterModel(embedding, width=16, layers=1, heads=2, picks=3)
    builder = DeckBuilderModel(embedding, width=16, layers=1, heads=2)
    score = DeckScore(embedding, width=16, layers=1, heads=2)

    def policy(boards: Boards, _guess: Guess) -> tuple[Tensor, Tensor]:
        output = run(player, torch.cat([boards["hand"], boards["seen"]], dim=-1))
        return output[:, :HAND_SIZE], output[:, HAND_SIZE].sigmoid()

    def guesser(boards: Boards, _earlier: Earlier) -> Guess:
        output = run(guess, torch.cat([boards["hand"], boards["seen"]], dim=-1))
        return Guess(deck=output[:, :TOY_CARDS].softmax(dim=-1), hand=output[:, TOY_CARDS:].sigmoid())

    def own(module: nn.Module) -> list[nn.Parameter]:
        return [parameter for name, parameter in module.named_parameters() if "embedding." not in name]

    trained = {
        "player": player,
        "guess": guess,
        "drafter": drafter,
        "deck_builder": builder,
        "deck_score": score,
    }
    optimizers = {name: torch.optim.Adam(own(module), lr=1e-2) for name, module in trained.items()}
    optimizers["embedding"] = torch.optim.Adam(embedding.parameters(), lr=1e-2)
    before = {name: [parameter.detach().clone() for parameter in own(module)] for name, module in trained.items()}
    setup = LimitedSetup(
        pool=pool,
        draft_format=DraftFormat(seats=2, rounds=1, picks_per_round=3),
        booster=[{Rarity.COMMON: 1.0}] * 4,
        rules=BuildRules(size=5, basics=(0,), nonbasic_lands=frozenset(), min_lands=2, max_lands=3),
        pods=4,
        proposals=8,
        search_rounds=2,
        score_chunk=256,
        decisions=64,
        history=2,
        slice_rows=64,
        max_grad_norm=1.0,
        player_rule=PlayerRule(clip=0.2, stay=0.1, spread=0.0, keep_quantile=0.0, keep_floor=0.0, value_weight=0.5),
        lam_advantage=0.5,
        lam_value=0.8,
    )
    learners = Learners(
        policy=policy,
        guesser=guesser,
        drafter=drafter,
        human_drafter=None,
        builder=builder,
        score=score,
        optimizers=optimizers,
    )

    metrics = limited_iteration(
        setup,
        learners,
        dials=phased_schedule(drafter_starts=0, pull_ends=10)(5),
        opponent=policy,
        other_seats=[],
        start=lambda matchups: ToyBatch(matchups, games=8, seed=0),
        rng=random.Random(0),
    )

    assert metrics["games"] > 0
    for name, module in trained.items():
        moved = any(not torch.equal(old, new) for old, new in zip(before[name], own(module), strict=True))
        assert moved, f"{name} was not updated"

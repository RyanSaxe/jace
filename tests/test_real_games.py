"""Train the real models for one step on real Magic games.

This is the one test that joins the manabot adapter, the play models and the
training pieces. It catches a mismatch at any seam between them, such as a
board the models cannot read or labels shaped for another card pool. It does
not check that anything learns.
"""

import random

import torch
from torch import nn

from jace.cards import CardEmbedding, CardPool, CardRecord, Deck, Rarity
from jace.engine.manabot import engine_card_ids, engine_cards, start_batch
from jace.play import GuessModel, PlayerModel, as_boards
from jace.train import PlayerRule, collect, update_guess, update_player

__all__ = ()


def test_real_games_train_the_player_and_the_guess_model_for_one_step() -> None:
    torch.manual_seed(0)
    cards = engine_cards()
    pool = CardPool(
        [
            CardRecord(
                name=card.name, set_code=card.set_code, rarity=Rarity.COMMON, text=card.text, program=card.program
            )
            for card in cards
        ]
    )
    basics = [pool.card_id(name) for name in ("Island", "Mountain")]
    spells = [card for card, record in enumerate(cards) if record.set_code == "TLA" and card not in basics]
    rng = random.Random(0)

    def matchup() -> tuple[Deck, Deck]:
        first = (*([basics[0]] * 17), *rng.sample(spells, 23))
        second = (*([basics[1]] * 17), *rng.sample(spells, 23))
        return first, second

    batch = start_batch(matchup, pool=pool, games=4, seed=0)
    sample = as_boards(batch.decisions(), device=torch.device("cpu"))
    embedding = CardEmbedding(torch.randn(len(pool), 16), hidden=16, offset_bound=None)
    engine_to_card = engine_card_ids(pool)
    player = PlayerModel(
        embedding, engine_to_card=engine_to_card, sample=sample, width=32, layers=1, heads=2, guess_cards=4
    )
    guess = GuessModel(embedding, engine_to_card=engine_to_card, sample=sample, width=32, layers=1, heads=2)

    def own(module: nn.Module) -> list[nn.Parameter]:
        return [parameter for name, parameter in module.named_parameters() if not name.startswith("embedding.")]

    before = [parameter.detach().clone() for parameter in (*own(player), *own(guess))]
    embedding_optimizer = torch.optim.Adam(embedding.parameters(), lr=1e-3)
    collected = collect(
        batch,
        None,
        learner=player.policy,
        opponent=player.policy,
        guesser=guess.guess,
        cards=len(pool),
        history=2,
        decisions=40,
        generator=torch.Generator().manual_seed(0),
    )
    rollout = collected.rollout
    update_player(
        player.policy,
        rollout,
        rule=PlayerRule(clip=0.2, stay=0.1, spread=0.05, keep_quantile=0.0, keep_floor=0.0, value_weight=1.0),
        lam_advantage=0.5,
        lam_value=0.8,
        slice_rows=64,
        max_grad_norm=1.0,
        optimizers=[torch.optim.Adam(own(player), lr=1e-3), embedding_optimizer],
    )
    update_guess(
        guess.guess,
        rollout,
        history=2,
        slice_rows=64,
        max_grad_norm=1.0,
        optimizers=[torch.optim.Adam(own(guess), lr=1e-3), embedding_optimizer],
    )

    assert len(collected.games) > 0
    assert bool((rollout.hand_truth.sum(dim=-1) > 0).any())
    after = [*own(player), *own(guess)]
    assert any(not torch.equal(old, new) for old, new in zip(before, after, strict=True))

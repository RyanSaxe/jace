"""Check pretraining on human picks through `jace.loops.pretrain` and the drafter.

The drafts here are made up: every pack is random and the "human" always
takes the pack's lowest card id, a rule a tiny model can learn in a few
seconds on the CPU. The tests use a real run folder in a temporary
directory, a real model and real optimizers.
"""

import functools
from datetime import UTC, datetime
from pathlib import Path

import torch

from jace.cards import CardEmbedding
from jace.data import HumanDrafts
from jace.draft import DrafterModel
from jace.loops.pretrain import PretrainSettings, pick_accuracy, pretrain_drafter
from jace.tracking import Identity, create_run, writing

__all__ = ()

_CARDS = 10
_ROUNDS = 2
_PICKS_PER_ROUND = 3


def _lowest_card_drafts(drafts: int, *, generator: torch.Generator) -> HumanDrafts:
    steps = _ROUNDS * _PICKS_PER_ROUND
    packs = torch.full((drafts, steps, _PICKS_PER_ROUND), -1, dtype=torch.int16)
    for draft in range(drafts):
        for step in range(steps):
            size = _PICKS_PER_ROUND - step % _PICKS_PER_ROUND
            packs[draft, step, :size] = torch.randperm(_CARDS, generator=generator)[:size].to(torch.int16)
    lowest = torch.where(packs < 0, _CARDS, packs).min(dim=-1).values
    held_out = torch.arange(drafts) >= drafts * 3 // 4
    return HumanDrafts(packs=packs, picks=lowest, weights=torch.ones(drafts, steps), held_out=held_out)


def _model() -> DrafterModel:
    embedding = CardEmbedding(torch.randn(_CARDS, 8), hidden=8, offset_bound=None)
    return DrafterModel(embedding, width=16, layers=2, heads=2, picks=_ROUNDS * _PICKS_PER_ROUND)


def test_scoring_a_whole_draft_gives_each_pick_the_logits_it_gets_from_its_own_history() -> None:
    torch.manual_seed(0)
    model = _model()
    seen = (torch.rand(3, 6, _CARDS) < 0.4).float()
    seen[:, :, 0] = 1.0
    picks = torch.randint(_CARDS, (3, 6))

    whole = model.whole_draft(seen, picks)

    for step in range(6):
        alone = model.forward(seen[:, :step], picks[:, :step], pack=seen[:, step])
        in_pack = alone.isfinite()
        assert torch.equal(in_pack, whole[:, step].isfinite())
        assert torch.allclose(whole[:, step][in_pack], alone[in_pack], atol=1e-5)


def test_pretraining_on_picks_that_follow_a_rule_learns_the_rule_on_held_out_drafts(tmp_path: Path) -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    drafts = _lowest_card_drafts(64, generator=generator)
    model = _model()
    own = [parameter for name, parameter in model.named_parameters() if not name.startswith("embedding.")]
    optimizers = [torch.optim.Adam(own, lr=1e-2), torch.optim.Adam(model.embedding.parameters(), lr=1e-2)]
    settings = PretrainSettings(steps=60, drafts_per_step=16, evaluate_every=20, evaluate_drafts=16, max_grad_norm=1.0)
    identity = Identity(config={}, seed=0, jace_commit="test", manabot_commit="test", inputs=())
    clock = functools.partial(datetime, 2026, 1, 1, tzinfo=UTC)
    folder = create_run(tmp_path, identity, clock=clock)
    held_out = drafts.held_out.nonzero().squeeze(1)
    untrained = pick_accuracy(
        model.whole_draft, drafts, rows=held_out, cards=_CARDS, device=torch.device("cpu"), settings=settings
    )

    with writing(folder, clock=clock) as run:
        curve = pretrain_drafter(
            model,
            drafts,
            optimizers=optimizers,
            rate_scale=lambda _: 1.0,
            settings=settings,
            run=run,
            generator=generator,
        )

    assert [point.step for point in curve] == [20, 40, 60]
    assert untrained.top1 < 0.8
    assert curve[-1].top1 > 0.95

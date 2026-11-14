"""Check pretraining on human decks through `jace.loops.pretrain_decks`.

The decks here are made up. Every pool is six random spells out of ten, the
"human" plays the four with the lowest ids plus two copies of one basic, and
a deck wins most of its games only when it holds the lowest spell of all.
Tiny models learn both rules in a few seconds on the CPU. The tests use a
real run folder in a temporary directory, real models and real optimizers.
"""

import functools
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor

from jace.cards import CardTable
from jace.data import HumanDecks
from jace.decks import BuildRules, DeckBuilder, DeckScore, sample_builds
from jace.loops.pretrain_decks import (
    DeckSettings,
    deck_overlap,
    pretrain_deck_builder,
    pretrain_deck_score,
    score_fit,
)
from jace.tracking import Identity, create_run, writing

__all__ = ()

_CARDS = 12
_BASICS = (0, 1)
_BEST_SPELL = 2
_POOL_SPELLS = 6
_DECK_SPELLS = 4
_GAMES = 7
_RULES = BuildRules(size=6, basics=_BASICS, nonbasic_lands=frozenset(), min_lands=2, max_lands=2)
_CPU = torch.device("cpu")


def _lowest_spell_decks(drafts: int, *, generator: torch.Generator) -> HumanDecks:
    decks = torch.zeros((drafts, _CARDS), dtype=torch.uint8)
    pools = torch.zeros((drafts, _CARDS), dtype=torch.uint8)
    for draft in range(drafts):
        spells = (torch.randperm(_CARDS - len(_BASICS), generator=generator)[:_POOL_SPELLS] + len(_BASICS)).sort()
        pools[draft, spells.values] = 1
        decks[draft, spells.values[:_DECK_SPELLS]] = 1
    decks[:, _BASICS[0]] = 2
    pools[:, _BASICS[0]] = 2
    win_chance = torch.where(decks[:, _BEST_SPELL] > 0, 0.85, 0.25)
    wins = torch.binomial(torch.full((drafts,), float(_GAMES)), win_chance, generator=generator).to(torch.int16)
    held_out = torch.arange(drafts) >= drafts * 3 // 4
    return HumanDecks(decks=decks, pools=pools, wins=wins, losses=_GAMES - wins, held_out=held_out)


def test_pretraining_the_score_on_decks_that_win_with_one_card_beats_the_overall_win_rate(tmp_path: Path) -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    decks = _lowest_spell_decks(256, generator=generator)
    model = DeckScore(CardTable(torch.randn(_CARDS, 8)), width=16, layers=1, heads=2)
    settings = DeckSettings(steps=150, decks_per_step=32, evaluate_every=50, evaluate_decks=64, max_grad_norm=1.0)
    clock = functools.partial(datetime, 2026, 1, 1, tzinfo=UTC)
    identity = Identity(config={}, seed=0, jace_commit="test", manabot_commit="test", inputs=())
    folder = create_run(tmp_path, identity, clock=clock)
    held_out = decks.held_out.nonzero().squeeze(1)
    overall_rate = decks.wins[~decks.held_out].sum() / (_GAMES * (~decks.held_out).sum())
    overall = score_fit(lambda counts: overall_rate.logit().expand(len(counts)), decks, rows=held_out, device=_CPU)

    with writing(folder, clock=clock) as run:
        curve = pretrain_deck_score(
            model,
            decks,
            optimizers=[torch.optim.Adam(model.parameters(), lr=1e-2)],
            rate_scale=lambda _: 1.0,
            settings=settings,
            run=run,
            generator=generator,
        )

    assert [point.step for point in curve] == [50, 100, 150]
    assert curve[-1].log_loss < overall.log_loss - 0.1
    assert curve[-1].correlation > 0.7


def test_pretraining_the_builder_on_decks_that_follow_a_rule_rebuilds_held_out_decks(tmp_path: Path) -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    decks = _lowest_spell_decks(256, generator=generator)
    # The vectors are wider than there are cards, so a linear map of them can put the cards in any order.
    model = DeckBuilder(CardTable(torch.randn(_CARDS, 16)), width=16, layers=1, heads=2)
    settings = DeckSettings(steps=200, decks_per_step=16, evaluate_every=100, evaluate_decks=64, max_grad_norm=1.0)
    clock = functools.partial(datetime, 2026, 1, 1, tzinfo=UTC)
    identity = Identity(config={}, seed=0, jace_commit="test", manabot_commit="test", inputs=())
    folder = create_run(tmp_path, identity, clock=clock)
    held_out = decks.held_out.nonzero().squeeze(1)

    def likeliest(pools: Tensor) -> Tensor:
        builds, _ = sample_builds(
            model, pools, rules=_RULES, spells_per_round=2, temperature=1.0, generator=generator, greedy=True
        )
        return builds.decks

    untrained = deck_overlap(likeliest, decks, rows=held_out, rules=_RULES, device=_CPU)
    with writing(folder, clock=clock) as run:
        curve = pretrain_deck_builder(
            model,
            decks,
            rules=_RULES,
            spells_per_round=2,
            optimizers=[torch.optim.Adam(model.parameters(), lr=1e-2)],
            rate_scale=lambda _: 1.0,
            settings=settings,
            run=run,
            generator=generator,
        )

    assert [point.step for point in curve] == [100, 200]
    assert untrained < 0.8
    assert curve[-1].overlap > 0.95

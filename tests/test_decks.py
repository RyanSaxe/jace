"""Check deck building through the public interface of `jace.decks`.

The tests use a twelve-card set with two basics, one non-basic land, a red
group, a blue group and one colourless spell. Legality is judged by a helper
written here from the rules' wording, not by the builder's own code, so a
mistake in the builder's masks cannot hide itself. Models are tiny and real.
"""

import pytest
import torch
from torch import Tensor

from jace.cards import CardEmbedding, Pool
from jace.decks import (
    BuildRules,
    DeckBuilderModel,
    DeckScore,
    NoLegalDeckError,
    build_decks,
    improve,
    model_proposer,
    random_proposer,
)
from jace.modules import run

__all__ = ()

CARDS = 12
RED_BASIC, BLUE_BASIC, NONBASIC_LAND, COLOURLESS = 0, 1, 2, 11
RED_SPELLS: Pool = (3, 3, 4, 4, 5, 6)
BLUE_SPELLS: Pool = (7, 7, 8, 8, 9, 10)
TWO_BASICS = BuildRules(
    size=10, basics=(RED_BASIC, BLUE_BASIC), nonbasic_lands=frozenset({NONBASIC_LAND}), min_lands=3, max_lands=5
)
NO_BASICS = BuildRules(size=10, basics=(), nonbasic_lands=frozenset({NONBASIC_LAND}), min_lands=3, max_lands=5)


def _embedding(cards: int) -> CardEmbedding:
    starting = torch.randn((cards, 8), generator=torch.Generator().manual_seed(cards))
    return CardEmbedding(starting, hidden=8, offset_bound=None)


def _counts(pool: Pool) -> Tensor:
    counts = torch.zeros(CARDS)
    for card in pool:
        counts[card] += 1
    return counts


def _legal(decks: Tensor, *, pool: Pool, rules: BuildRules) -> Tensor:
    # decks: float [n, cards]. Returns bool [n].
    limit = _counts(pool)
    limit[list(rules.basics)] = rules.size
    lands = decks[:, [*rules.basics, *rules.nonbasic_lands]].sum(dim=1)
    return (
        (decks.sum(dim=1) == rules.size)
        & (decks >= 0).all(dim=1)
        & (decks <= limit).all(dim=1)
        & (lands >= rules.min_lands)
        & (lands <= rules.max_lands)
    )


def _colour_scorer(decks: Tensor) -> Tensor:
    # Rewards a deck for being one colour, blue above red, with exactly four lands.
    red = decks[:, [RED_BASIC, *set(RED_SPELLS)]].sum(dim=1)
    blue = decks[:, [BLUE_BASIC, *set(BLUE_SPELLS)]].sum(dim=1)
    lands = decks[:, [RED_BASIC, BLUE_BASIC, NONBASIC_LAND]].sum(dim=1)
    return red**2 + 1.5 * blue**2 - (lands - 4) ** 2


@pytest.mark.parametrize(
    ("rules", "pool"),
    [
        pytest.param(TWO_BASICS, (*RED_SPELLS, *BLUE_SPELLS, NONBASIC_LAND, COLOURLESS), id="roomy pool"),
        pytest.param(TWO_BASICS, (3, 3, 4, 7, 8, NONBASIC_LAND), id="five spells force five lands"),
        pytest.param(NO_BASICS, (*RED_SPELLS, 7, *[NONBASIC_LAND] * 4), id="no basics and four lands"),
    ],
)
def test_proposed_and_improved_decks_are_legal(rules: BuildRules, pool: Pool) -> None:
    generator = torch.Generator().manual_seed(0)
    embedding = _embedding(CARDS)
    builder = DeckBuilderModel(embedding, width=8, layers=1, heads=2)
    scorer = DeckScore(embedding, width=8, layers=1, heads=2)

    for propose in (
        random_proposer(rules=rules, generator=generator),
        model_proposer(builder, rules=rules, generator=generator),
    ):
        proposed = propose(_counts(pool), 24)
        improved = improve(proposed, pool_counts=_counts(pool), score=scorer, rules=rules, rounds=3)

        assert _legal(proposed, pool=pool, rules=rules).all()
        assert _legal(improved, pool=pool, rules=rules).all()
        assert (run(scorer, improved) >= run(scorer, proposed) - 1e-5).all()


def test_build_decks_finds_the_best_deck_that_swaps_from_one_start_cannot_reach() -> None:
    pool = (*RED_SPELLS, *BLUE_SPELLS, NONBASIC_LAND, COLOURLESS)
    all_red = (*[RED_BASIC] * 4, *RED_SPELLS)
    all_blue = (*[BLUE_BASIC] * 4, *BLUE_SPELLS)

    stuck = improve(
        _counts(all_red).unsqueeze(0), pool_counts=_counts(pool), score=_colour_scorer, rules=TWO_BASICS, rounds=20
    )
    candidates = build_decks(
        pool,
        cards=CARDS,
        propose=random_proposer(rules=TWO_BASICS, generator=torch.Generator().manual_seed(0)),
        score=_colour_scorer,
        rules=TWO_BASICS,
        proposals=32,
        rounds=20,
    )

    assert torch.equal(stuck, _counts(all_red).unsqueeze(0))
    assert candidates[0].deck == all_blue
    assert candidates[0].score == 150.0
    scores = [candidate.score for candidate in candidates]
    assert scores == sorted(scores, reverse=True)
    assert len({candidate.deck for candidate in candidates}) == len(candidates)


def test_models_give_the_same_answers_when_the_card_pool_grows() -> None:
    small_embedding = _embedding(6)
    extra = torch.randn((9, 8), generator=torch.Generator().manual_seed(1))
    large_embedding = CardEmbedding(torch.cat((small_embedding.starting, extra)), hidden=8, offset_bound=None)
    deck = torch.tensor([[2.0, 0.0, 1.0, 0.0, 3.0, 0.0]])
    available = torch.tensor([[1.0, 2.0, 0.0, 1.0, 4.0, 0.0]])
    padding = torch.zeros((1, 9))

    for small, large in (
        (
            DeckScore(small_embedding, width=8, layers=1, heads=2),
            DeckScore(large_embedding, width=8, layers=1, heads=2),
        ),
        (
            DeckBuilderModel(small_embedding, width=8, layers=1, heads=2),
            DeckBuilderModel(large_embedding, width=8, layers=1, heads=2),
        ),
    ):
        with torch.no_grad():
            for grown, original in zip(large.parameters(), small.parameters(), strict=True):
                grown.copy_(original)
            if isinstance(small, DeckScore):
                before = run(small, deck)
                after = run(large, torch.cat((deck, padding), dim=1))
            else:
                before = run(small, available, deck)
                after = run(large, torch.cat((available, padding), dim=1), torch.cat((deck, padding), dim=1))[:, :6]

        assert torch.allclose(before, after, atol=1e-5)


def test_a_pool_with_too_few_spells_for_any_legal_deck_is_rejected() -> None:
    propose = random_proposer(rules=TWO_BASICS, generator=torch.Generator().manual_seed(0))

    with pytest.raises(NoLegalDeckError, match="at least 8 lands and can hold at most 5"):
        build_decks((3, 7), cards=CARDS, propose=propose, score=_colour_scorer, rules=TWO_BASICS, proposals=4, rounds=1)

"""Check deck building through the public interface of `jace.decks`.

Most tests use a set of twelve cards. Cards 0 and 1 are the red and blue
basics, card 2 is a nonbasic land, cards 3 to 6 are red spells, cards 7 to 10
are blue spells, and card 11 is a colourless spell. The test of a growing card
pool starts from six cards of its own. A helper written here from the rules'
wording checks which decks are legal. It shares no code with the builder's
masks, so a mistake in those masks cannot also be in the check. The models are
tiny and real.
"""

import pytest
import torch
from torch import Tensor

from jace.cards import FIRST_PROGRAM_TOKEN, NO_PROGRAM_TOKEN, CardEncoder, CardInputs, CardTable, Pool, own_parameters
from jace.decks import BuildRules, DeckBuilder, DeckScore, build_decks, build_log_probs, sample_builds

__all__ = ()

CARDS = 12
RED_BASIC, BLUE_BASIC, NONBASIC_LAND, COLOURLESS = 0, 1, 2, 11
RED_SPELLS: Pool = (3, 3, 4, 4, 5, 6)
BLUE_SPELLS: Pool = (7, 7, 8, 8, 9, 10)
ROOMY_POOL: Pool = (*RED_SPELLS, *BLUE_SPELLS, NONBASIC_LAND, COLOURLESS)
TWO_BASICS = BuildRules(
    size=10, basics=(RED_BASIC, BLUE_BASIC), nonbasic_lands=frozenset({NONBASIC_LAND}), min_lands=3, max_lands=5
)


def _embedding(cards: int) -> CardTable:
    vectors = torch.randn((cards, 8), generator=torch.Generator().manual_seed(cards))
    return CardTable(vectors)


def _counts(pool: Pool) -> Tensor:
    counts = torch.zeros(CARDS)
    for card in pool:
        counts[card] += 1
    return counts


def _legal(decks: Tensor, *, pool: Pool, rules: BuildRules) -> Tensor:
    # `decks` is float [n, cards], each deck's copies of each card. The result is bool [n], true for each
    # legal deck.
    limit = _counts(pool)
    limit[list(rules.basics)] = rules.size
    lands = decks[:, [*rules.basics, *rules.nonbasic_lands]].sum(dim=1)
    spells_in_pool = sum(1 for card in pool if card not in {*rules.basics, *rules.nonbasic_lands})
    # When a pool has too few spells to fill a deck within the land range, a legal deck contains every spell in
    # the pool, and lands in the other slots.
    fewest = max(rules.min_lands, rules.size - spells_in_pool)
    most = max(rules.max_lands, rules.size - spells_in_pool)
    return (
        (decks.sum(dim=1) == rules.size)
        & (decks >= 0).all(dim=1)
        & (decks <= limit).all(dim=1)
        & (lands >= fewest)
        & (lands <= most)
    )


def _colour_scorer(decks: Tensor) -> Tensor:
    # Rewards a deck for staying in one colour, blue more than red, and for having exactly four lands.
    red = decks[:, [RED_BASIC, *set(RED_SPELLS)]].sum(dim=1)
    blue = decks[:, [BLUE_BASIC, *set(BLUE_SPELLS)]].sum(dim=1)
    lands = decks[:, [RED_BASIC, BLUE_BASIC, NONBASIC_LAND]].sum(dim=1)
    return red**2 + 1.5 * blue**2 - (lands - 4) ** 2


@pytest.mark.parametrize(
    ("pool", "spells_per_round"),
    [
        pytest.param(ROOMY_POOL, 3, id="roomy pool"),
        pytest.param((3, 3, 4, 7, 8, NONBASIC_LAND), 2, id="five spells force five lands"),
        pytest.param((3, 7, NONBASIC_LAND, NONBASIC_LAND), 4, id="two spells force eight lands"),
        pytest.param((*RED_SPELLS, 7, *[NONBASIC_LAND] * 6), 1, id="more nonbasic lands than land slots"),
    ],
)
def test_every_sampled_deck_is_legal_for_its_pool(pool: Pool, spells_per_round: int) -> None:
    builder = DeckBuilder(_embedding(CARDS), width=8, layers=1, heads=2)

    for temperature, greedy in ((1.0, False), (3.0, False), (1.0, True)):
        builds, _ = sample_builds(
            builder,
            _counts(pool).expand(64, -1),
            rules=TWO_BASICS,
            spells_per_round=spells_per_round,
            temperature=temperature,
            generator=torch.Generator().manual_seed(0),
            greedy=greedy,
        )

        assert _legal(builds.decks, pool=pool, rules=TWO_BASICS).all()


def test_a_pool_whose_lands_are_all_basics_still_gets_its_land_total() -> None:
    builder = DeckBuilder(_embedding(CARDS), width=8, layers=1, heads=2)

    builds, _ = sample_builds(
        builder,
        _counts((*RED_SPELLS, *BLUE_SPELLS)).expand(64, -1),
        rules=TWO_BASICS,
        spells_per_round=3,
        temperature=1.0,
        generator=torch.Generator().manual_seed(0),
    )

    basics = builds.decks[:, [RED_BASIC, BLUE_BASIC]].sum(dim=1)
    assert torch.equal(basics.long(), builds.lands)
    assert set(builds.lands.tolist()) == {3, 4, 5}


def test_a_sampled_decks_log_probability_is_the_one_its_build_is_scored_with() -> None:
    builder = DeckBuilder(_embedding(CARDS), width=8, layers=1, heads=2)

    builds, sampled = sample_builds(
        builder,
        _counts(ROOMY_POOL).expand(32, -1),
        rules=TWO_BASICS,
        spells_per_round=3,
        temperature=1.5,
        generator=torch.Generator().manual_seed(0),
    )
    scored, entropies = build_log_probs(builder, builds, rules=TWO_BASICS, spells_per_round=3, temperature=1.5)
    scored.sum().backward()

    assert torch.allclose(sampled, scored.detach(), atol=1e-4)
    assert (sampled < 0).all()
    assert (entropies > 0).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in builder.parameters() if parameter.grad is not None)


def test_the_search_returns_distinct_decks_best_first() -> None:
    builder = DeckBuilder(_embedding(CARDS), width=8, layers=1, heads=2)

    builds, scores = build_decks(
        ROOMY_POOL,
        builder=builder,
        score=_colour_scorer,
        rules=TWO_BASICS,
        cards=CARDS,
        samples=128,
        keep=6,
        spells_per_round=3,
        temperature=1.0,
        generator=torch.Generator().manual_seed(0),
    )

    assert len(scores) == 6
    assert torch.equal(scores, _colour_scorer(builds.decks))
    assert scores.tolist() == sorted(scores.tolist(), reverse=True)
    assert len(builds.decks.unique(dim=0)) == 6
    assert _legal(builds.decks, pool=ROOMY_POOL, rules=TWO_BASICS).all()


def test_models_give_the_same_answers_when_the_card_pool_grows() -> None:
    small_embedding = _embedding(6)
    extra = torch.randn((9, 8), generator=torch.Generator().manual_seed(1))
    large_embedding = CardTable(torch.cat((small_embedding.table, extra)))
    rules = BuildRules(size=8, basics=(0,), nonbasic_lands=frozenset({2}), min_lands=3, max_lands=4)
    deck = torch.tensor([[2.0, 0.0, 1.0, 0.0, 3.0, 0.0]])
    pool = torch.tensor([[1.0, 2.0, 1.0, 1.0, 4.0, 1.0]])
    padding = torch.zeros((1, 9))
    small_builder = DeckBuilder(small_embedding, width=8, layers=1, heads=2)
    large_builder = DeckBuilder(large_embedding, width=8, layers=1, heads=2)
    small_score = DeckScore(small_embedding, width=8, layers=1, heads=2)
    large_score = DeckScore(large_embedding, width=8, layers=1, heads=2)

    with torch.no_grad():
        for small, large in ((small_builder, large_builder), (small_score, large_score)):
            for grown, original in zip(large.parameters(), small.parameters(), strict=True):
                grown.copy_(original)
        before = small_builder(pool, deck * (pool > 0), rules=rules)
        after = large_builder(
            torch.cat((pool, padding), dim=1), torch.cat((deck * (pool > 0), padding), dim=1), rules=rules
        )

        assert torch.allclose(small_score(deck), large_score(torch.cat((deck, padding), dim=1)), atol=1e-5)
        assert torch.allclose(before.cards, after.cards[:, :6], atol=1e-5)
        assert torch.allclose(before.lands, after.lands, atol=1e-5)
        assert torch.allclose(before.basics, after.basics, atol=1e-5)


def test_a_deck_models_own_parameters_hold_none_of_the_shared_embeddings() -> None:
    generator = torch.Generator().manual_seed(0)
    inputs = CardInputs(
        text=torch.randn((CARDS, 4), generator=generator),
        program=torch.full((CARDS, 1), NO_PROGRAM_TOKEN),
        printed=torch.randn((CARDS, 3), generator=generator),
    )
    embedding = CardEncoder(inputs, width=8, program_vocabulary=FIRST_PROGRAM_TOKEN, program_layers=1, heads=2)
    shared = {id(parameter) for parameter in embedding.parameters()}

    for model in (
        DeckScore(embedding, width=16, layers=1, heads=2),
        DeckBuilder(embedding, width=16, layers=1, heads=2),
    ):
        own = own_parameters(model, embedding)

        assert not shared & {id(parameter) for parameter in own}
        assert len(own) == len(list(model.parameters())) - len(shared)

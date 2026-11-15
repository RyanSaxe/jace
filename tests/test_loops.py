"""Run the limited loop end to end on the toy engine with tiny models.

The first two tests drive a whole iteration through `limited_iteration`,
the way a user would. In their toy setup, four pods of two seats draft
eight pools, and each of the two blocks of games has one deck from every
pool. The player, the win chance and the guess are small networks over the
toy boards, and the drafter, the deck builder and the deck scores are the
real models at a tiny size.

The deck search uses the human-trained deck score until `check_scores` finds
the learning deck score better, and from then on it uses the learning one. The
second test runs an iteration before the switch. It compares the drafter's
logits as well as its weights, so it also fails if another model's update
changes something the drafter reads. These tests show that the pieces fit
together. How well each piece learns is left to its own tests.
"""

import collections
import copy
import functools
import random
from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from jace.cards import CardPool, CardRecord, CardTable, Rarity, own_parameters
from jace.decks import BuildRules, DeckBuilder, DeckScore
from jace.draft import DrafterModel, DraftFormat
from jace.engine.toy import HAND_SIZE, TOY_CARDS, ToyBatch
from jace.loops import (
    FIRST_CHECK,
    Learners,
    LimitedFormat,
    LimitedSetup,
    ScoreRule,
    balanced_matchups,
    block_builds,
    limited_iteration,
    phased_schedule,
)
from jace.play import Boards, Earlier, Guess
from jace.train import PlayerRule, PlayGames, play_games

__all__ = ()


@dataclass(frozen=True, slots=True)
class _Toy:
    setup: LimitedSetup
    learners: Learners
    play: PlayGames
    # Every model the loop updates once the search uses the learning deck score, keyed by name.
    trained: dict[str, nn.Module]
    embedding: CardTable


def _toy_loop() -> _Toy:
    torch.manual_seed(0)
    records = [CardRecord(name="Island", set_code="toy", rarity=Rarity.BASIC, text="", program=())]
    records += [
        CardRecord(name=f"Card {card}", set_code="toy", rarity=Rarity.COMMON, text="", program=())
        for card in range(1, TOY_CARDS)
    ]
    pool = CardPool(records)
    embedding = CardTable(torch.randn(TOY_CARDS, 8))
    player = nn.Sequential(nn.Linear(HAND_SIZE + 1, 16), nn.Tanh(), nn.Linear(16, HAND_SIZE))
    win_chance_model = nn.Sequential(nn.Linear(HAND_SIZE + 1 + TOY_CARDS, 16), nn.Tanh(), nn.Linear(16, 1))
    guess = nn.Sequential(nn.Linear(HAND_SIZE + 1, 16), nn.Tanh(), nn.Linear(16, 2 * TOY_CARDS))
    drafter = DrafterModel(embedding, width=16, layers=1, heads=2, picks=3)
    builder = DeckBuilder(embedding, width=16, layers=1, heads=2)
    score = DeckScore(embedding, width=16, layers=1, heads=2)
    human_score = copy.deepcopy(score).requires_grad_(requires_grad=False)

    def policy(boards: Boards, _guess: Guess) -> Tensor:
        view = torch.cat([boards["hand"], boards["seen"]], dim=-1)
        return player(view)

    def win_chance(boards: Boards, truth: Guess) -> Tensor:
        view_and_truth = torch.cat([boards["hand"], boards["seen"], truth.hand], dim=-1)
        return win_chance_model(view_and_truth).squeeze(-1).sigmoid()

    def guesser(boards: Boards, _earlier: Earlier) -> Guess:
        output = guess(torch.cat([boards["hand"], boards["seen"]], dim=-1))
        return Guess(deck=output[:, :TOY_CARDS].softmax(dim=-1), hand=output[:, TOY_CARDS:].sigmoid())

    trained = {
        "player": player,
        "win_chance": win_chance_model,
        "guess": guess,
        "drafter": drafter,
        "deck_builder": builder,
        "deck_score": score,
    }
    setup = LimitedSetup(
        pool=pool,
        limited_format=LimitedFormat(
            draft=DraftFormat(seats=2, rounds=1, picks_per_round=3),
            booster=[{Rarity.COMMON: 1.0}] * 4,
            deck_size=5,
            min_lands=2,
            max_lands=3,
        ),
        booster_cards={Rarity.COMMON: pool.of_rarity(Rarity.COMMON)},
        rules=BuildRules(size=5, basics=(0,), nonbasic_lands=frozenset(), min_lands=2, max_lands=3),
        pods=4,
        deck_samples=16,
        spells_per_round=2,
        deck_temperature=1.0,
        score_rule=ScoreRule(window=2, passes=2, pull=1.0, switch_iterations=3, switch_margin=0.01),
        blocks=2,
        games_per_block=3,
        history=2,
        slice_rows=64,
        max_grad_norm=1.0,
        player_rule=PlayerRule(clip=0.2, stay=0.1, spread=0.0, keep_quantile=0.0, keep_floor=0.0, normalize=True),
        passes=2,
        lam_advantage=0.5,
        lam_value=0.8,
    )
    learners = Learners(
        policy=policy,
        win_chance=win_chance,
        guesser=guesser,
        drafter=drafter,
        human_drafter=None,
        builder=builder,
        score=score,
        human_score=human_score,
        optimizers={
            "player": torch.optim.Adam(own_parameters(player, embedding), lr=1e-2),
            "win_chance": torch.optim.Adam(own_parameters(win_chance_model, embedding), lr=1e-2),
            "guess": torch.optim.Adam(own_parameters(guess, embedding), lr=1e-2),
            "drafter": torch.optim.Adam(own_parameters(drafter, embedding), lr=1e-2),
            "deck_builder": torch.optim.Adam(own_parameters(builder, embedding), lr=1e-2),
            "deck_score": torch.optim.Adam(own_parameters(score, embedding), lr=1e-2),
        },
    )
    play = functools.partial(
        play_games,
        lambda matchups: ToyBatch(matchups, games=8, seed=0),
        learner=policy,
        opponent=policy,
        guesser=guesser,
        cards=TOY_CARDS,
        history=2,
        generator=torch.Generator().manual_seed(0),
    )
    return _Toy(setup=setup, learners=learners, play=play, trained=trained, embedding=embedding)


def _snapshot(modules: dict[str, nn.Module], *, embedding: CardTable) -> dict[str, list[Tensor]]:
    return {
        name: [parameter.detach().clone() for parameter in own_parameters(module, embedding)]
        for name, module in modules.items()
    }


def _moved(modules: dict[str, nn.Module], before: dict[str, list[Tensor]], *, embedding: CardTable) -> set[str]:
    return {
        name
        for name, module in modules.items()
        if any(
            not torch.equal(old, new) for old, new in zip(before[name], own_parameters(module, embedding), strict=True)
        )
    }


def test_an_iteration_after_the_switch_updates_every_model_but_the_human_trained_score() -> None:
    toy = _toy_loop()
    watched = {**toy.trained, "human_score": toy.learners.human_score}
    before = _snapshot(watched, embedding=toy.embedding)

    metrics, check = limited_iteration(
        toy.setup,
        toy.learners,
        check=replace(FIRST_CHECK, reads_learning=True),
        dials=phased_schedule(drafter_starts=0, pull_ends=10)(5),
        other_seats=[],
        play=toy.play,
        rng=random.Random(0),
    )

    # In each of the two blocks, eight decks play three games each, and every game seats two of them, so the
    # blocks have 24 games in all. `games` counts only the games that finished with a result, so every game
    # was played to its end.
    assert metrics["games"] == 24
    assert _moved(watched, before, embedding=toy.embedding) == set(toy.trained)
    assert check.reads_learning
    assert len(check.recent) == 1


def test_before_the_switch_the_builder_and_a_drafter_whose_rate_is_zero_stay_as_they_were() -> None:
    toy = _toy_loop()
    before = _snapshot(toy.trained, embedding=toy.embedding)
    drafter = toy.learners.drafter
    generator = torch.Generator().manual_seed(1)
    seen = (torch.rand(5, 2, TOY_CARDS, generator=generator) < 0.5).float()
    picks = torch.randint(0, TOY_CARDS, (5, 2), generator=generator)
    pack = torch.ones(5, TOY_CARDS)
    with torch.no_grad():
        picked_before = drafter.forward(seen, picks, pack=pack)

    _, check = limited_iteration(
        toy.setup,
        toy.learners,
        check=FIRST_CHECK,
        dials=phased_schedule(drafter_starts=100, pull_ends=200)(5),
        other_seats=[],
        play=toy.play,
        rng=random.Random(0),
    )

    with torch.no_grad():
        assert torch.equal(drafter.forward(seen, picks, pack=pack), picked_before)
    assert _moved(toy.trained, before, embedding=toy.embedding) == {"player", "win_chance", "guess", "deck_score"}
    assert not check.reads_learning


def test_each_pool_plays_each_build_once_and_every_block_mixes_every_rank() -> None:
    pools, blocks = 16, 4
    builds = block_builds(pools, blocks=blocks)

    assert builds.shape == (blocks, pools)
    for pool in range(pools):
        assert sorted(builds[:, pool].tolist()) == list(range(blocks))
    for block in range(blocks):
        assert collections.Counter(builds[block].tolist()) == dict.fromkeys(range(blocks), pools // blocks)


def test_every_deck_plays_the_same_number_of_games_against_other_pods_and_never_itself() -> None:
    seats = 8
    for pods in (1, 2, 3):
        games = balanced_matchups(pods, seats=seats, games_per_deck=5, rng=random.Random(pods))

        played = collections.Counter(deck for game in games for deck in game)
        assert played == dict.fromkeys(range(pods * seats), 5)
        assert all(first != second for first, second in games)
        if pods > 1:
            assert all(first // seats != second // seats for first, second in games)

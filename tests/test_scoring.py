"""Check that a match is scored fairly and that no game goes missing.

The pairing and the statistics are checked on the toy game and on outcomes
written by hand, where the right answer is known exactly. One test plays
real Magic in child processes to check the time limit: a game that stalls
must be recorded as hung while every other game still gets its result.
"""

import functools
import random
import time

import numpy as np
import torch

from jace.cards import CardPool, CardRecord, Rarity
from jace.engine import Outcome
from jace.engine.manabot import engine_cards
from jace.engine.scoring import Seat, Seating, play_games, random_seat
from jace.engine.toy import ToyBatch
from jace.eval import Deal, Game, add_tallies, check_guess, guess_report, legs, score_match, tally_guess, win_rate
from jace.play import Guess

__all__ = ()


def _deals(count: int) -> list[Deal]:
    return [Deal(decks=((0, 1), (2, 3)), seed=seed) for seed in range(count)]


def _a_wins(game: Game) -> Outcome:
    return Outcome.SEAT_0_WON if game.a_seat == 0 else Outcome.SEAT_1_WON


def _highest_card_outcome(game: Game) -> Outcome:
    # Plays one game of High Card in which both seats pick their highest card.
    batch = ToyBatch([game.decks], games=1, seed=game.seed)
    while True:
        highest = batch.decisions().boards["hand"].argmax(axis=1)
        finished = batch.act(highest.astype(np.int64))
        if finished:
            return finished[0].outcome


def _stalls_in_one_game(seed: int, *, stalled_seed: int) -> Seat:
    # Random play that never starts the one game whose seed it is given.
    if seed == stalled_seed:
        time.sleep(600)
    return random_seat(seed)


def test_a_player_scores_exactly_half_against_itself_whichever_seat_wins_each_deal() -> None:
    # An odd number of deals, so the seat that wins more of them cannot come out at a half.
    games = legs(_deals(51))
    outcomes = [_highest_card_outcome(game) for game in games]

    score = score_match(games, outcomes, resamples=200, generator=torch.Generator().manual_seed(0))

    assert len(set(outcomes)) == 2
    assert score.rate.rate == score.rate.low == score.rate.high == 0.5
    assert (score.deals, score.games) == (51, 204)


def test_games_without_a_result_are_counted_apart_and_never_as_a_win_or_a_loss() -> None:
    games = legs(_deals(3))
    outcomes = [_a_wins(game) for game in games]
    outcomes[1] = Outcome.ABANDONED
    outcomes[2] = Outcome.HUNG
    outcomes[8:] = [Outcome.HUNG, Outcome.ABANDONED, Outcome.HUNG, Outcome.HUNG]

    score = score_match(games, outcomes, resamples=200, generator=torch.Generator().manual_seed(0))

    assert score.rate.rate == 1.0
    assert (score.deals, score.games, score.abandoned, score.hung) == (2, 6, 2, 4)


def test_the_range_of_a_match_comes_from_its_deals_not_from_its_single_games() -> None:
    games = legs(_deals(40))
    # Player A wins all four games of every even deal and loses all four of every odd one.
    seat_0_won = [(game.a_seat == 0) == (game.deal % 2 == 0) for game in games]
    outcomes = [Outcome.SEAT_0_WON if won else Outcome.SEAT_1_WON for won in seat_0_won]
    generator = torch.Generator().manual_seed(0)

    by_deal = score_match(games, outcomes, resamples=2000, generator=generator).rate
    by_game = win_rate(torch.tensor([1.0, 0.0]).repeat_interleave(4).repeat(20), resamples=2000, generator=generator)

    assert by_deal.rate == by_game.rate == 0.5
    assert (by_deal.high - by_deal.low) > 1.6 * (by_game.high - by_game.low)


def test_guess_tallies_of_separate_games_add_up_to_the_report_of_all_their_decisions() -> None:
    generator = torch.Generator().manual_seed(0)
    chance = torch.rand(300, 7, generator=generator)
    truth = (torch.rand(chance.shape, generator=generator) < chance).float()

    whole = check_guess(Guess(deck=chance, hand=chance), hand_truth=truth)
    parts = guess_report(add_tallies(tally_guess(chance[rows], truth[rows]) for rows in (slice(0, 10), slice(10, 300))))

    for name in ("loss", "base_rate_loss", "stated", "observed"):
        assert torch.allclose(torch.tensor(getattr(parts, name)), torch.tensor(getattr(whole, name)), atol=1e-9)


def test_a_game_that_stalls_is_recorded_as_hung_and_every_other_game_still_gets_its_result() -> None:
    cards = engine_cards()
    pool = CardPool(
        [
            CardRecord(
                name=card.name, set_code=card.set_code, rarity=Rarity.COMMON, text=card.text, program=card.program
            )
            for card in cards
        ]
    )
    rng = random.Random(0)
    spells = [card for card, record in enumerate(cards) if record.set_code == "TLA" and "Land" not in record.text]
    decks = tuple((*([pool.card_id(basic)] * 17), *rng.sample(spells, 23)) for basic in ("Island", "Mountain"))
    games = legs([Deal(decks=(decks[0], decks[1]), seed=seed) for seed in (1, 2)])
    stalled = 5
    stalling = functools.partial(_stalls_in_one_game, stalled_seed=games[stalled].player_seeds[games[stalled].a_seat])

    seatings = [
        Seating(
            seed=game.seed,
            decks=game.decks,
            players=(stalling, random_seat) if game.a_seat == 0 else (random_seat, stalling),
            player_seeds=game.player_seeds,
        )
        for game in games
    ]

    played = play_games(
        seatings,
        check_guesses=tally_guess,
        names=[card.name for card in cards],
        workers=2,
        game_seconds=5.0,
        decklists="closed",
    )

    assert [index for index, game in enumerate(played) if game.outcome is Outcome.HUNG] == [stalled]
    assert all(game.decisions > 0 for index, game in enumerate(played) if index != stalled)

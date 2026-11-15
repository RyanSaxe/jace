"""Matches on paired deals, played the same way by the scoring command and the training loops.

A paired deal is two decks and an engine seed. It is played as four games,
so that each player plays each deck from each seat.

`jace.loops.score` scores saved checkpoints against each other, and the
training entry points score the player they are training every few
iterations. Both deal from held-out human decks, seat the players through
`jace.engine.scoring` and summarize the games with `jace.eval`. The pieces
they share are here, so a number in a training curve means the same as
that number in the scoring table.

The training entry points use `score_against_scripted`. It plays one
player against manabot's three scripted players, once sampling its actions
and once taking its top action, and appends each result to the run's
evaluations.
"""

import copy
import functools
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import torch

from jace.cards import Deck
from jace.engine import SEATS
from jace.engine.scoring import Played, Player, Seating, greedy_seat, heuristic_seat, play_games, random_seat
from jace.eval import (
    ActionChoice,
    Deal,
    Game,
    GuessTally,
    MatchScore,
    add_tallies,
    guess_report,
    legs,
    model_seat,
    score_match,
    tally_guess,
)
from jace.play import GuessModel, PlayerModel
from jace.tracking import CheckpointRef, Running, record_evaluation

__all__ = (
    "SCRIPTED",
    "Harness",
    "draw_deals",
    "guess_details",
    "model_player",
    "play_matches",
    "score_against_scripted",
)

# manabot's scripted players, by the name a result is recorded under.
SCRIPTED: Mapping[str, Player] = {"random": random_seat, "greedy": greedy_seat, "heuristic": heuristic_seat}
# A deal's engine seed is drawn with this many bits. The engine accepts any seed from 0 to 2**64 - 1, so every
# such seed fits.
_ENGINE_SEED_BITS = 31


@dataclass(frozen=True, slots=True)
class Harness:
    """The settings of a training run's own scoring, fixed for the whole run.

    `deals` are the paired deals played at every evaluation, so every
    evaluation in a run uses the same decks and shuffles. `names` is the
    engine's name of each card, indexed by `CardId`.

    The games run in `workers` processes, and a game longer than
    `game_seconds` is recorded as hung. Each win rate's range comes from
    resampling the deals `resamples` times.
    """

    deals: Sequence[Deal]
    names: Sequence[str]
    workers: int
    game_seconds: float
    resamples: int


def draw_deals(decks: Sequence[Deck], *, count: int, seed: int) -> list[Deal]:
    # Draws `count` deals: two different decks and an engine seed each.
    rng = random.Random(seed)
    deals: list[Deal] = []
    for _ in range(count):
        first, second = rng.sample(decks, SEATS)
        deals.append(Deal(decks=(first, second), seed=rng.getrandbits(_ENGINE_SEED_BITS)))
    return deals


def model_player(player: PlayerModel, guess: GuessModel, *, history: int, action: ActionChoice) -> Player:
    """Freeze the two models as they are now into a player for `jace.engine.scoring`.

    The player contains its own copies of the models on the CPU. A match
    therefore scores the weights of this moment while training goes on,
    and the copies can be sent to the processes that play the games
    whatever device the models train on.

    The guess model's copy has the meta that the guess model has now, built
    from the decks of earlier rounds. The decks dealt in the match are never
    added to it.

    `history` is how many earlier guesses the guess model is shown, and
    `action` is whether the player samples its action or takes its top one.
    """
    frozen_player = copy.deepcopy(player).cpu().requires_grad_(requires_grad=False)
    frozen_guess = copy.deepcopy(guess).cpu().requires_grad_(requires_grad=False)
    cards = frozen_player.embedding.cards
    return functools.partial(
        model_seat,
        policy=frozen_player.policy,
        guesser=frozen_guess.guess,
        cards=cards,
        history=history,
        action=action,
    )


def play_matches(
    games: Sequence[Game],
    pairs: Sequence[tuple[Player, Player]],
    *,
    names: Sequence[str],
    workers: int,
    game_seconds: float,
    decklists: Literal["open", "closed"],
) -> list[tuple[Played[GuessTally], ...]]:
    # Plays every game of `games` once for each pair of players (A, B). Player A takes each game's `a_seat`, and
    # player B takes the other seat. All the games go to one `play_games` call so that the worker processes start
    # only once. Returns one tuple per pair, with a result for each game in the order of `games`, and each result
    # contains the tally of each seat's hand guesses.
    seatings = [
        Seating(
            seed=game.seed,
            decks=game.decks,
            players=pair if game.a_seat == 0 else (pair[1], pair[0]),
            player_seeds=game.player_seeds,
        )
        for pair in pairs
        for game in games
    ]
    played = play_games(
        seatings,
        check_guesses=tally_guess,
        names=names,
        workers=workers,
        game_seconds=game_seconds,
        decklists=decklists,
    )
    return [played[first : first + len(games)] for first in range(0, len(played), len(games))]


def guess_details(tally: GuessTally) -> dict[str, object]:
    # The guess report as JSON values, or {} when the tally contains no guess. A range of stated chance that contains
    # no guess is NaN in the report and null here, because run records are strict JSON, which has no NaN.
    if not tally.pairs:
        return {}
    report = guess_report(tally)
    return {
        "guess": {
            "loss": report.loss,
            "base_rate_loss": report.base_rate_loss,
            "stated": [None if math.isnan(value) else value for value in report.stated],
            "observed": [None if math.isnan(value) else value for value in report.observed],
        }
    }


def score_against_scripted(
    run: Running,
    checkpoint: CheckpointRef,
    *,
    players: Mapping[ActionChoice, Player],
    harness: Harness,
    seed: int,
) -> dict[ActionChoice, dict[str, MatchScore]]:
    """Score each of `players` against each scripted player, and record the results under `checkpoint`.

    `players` maps each way of choosing an action to a player that chooses
    that way. Each player meets every scripted opponent on all of the
    harness's deals, four games a deal. All the games of all the matches
    are played in one `play_matches` call, so the worker processes start
    only once and share out the games.

    Each result is appended to the run's evaluations. Besides the win rate
    and its range, an entry contains the counts of finished, abandoned and
    hung games, the guess model's report on the games and how the player
    chose its actions. Its fields are named as in the entries that
    `jace.loops.score` writes, which also contain `seed` and
    `games_per_second`.

    The ranges for each action choice are resampled from a generator
    seeded with `seed`. Returns the score against each opponent, by action
    choice and then by opponent name.
    """
    games = legs(harness.deals)
    matches = [(action, name) for action in players for name in SCRIPTED]
    played = play_matches(
        games,
        [(players[action], SCRIPTED[name]) for action, name in matches],
        names=harness.names,
        workers=harness.workers,
        game_seconds=harness.game_seconds,
        decklists="closed",
    )
    scores: dict[ActionChoice, dict[str, MatchScore]] = {action: {} for action in players}
    generators = {action: torch.Generator().manual_seed(seed) for action in players}
    for (action, name), match in zip(matches, played, strict=True):
        score = score_match(
            games, [game.outcome for game in match], resamples=harness.resamples, generator=generators[action]
        )
        guesses = add_tallies(
            tally
            for game, result in zip(games, match, strict=True)
            if (tally := result.guesses[game.a_seat]) is not None
        )
        record_evaluation(
            run,
            checkpoint=checkpoint,
            opponent=name,
            deals=score.deals,
            rate=score.rate.rate,
            low=score.rate.low,
            high=score.rate.high,
            details={
                "games": score.games,
                "abandoned": score.abandoned,
                "hung": score.hung,
                "game_seconds": harness.game_seconds,
                "action": action,
                **guess_details(guesses),
            },
        )
        scores[action][name] = score
    return scores

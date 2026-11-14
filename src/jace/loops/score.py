"""Score checkpoints of a limited run against each other and against scripted opponents.

Every pair of players meets on the same deals: two human decks the run never
trained on and an engine seed, played four times so that each player holds
each deck in each seat. The table gives one player's win rate over the deals
with a range from resampling whole deals, the games the engine abandoned or
hung on, and a rating for every player on a scale that puts random play at
1000. Each checkpoint's results, and how well its guess model predicted the
opponent's hand in those games, are appended to the `evaluations.jsonl` of
the run it came from. A player is named by the iteration of its checkpoint,
or `top-<iteration>` for that checkpoint taking its top action where it
would otherwise sample, or `random`, `greedy`, `heuristic` or
`flat-<games per action>` for one of manabot's players. The model sizes, the card set and the data folder are
read from the run's own record. Run it with
`uv run python -m jace.loops.score --help`.
"""

import argparse
import dataclasses
import functools
import itertools
import json
import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from jace.cards import CardTable, Deck
from jace.data import SetCards, card_table, load_decks, read_set
from jace.engine import BoardLayout
from jace.engine.manabot import board_layout, engine_card_ids, engine_cards
from jace.engine.scoring import (
    Played,
    Player,
    RatedGame,
    Rating,
    fit_ratings,
    flat_seat,
)
from jace.eval import (
    ActionChoice,
    Game,
    GuessTally,
    MatchScore,
    WinRate,
    a_results,
    add_tallies,
    guess_report,
    legs,
    model_seat,
    score_match,
)
from jace.loops.entry_points import card_inputs, load_card_table, playable_decks
from jace.loops.matches import SCRIPTED, draw_deals, guess_details, play_matches
from jace.play import GuessModel, PlayerModel
from jace.tracking import CheckpointRef, ModelName, Running, evaluating, load_checkpoint, record_evaluation

__all__ = ("main",)

# The player every rating is measured from, at 1000.
_ANCHOR = "random"
_FLAT_PREFIX = "flat-"
# A checkpoint named with this prefix takes its top action at every decision, where it would otherwise sample.
_TOP_PREFIX = "top-"


@dataclass(frozen=True, slots=True)
class _Arguments:
    run: Path
    players: Sequence[str]
    pairs: Sequence[str]
    deals: int
    decks: int
    workers: int
    game_seconds: float
    resamples: int
    seed: int


@dataclass(frozen=True, slots=True)
class _Match:
    # a, b: the names of player A and player B. They are equal when a player faces itself.
    a: str
    b: str
    played: tuple[Played[GuessTally], ...]
    seconds: float


def _arguments(argv: Sequence[str]) -> _Arguments:
    parser = argparse.ArgumentParser(
        description="Score checkpoints of a limited run against each other and against scripted opponents.",
    )
    parser.add_argument("--run", type=Path, required=True, help="the run folder whose checkpoints to score")
    parser.add_argument(
        "--players",
        nargs="+",
        required=True,
        help="each an iteration that has a checkpoint, top-N for checkpoint N taking its top action, or random, "
        "greedy, heuristic, or flat-N: a search of N games",
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=[],
        help="the pairs to play, each as A:B; a player may face itself. The default is every two of --players",
    )
    parser.add_argument("--deals", type=int, default=50, help="deals per pair; each is four games")
    parser.add_argument("--decks", type=int, default=64, help="held-out human decks the deals are drawn from")
    parser.add_argument("--workers", type=int, default=8, help="processes that play games at once")
    parser.add_argument(
        "--game-seconds", type=float, default=120.0, help="a game that takes longer is recorded as hung"
    )
    parser.add_argument("--resamples", type=int, default=1000, help="resamples of the deals behind every range")
    parser.add_argument("--seed", type=int, default=0)
    return _Arguments(**vars(parser.parse_args(argv)))


def _checkpoint(run: Path, model: ModelName, *, update: int) -> CheckpointRef:
    lines = (run / "checkpoints" / "index.jsonl").read_text().splitlines()
    for record in map(json.loads, lines):
        if record["model"] == model and record["update"] == update:
            return CheckpointRef(run_id=run.name, model=model, update=update, sha256=record["sha256"])
    message = f"run {run} has no {model} checkpoint of iteration {update}"
    raise ValueError(message)


def _action(name: str) -> ActionChoice:
    return "top" if name.startswith(_TOP_PREFIX) else "sampled"


def _model_player(
    update: int,
    *,
    run: Path,
    config: Mapping[str, object],
    cards: SetCards,
    embedding: CardTable,
    layout: BoardLayout,
    decks: Sequence[Deck],
    action: ActionChoice,
) -> tuple[Player, CheckpointRef]:
    # Builds the player and the guess model of iteration `update` as the run built them, on the
    # run's card table, and returns the player with its player checkpoint. The guess model's field
    # is `decks`, the decks the opponents are dealt, in place of the training decks it was saved with.
    numbers = {name: value for name, value in config.items() if isinstance(value, int)}
    engine_to_card = engine_card_ids(cards.pool)
    shape = {"width": numbers["play_width"], "layers": numbers["play_layers"], "heads": numbers["play_heads"]}
    player = PlayerModel(
        embedding,
        engine_to_card=engine_to_card,
        layout=layout,
        guess_cards=numbers["guess_cards"],
        deck_guess_cards=numbers["deck_guess_cards"],
        **shape,
    )
    guess = GuessModel(embedding, engine_to_card=engine_to_card, layout=layout, **shape)
    reference = _checkpoint(run, "player", update=update)
    load_checkpoint(run.parent, reference, into=player, keeping=embedding)
    load_checkpoint(run.parent, _checkpoint(run, "guess", update=update), into=guess, keeping=embedding)
    guess.set_field(decks)
    for module in (embedding, player, guess):
        module.requires_grad_(requires_grad=False)
    seat = functools.partial(
        model_seat,
        policy=player.policy,
        guesser=guess.guess,
        cards=len(cards.pool),
        history=numbers["history"],
        action=action,
    )
    return seat, reference


def _players(
    names: Sequence[str],
    *,
    run: Path,
    config: Mapping[str, object],
    cards: SetCards,
    embedding: CardTable,
    layout: BoardLayout,
    decks: Sequence[Deck],
) -> tuple[dict[str, Player], dict[str, CheckpointRef]]:
    # Returns every named player, and the player checkpoint of each one that is a checkpoint.
    players: dict[str, Player] = {}
    checkpoints: dict[str, CheckpointRef] = {}
    for name in names:
        if name in SCRIPTED:
            players[name] = SCRIPTED[name]
        elif name.startswith(_FLAT_PREFIX):
            players[name] = functools.partial(flat_seat, sims=int(name.removeprefix(_FLAT_PREFIX)))
        else:
            players[name], checkpoints[name] = _model_player(
                int(name.removeprefix(_TOP_PREFIX)),
                run=run,
                config=config,
                cards=cards,
                embedding=embedding,
                layout=layout,
                decks=decks,
                action=_action(name),
            )
    return players, checkpoints


def _seen_by(name: str, match: _Match, *, games: Sequence[Game]) -> tuple[Game, ...]:
    # The games with `name` as player A, which exchanges the two players when `name` is the match's player B.
    if name == match.a:
        return tuple(games)
    return tuple(dataclasses.replace(game, a_seat=1 - game.a_seat) for game in games)


def _guesses(name: str, match: _Match, *, games: Sequence[Game]) -> GuessTally:
    # The tally of every hand guess `name` made in the match, from both seats when it faced itself.
    tallies: list[GuessTally] = []
    for game, played in zip(games, match.played, strict=True):
        seats = {seat for seat, seated in ((game.a_seat, match.a), (1 - game.a_seat, match.b)) if seated == name}
        tallies += [tally for seat in seats if (tally := played.guesses[seat]) is not None]
    return add_tallies(tallies)


def _record_match(
    run: Running,
    match: _Match,
    *,
    games: Sequence[Game],
    checkpoints: Mapping[str, CheckpointRef],
    arguments: _Arguments,
    generator: torch.Generator,
) -> MatchScore:
    # Writes one evaluation for each checkpoint in the match, from its own side, and returns player A's score.
    score = score_match(
        games, [played.outcome for played in match.played], resamples=arguments.resamples, generator=generator
    )
    # Player B's rate is one minus player A's, so its range is A's range turned around.
    b_rate = WinRate(rate=1.0 - score.rate.rate, low=1.0 - score.rate.high, high=1.0 - score.rate.low)
    # A player that faced itself is recorded once.
    sides = [(match.a, match.b, score.rate), (match.b, match.a, b_rate)][: len({match.a, match.b})]
    for name, opponent, rate in sides:
        if name not in checkpoints:
            continue
        record_evaluation(
            run,
            checkpoint=checkpoints[name],
            opponent=opponent,
            deals=score.deals,
            rate=rate.rate,
            low=rate.low,
            high=rate.high,
            details={
                "games": score.games,
                "abandoned": score.abandoned,
                "hung": score.hung,
                "seed": arguments.seed,
                "game_seconds": arguments.game_seconds,
                "games_per_second": len(games) / match.seconds,
                "action": _action(name),
                **guess_details(_guesses(name, match, games=games)),
            },
        )
    return score


def _rated_games(matches: Sequence[_Match], *, games: Sequence[Game]) -> list[RatedGame]:
    # Every game with a result between two different players. A player facing itself says nothing about ratings.
    rated: list[RatedGame] = []
    for match in matches:
        if match.a == match.b:
            continue
        results = a_results(games, [played.outcome for played in match.played]).tolist()
        rated += [
            RatedGame(player_a=match.a, player_b=match.b, a_seat=game.a_seat, a_result=result, deal=game.deal)
            for game, result in zip(games, results, strict=True)
            if not math.isnan(result)
        ]
    return rated


def _record_ratings(
    run: Running,
    ratings: Mapping[str, Rating],
    *,
    matches: Sequence[_Match],
    games: Sequence[Game],
    checkpoints: Mapping[str, CheckpointRef],
    arguments: _Arguments,
    generator: torch.Generator,
) -> None:
    # Writes one evaluation for each rated checkpoint: its rating, and its win rate over every
    # game it played against the other players, with deals resampled across all of them at once.
    for name in ratings.keys() & checkpoints.keys():
        faced = [match for match in matches if name in {match.a, match.b} and match.a != match.b]
        score = score_match(
            [game for match in faced for game in _seen_by(name, match, games=games)],
            [played.outcome for match in faced for played in match.played],
            resamples=arguments.resamples,
            generator=generator,
        )
        record_evaluation(
            run,
            checkpoint=checkpoints[name],
            opponent="population",
            deals=score.deals,
            rate=score.rate.rate,
            low=score.rate.low,
            high=score.rate.high,
            details={
                "games": score.games,
                "abandoned": score.abandoned,
                "hung": score.hung,
                "seed": arguments.seed,
                "action": _action(name),
                "opponents": sorted({player for match in faced for player in (match.a, match.b)} - {name}),
                "anchor": _ANCHOR,
                "rating": ratings[name].rating,
                "rating_low": ratings[name].low,
                "rating_high": ratings[name].high,
            },
        )


def _write(line: str) -> None:
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def _write_guesses(matches: Sequence[_Match], *, games: Sequence[Game], names: Sequence[str]) -> None:
    # One block for each player that guessed: its loss over all its matches and the table of chances.
    for name in names:
        tally = add_tallies([_guesses(name, match, games=games) for match in matches if name in {match.a, match.b}])
        if not tally.pairs:
            continue
        report = guess_report(tally)
        _write(f"\nguess model of {name}: loss {report.loss:.4f}, base-rate loss {report.base_rate_loss:.4f}")
        _write("  stated chance   observed   guesses")
        for stated, observed, count in zip(report.stated, report.observed, tally.counts, strict=True):
            if count:
                _write(f"  {stated:13.3f}   {observed:8.3f}   {int(count):7d}")


def main(argv: Sequence[str]) -> None:
    """Play every pair, print the table and append each checkpoint's results to its run.

    A pair's line is printed and recorded as soon as its games end, so a
    stopped command keeps the pairs it finished. Ratings need random play
    among the pairs, as the player they are measured from, and are fitted
    after the last pair. The run's writer lock is held throughout, so a
    run that is still training cannot be scored.
    """
    arguments = _arguments(argv)
    config = json.loads((arguments.run / "identity.json").read_text())["config"]
    data = read_set(str(config["set_code"]))
    cards = card_table(data.attributes, data.statistics, programs={card.name: card.program for card in engine_cards()})
    data_dir = Path(str(config["data_dir"]))
    human = load_decks(
        data.games, pool=cards.pool, cache_dir=data_dir, held_out_share=float(str(config["held_out_share"]))
    )
    # Self-play never trains the card table, so every checkpoint of the run reads the newest one.
    embedding, _ = load_card_table(arguments.run, inputs=card_inputs(data, cards, data_dir=data_dir))
    # The decks the run kept out of training, the same ones it scored itself on.
    decks = playable_decks(human.decks[human.held_out], pool=cards.pool, count=arguments.decks)
    deals = draw_deals(decks, count=arguments.deals, seed=arguments.seed)
    games = legs(deals)
    players, checkpoints = _players(
        arguments.players,
        run=arguments.run,
        config=config,
        cards=cards,
        embedding=embedding,
        layout=board_layout(),
        decks=[deck for deal in deals for deck in deal.decks],
    )
    pairs = [tuple(pair.split(":")) for pair in arguments.pairs] or list(itertools.combinations(arguments.players, 2))
    names = tuple(cards.pool[card].name for card in range(len(cards.pool)))
    generator = torch.Generator().manual_seed(arguments.seed)
    matches: list[_Match] = []
    _write(
        f"{'A':<12} {'B':<12} {'A wins':>6}  {'95% range':<14} {'deals':>5} {'games':>5} {'abandoned':>9} {'hung':>4}"
    )
    with evaluating(arguments.run) as run:
        for a, b in pairs:
            began = time.perf_counter()
            (played,) = play_matches(
                games,
                [(players[a], players[b])],
                names=names,
                workers=arguments.workers,
                game_seconds=arguments.game_seconds,
                decklists="open" if _FLAT_PREFIX in a + b else "closed",
            )
            match = _Match(a=a, b=b, played=played, seconds=time.perf_counter() - began)
            matches.append(match)
            score = _record_match(
                run, match, games=games, checkpoints=checkpoints, arguments=arguments, generator=generator
            )
            _write(
                f"{a:<12} {b:<12} {score.rate.rate:6.3f}  {score.rate.low:.3f} to {score.rate.high:.3f} "
                f"{score.deals:5d} {score.games:5d} {score.abandoned:9d} {score.hung:4d}"
                f"   {len(games) / match.seconds:.1f} games a second"
            )
        rated = _rated_games(matches, games=games)
        if any(_ANCHOR in {game.player_a, game.player_b} for game in rated):
            ratings = fit_ratings(rated, anchor=_ANCHOR, resamples=arguments.resamples, seed=arguments.seed)
            _record_ratings(
                run,
                ratings,
                matches=matches,
                games=games,
                checkpoints=checkpoints,
                arguments=arguments,
                generator=generator,
            )
            _write(f"\n{'player':<12} {'rating':>6}  95% range")
            for name, rating in sorted(ratings.items(), key=lambda item: -item[1].rating):
                _write(f"{name:<12} {rating.rating:6.0f}  {rating.low:.0f} to {rating.high:.0f}")
    _write_guesses(matches, games=games, names=arguments.players)


if __name__ == "__main__":
    main(sys.argv[1:])

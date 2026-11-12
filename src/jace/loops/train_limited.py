"""Train every model on real games of one card set, starting from human data.

This is the limited loop run for real: packs of the set's playable cards are
drafted, a deck is built from every pool, and the decks play each other on
the manabot engine. The drafter, the deck models and the card embedding can
start from checkpoints trained on human data; the player and the guess model
always start untrained. Every few iterations the player is scored against
uniformly random play on human decks it never trains on, which is the one
number that shows whether play is improving. Run it with
`uv run python -m jace.loops.train_limited --help`.
"""

import argparse
import copy
import functools
import json
import random
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import Tensor, nn

from jace.cards import CardEmbedding, CardPool, Deck, Rarity
from jace.data import SetCards, card_table, load_decks, read_set
from jace.decks import BuildRules, DeckBuilderModel, DeckScore
from jace.draft import TLA_PLAY_BOOSTER, DrafterModel, DraftFormat
from jace.engine import SEATS, GameBatch
from jace.engine.manabot import engine_card_ids, engine_cards, start_batch
from jace.eval import WinRate, win_rate
from jace.loops.limited import Learners, LimitedSetup, StartBatch, limited_iteration
from jace.loops.schedules import phased_schedule
from jace.play import Boards, Guesser, GuessModel, PlayerModel, Policy, as_boards, random_policy
from jace.tracking import (
    CheckpointRef,
    Identity,
    ModelName,
    Running,
    create_run,
    load_checkpoint,
    log_update,
    save_checkpoint,
    writing,
)
from jace.train import PlayerRule, collect

__all__ = ("main",)

_DECK_SIZE = 40
_MIN_LANDS = 15
_MAX_LANDS = 18
_SEATS_IN_A_POD = 8
_BOOSTERS_EACH = 3
# Ataraxos's player rule: learn only from the quarter of decisions with the largest advantages.
_PLAYER_RULE = PlayerRule(clip=0.2, stay=0.1, spread=0.0, keep_quantile=0.75, keep_floor=0.0, value_weight=0.5)
_LAM_ADVANTAGE = 0.5
_LAM_VALUE = 0.8
_RANGE_RESAMPLES = 1000
_SCORE_CHUNK = 1024
_SLICE_ROWS = 512
# PyTorch seeds a generator with a signed 64-bit integer, and the engine with an unsigned 32-bit one.
_SEED_BITS = 63
_ENGINE_SEED_BITS = 31
_TRAINED: tuple[ModelName, ...] = ("player", "guess", "drafter", "deck_builder", "deck_score", "embedding")


def _arguments(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train every model on real games of one card set.")
    parser.add_argument("--data-dir", type=Path, required=True, help="where the processed 17Lands tensors are cached")
    parser.add_argument("--run-dir", type=Path, required=True, help="the folder that holds run folders")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--set", default="TLA", help="the card set, already downloaded with `spells add`")
    parser.add_argument("--drafter-run", type=Path, help="a run folder whose drafter checkpoint to start from")
    parser.add_argument("--decks-run", type=Path, help="a run folder whose deck model checkpoints to start from")
    parser.add_argument("--embedding-run", type=Path, help="a run folder whose card embedding to start from")
    parser.add_argument("--games", type=int, default=64, help="games played at once")
    parser.add_argument("--decisions", type=int, default=4096, help="learner decisions collected per iteration")
    parser.add_argument("--pods", type=int, default=2, help="eight-seat drafts per iteration")
    parser.add_argument("--proposals", type=int, default=8, help="candidate decks per pool")
    parser.add_argument("--search-rounds", type=int, default=1)
    parser.add_argument("--evaluate-every", type=int, default=10)
    parser.add_argument("--evaluate-decisions", type=int, default=8192)
    parser.add_argument("--evaluate-decks", type=int, default=64, help="held-out human decks used for scoring")
    parser.add_argument("--drafter-starts", type=int, default=20, help="the iteration the drafter starts learning at")
    parser.add_argument("--pull-ends", type=int, default=200, help="the iteration the human pull reaches zero at")
    parser.add_argument("--history", type=int, default=4, help="earlier guesses the guess model is shown")
    parser.add_argument("--play-width", type=int, default=128)
    parser.add_argument("--play-layers", type=int, default=2)
    parser.add_argument("--play-heads", type=int, default=4)
    parser.add_argument("--guess-cards", type=int, default=8)
    parser.add_argument("--drafter-width", type=int, default=256)
    parser.add_argument("--drafter-layers", type=int, default=6)
    parser.add_argument("--drafter-heads", type=int, default=8)
    parser.add_argument("--deck-width", type=int, default=128)
    parser.add_argument("--deck-layers", type=int, default=2)
    parser.add_argument("--deck-heads", type=int, default=4)
    parser.add_argument("--embedding-hidden", type=int, default=256)
    parser.add_argument("--rate", type=float, default=1e-4, help="the learning rate of the models with no schedule")
    parser.add_argument("--held-out-share", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jace-commit", default="unknown")
    parser.add_argument("--manabot-commit", default="unknown")
    return parser.parse_args(argv)


def _latest(run_folder: Path, model: ModelName) -> CheckpointRef:
    # The newest checkpoint of `model` that the run at `run_folder` recorded.
    lines = (run_folder / "checkpoints" / "index.jsonl").read_text().splitlines()
    records = [record for record in map(json.loads, lines) if record["model"] == model]
    newest = max(records, key=lambda record: record["update"])
    return CheckpointRef(run_id=run_folder.name, model=model, update=newest["update"], sha256=newest["sha256"])


def _own(module: nn.Module) -> list[nn.Parameter]:
    # The shared embedding is trained by its own optimizer, so it is left out of each model's.
    return [parameter for name, parameter in module.named_parameters() if not name.startswith("embedding.")]


def _human_decks(decks: Tensor, *, pool: CardPool, count: int) -> list[Deck]:
    # decks: uint8 [n, cards]. The first `count` decks whose every card the engine can play.
    playable = torch.tensor([bool(pool[card].program) for card in range(len(pool))])
    usable = decks[(decks[:, ~playable] == 0).all(dim=1)][:count]
    return [tuple(torch.repeat_interleave(torch.arange(len(pool)), row.long()).tolist()) for row in usable]


@dataclass(frozen=True, slots=True)
class _Models:
    # Every trained thing of a run. The five models share `embedding`.
    player: PlayerModel
    guess: GuessModel
    drafter: DrafterModel
    builder: DeckBuilderModel
    score: DeckScore
    embedding: CardEmbedding


def _by_name(models: _Models) -> dict[ModelName, nn.Module]:
    return {
        "player": models.player,
        "guess": models.guess,
        "drafter": models.drafter,
        "deck_builder": models.builder,
        "deck_score": models.score,
        "embedding": models.embedding,
    }


def _build_models(
    arguments: argparse.Namespace, cards: SetCards, *, sample: Boards, engine_to_card: Tensor, picks: int
) -> _Models:
    embedding = CardEmbedding(cards.features, hidden=arguments.embedding_hidden, offset_bound=None)
    deck_shape = {"width": arguments.deck_width, "layers": arguments.deck_layers, "heads": arguments.deck_heads}
    play_shape = {"width": arguments.play_width, "layers": arguments.play_layers, "heads": arguments.play_heads}
    return _Models(
        player=PlayerModel(
            embedding, engine_to_card=engine_to_card, sample=sample, guess_cards=arguments.guess_cards, **play_shape
        ),
        guess=GuessModel(embedding, engine_to_card=engine_to_card, sample=sample, **play_shape),
        drafter=DrafterModel(
            embedding,
            width=arguments.drafter_width,
            layers=arguments.drafter_layers,
            heads=arguments.drafter_heads,
            picks=picks,
        ),
        builder=DeckBuilderModel(embedding, **deck_shape),
        score=DeckScore(embedding, **deck_shape),
        embedding=embedding,
    )


def _load_inputs(arguments: argparse.Namespace, models: _Models) -> tuple[CheckpointRef, ...]:
    # Loads the checkpoints the arguments name and returns their references.
    # The embedding is loaded last: every model's checkpoint also holds the embedding it was saved with.
    modules = _by_name(models)
    sources: list[tuple[Path | None, ModelName]] = [
        (arguments.drafter_run, "drafter"),
        (arguments.decks_run, "deck_builder"),
        (arguments.decks_run, "deck_score"),
        (arguments.embedding_run, "embedding"),
    ]
    inputs: list[CheckpointRef] = []
    for run_folder, model in sources:
        if run_folder is not None:
            ref = _latest(run_folder, model)
            load_checkpoint(run_folder.parent, ref, into=modules[model])
            inputs.append(ref)
    return tuple(inputs)


def _score_against(
    opponent: Policy,
    *,
    learner: Policy,
    guesser: Guesser,
    start: StartBatch,
    decks: Sequence[Deck],
    arguments: argparse.Namespace,
    cards: int,
    rng: random.Random,
) -> WinRate:
    """Play `learner` against `opponent` on `decks` and return the learner's win rate.

    Each game deals two of `decks` at random and gives the learner a random
    seat. Games are played until the finished ones hold
    `arguments.evaluate_decisions` learner decisions, so long games that are
    still running at that point are left out, which favours short games a
    little. The range treats games as independent.
    """
    generator = torch.Generator().manual_seed(rng.getrandbits(_SEED_BITS))

    def matchup() -> tuple[Deck, Deck]:
        first, second = rng.sample(decks, SEATS)
        return first, second

    collected = collect(
        start(matchup),
        None,
        learner=learner,
        opponent=opponent,
        guesser=guesser,
        cards=cards,
        history=arguments.history,
        decisions=arguments.evaluate_decisions,
        generator=generator,
    )
    results = torch.tensor([game.learner_result for game in collected.games])
    return win_rate(results, resamples=_RANGE_RESAMPLES, generator=generator)


def _train(
    run: Running,
    arguments: argparse.Namespace,
    *,
    iterate: Callable[[int], dict[str, float]],
    score_against_random: Callable[[], WinRate],
    curve_path: Path,
) -> None:
    # Runs every iteration, scoring the player before the first and every `evaluate_every` after,
    # and rewrites the curve file after each iteration so a stopped run still has its curve.
    iterations: list[dict[str, float]] = []
    evaluations: list[dict[str, float]] = []
    started = time.perf_counter()
    for iteration in range(arguments.iterations + 1):
        if iteration > 0:
            began = time.perf_counter()
            metrics = {**iterate(iteration), "seconds": time.perf_counter() - began}
            log_update(run, model="player", update=iteration, metrics=metrics)
            iterations.append({"iteration": iteration, **metrics})
        if iteration % arguments.evaluate_every == 0:
            rate = score_against_random()
            evaluations.append({"iteration": iteration, "rate": rate.rate, "low": rate.low, "high": rate.high})
            sys.stdout.write(f"iteration {iteration:4d}  win rate against random {rate.rate:.3f}\n")
            sys.stdout.flush()
        summary = {"run_id": run.folder.name, "seconds": time.perf_counter() - started}
        curve_path.write_text(json.dumps({**summary, "iterations": iterations, "evaluations": evaluations}))


def main(argv: Sequence[str]) -> None:
    """Run limited training on one card set and save its record, checkpoints and curve.

    The run folder is created under `--run-dir` by `jace.tracking` and gets
    a checkpoint of every trained thing at the end. The curve is written
    beside it as `<run id>.curve.json`: the metrics of every iteration and
    the player's win rate against random play at each evaluation.
    """
    arguments = _arguments(argv)
    torch.manual_seed(arguments.seed)
    rng = random.Random(arguments.seed)
    data = read_set(arguments.set)
    cards = card_table(data.attributes, data.statistics, programs={card.name: card.program for card in engine_cards()})
    pool = cards.pool
    playable = {rarity: [card for card in pool.of_rarity(rarity) if pool[card].program] for rarity in Rarity}
    is_land = cards.features[:, cards.feature_names.index("type_land")] > 0
    basics = pool.of_rarity(Rarity.BASIC)
    draft_format = DraftFormat(seats=_SEATS_IN_A_POD, rounds=_BOOSTERS_EACH, picks_per_round=len(TLA_PLAY_BOOSTER))
    human = load_decks(data.games, pool=pool, cache_dir=arguments.data_dir, held_out_share=arguments.held_out_share)
    scoring_decks = _human_decks(human.decks[human.held_out], pool=pool, count=arguments.evaluate_decks)

    def start(matchups: Callable[[], tuple[Deck, Deck]]) -> GameBatch:
        # Every batch needs its own seed, so this cannot be a partial application of `start_batch`.
        # ast-grep-ignore: no-routing-functions
        return start_batch(matchups, pool=pool, games=arguments.games, seed=rng.getrandbits(_ENGINE_SEED_BITS))

    first_decks = (scoring_decks[0], scoring_decks[1])
    sample = as_boards(start(lambda: first_decks).decisions(), device=torch.device("cpu"))
    engine_to_card = engine_card_ids(pool)
    picks = draft_format.rounds * draft_format.picks_per_round
    models = _build_models(arguments, cards, sample=sample, engine_to_card=engine_to_card, picks=picks)
    inputs = _load_inputs(arguments, models)
    human_drafter = None
    if arguments.drafter_run is not None:
        human_drafter = copy.deepcopy(models.drafter).eval().requires_grad_(requires_grad=False)

    modules = _by_name(models)
    # `_own` leaves the embedding out of each model and keeps all of the embedding's own parameters.
    optimizers = {name: torch.optim.Adam(_own(module), lr=arguments.rate) for name, module in modules.items()}
    setup = LimitedSetup(
        pool=pool,
        draft_format=draft_format,
        booster=TLA_PLAY_BOOSTER,
        booster_cards=playable,
        rules=BuildRules(
            size=_DECK_SIZE,
            basics=basics,
            nonbasic_lands=frozenset(
                card for cards_of_rarity in playable.values() for card in cards_of_rarity if bool(is_land[card])
            )
            - frozenset(basics),
            min_lands=_MIN_LANDS,
            max_lands=_MAX_LANDS,
        ),
        pods=arguments.pods,
        proposals=arguments.proposals,
        search_rounds=arguments.search_rounds,
        score_chunk=_SCORE_CHUNK,
        decisions=arguments.decisions,
        history=arguments.history,
        slice_rows=_SLICE_ROWS,
        max_grad_norm=1.0,
        player_rule=_PLAYER_RULE,
        lam_advantage=_LAM_ADVANTAGE,
        lam_value=_LAM_VALUE,
    )
    learners = Learners(
        policy=models.player.policy,
        guesser=models.guess.guess,
        drafter=models.drafter,
        human_drafter=human_drafter,
        builder=models.builder,
        score=models.score,
        optimizers=optimizers,
    )
    schedule = phased_schedule(drafter_starts=arguments.drafter_starts, pull_ends=arguments.pull_ends)

    def iterate(iteration: int) -> dict[str, float]:
        metrics = limited_iteration(
            setup,
            learners,
            dials=schedule(iteration),
            opponent=models.player.policy,
            other_seats=(),
            start=start,
            rng=rng,
        )
        return dict(metrics)

    score_against_random = functools.partial(
        _score_against,
        random_policy(engine_to_card),
        learner=models.player.policy,
        guesser=models.guess.guess,
        start=start,
        decks=scoring_decks,
        arguments=arguments,
        cards=len(pool),
        rng=rng,
    )
    clock = functools.partial(datetime.now, UTC)
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(arguments).items()}
    identity = Identity(
        config=config,
        seed=arguments.seed,
        jace_commit=arguments.jace_commit,
        manabot_commit=arguments.manabot_commit,
        inputs=inputs,
    )
    folder = create_run(arguments.run_dir, identity, clock=clock)
    with writing(folder, clock=clock) as run:
        _train(
            run,
            arguments,
            iterate=iterate,
            score_against_random=score_against_random,
            curve_path=folder.with_name(f"{folder.name}.curve.json"),
        )
        for name in _TRAINED:
            save_checkpoint(run, model=name, update=arguments.iterations, module=modules[name])


if __name__ == "__main__":
    main(sys.argv[1:])

"""What the command-line entry points of `jace.loops` share, as plain functions.

Four entry points train models: `pretrain`, `pretrain_decks`,
`train_limited` and `train_player`. Each one parses its flags into a frozen
record, creates a run folder and writes a curve file beside the folder.
This module defines the flags they all take, the record that contains
them, the choice of device and the creation of the run folder, so that all
four agree on what a run folder and its config look like.

The card encoder's inputs and shape flags are here too, with
`load_card_table`. Every entry point after the drafter's pretraining reads
its card vectors with `load_card_table`. That includes `jace.loops.score`,
which takes none of the shared flags.

`train_limited` and `train_player` both train a player on engine games, and
they share the pieces that play those games and score the player.
`PlayArguments` are their common flags, and `play_actor` builds the `Actor`
that a worker process plays with. `starting_meta` fits the meta that their
guess model starts from, and `scoring_harness` and `score_player` score the
player on held-out decks.

This is not a framework. An entry point calls what it needs and does the
rest itself.
"""

import argparse
import copy
import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from jace.cards import FIRST_PROGRAM_TOKEN, CardEncoder, CardInputs, CardPool, CardTable, Deck, frozen_table
from jace.data import (
    TEXT_MODEL,
    SetCards,
    SetData,
    card_table,
    card_texts,
    load_decks,
    program_tokens,
    read_set,
    text_vectors,
)
from jace.data.hugging_face import hugging_face_encoder
from jace.engine import GameBatch, Matchup
from jace.engine.manabot import (
    board_layout,
    engine_card_ids,
    engine_cards,
    land_and_spell_actions,
    program_vocabulary,
    start_batch,
    turn_numbers,
)
from jace.loops.formats import booster_meta, limited_format
from jace.loops.matches import Harness, draw_deals, model_player, score_against_scripted
from jace.play import BOARD_ARRAYS, GuessModel, Meta, PlayerModel, WinChanceModel, fit_meta
from jace.tracking import (
    CheckpointRef,
    Clock,
    Identity,
    Running,
    create_run,
    latest_checkpoint,
    load_checkpoint,
    read_checkpoint,
)
from jace.train import Rollout
from jace.train.workers import Actor, Weights

__all__ = (
    "ADAM_BETAS",
    "NotCardVectorsError",
    "PlayArguments",
    "PlayModels",
    "RunArguments",
    "add_card_flags",
    "add_play_flags",
    "add_run_flags",
    "build_play_models",
    "card_inputs",
    "device_named",
    "land_and_spell_rates",
    "load_card_table",
    "play_actor",
    "play_weights",
    "playable_decks",
    "score_player",
    "scoring_harness",
    "set_cards",
    "start_run",
    "starting_meta",
)

# Adam's decay rates for pretraining on human data, the ones an earlier human-trained drafter was trained with.
ADAM_BETAS = (0.9, 0.98)

# Each round's engine seed is drawn with this many bits. `start_batch` accepts any non-negative seed and derives
# the engine's seeds from it.
_ENGINE_SEED_BITS = 31
# A scoring game that takes longer than this many seconds is recorded as hung.
_GAME_SECONDS = 120.0
_RANGE_RESAMPLES = 1000


@dataclass(frozen=True, slots=True)
class RunArguments:
    """The flags every training entry point takes, which `add_run_flags` adds to a parser.

    An entry point's own record extends this one with its other flags. That
    lets `start_run` read the fields below with their types while it still
    records every flag in the run's config.

    `run_dir` is the folder that contains the run folders, and `data_dir`
    is where the processed 17Lands tensors of `set_code` are cached.
    """

    data_dir: Path
    run_dir: Path
    set_code: str
    held_out_share: float
    seed: int
    jace_commit: str
    manabot_commit: str


@dataclass(frozen=True, slots=True)
class PlayArguments(RunArguments):
    """The flags of an entry point that trains a player on engine games, which `add_play_flags` adds to a parser.

    The games are played by `workers` processes, each running
    `games_per_worker` games at once. A worker that completes no step for
    `stall_seconds` is replaced.

    The player and the guess model make `passes` update passes over each
    block of games, and `normalize` is whether the advantages are scaled to
    mean zero and unit spread. `history` is how many earlier guesses the
    guess model is shown. After each iteration, the iteration's decks are
    blended into the guess model's meta, and every deck already in it counts
    `meta_decay` times what it counted before.

    `play_width`, `play_layers` and `play_heads` give the shape of the
    player, the guess model and the win chance. The player also reads
    `guess_cards` hand-guess tokens and `deck_guess_cards` deck-guess
    tokens.

    Every `evaluate_every` iterations, the player is scored on
    `evaluate_deals` paired deals drawn from `evaluate_decks` held-out human
    decks. `jace.loops.score` reads the model shapes and `history` back from
    a run's config under these names.
    """

    workers: int
    games_per_worker: int
    stall_seconds: float
    evaluate_every: int
    evaluate_deals: int
    evaluate_decks: int
    passes: int
    normalize: bool
    history: int
    meta_decay: float
    play_width: int
    play_layers: int
    play_heads: int
    guess_cards: int
    deck_guess_cards: int


@dataclass(frozen=True, slots=True)
class PlayModels:
    player: PlayerModel
    guess: GuessModel
    # The win chance is used only to train the player. It never plays, so `play_weights` sends no worker its weights.
    win_chance: WinChanceModel
    embedding: CardTable


class NotCardVectorsError(ValueError):
    def __init__(self, run_folder: Path) -> None:
        super().__init__(
            f"the embedding checkpoint of run {run_folder} is not a card table, and the run's config records "
            "no card encoder shape; it was probably saved before the card encoder existed"
        )


def add_play_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=24, help="processes that play games, one thread each")
    parser.add_argument("--games-per-worker", type=int, default=9, help="games each worker plays at once")
    parser.add_argument(
        "--stall-seconds", type=float, default=60.0, help="a worker that completes no step for this long is replaced"
    )
    parser.add_argument("--evaluate-every", type=int, default=10, help="score the player each N iterations")
    parser.add_argument("--evaluate-deals", type=int, default=25, help="deals per opponent; each is four games")
    parser.add_argument("--evaluate-decks", type=int, default=64, help="held-out human decks the deals are drawn from")
    parser.add_argument(
        "--passes",
        type=int,
        default=4,
        help="passes of the player's and the guess model's updates over each block of games",
    )
    parser.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scale advantages to mean zero and unit spread",
    )
    parser.add_argument("--history", type=int, default=4, help="earlier guesses the guess model is shown")
    parser.add_argument(
        "--meta-decay",
        type=float,
        default=0.9,
        help="what a deck behind the guess model's meta counts after each iteration, as a share of what it counted",
    )
    parser.add_argument("--play-width", type=int, default=128)
    parser.add_argument("--play-layers", type=int, default=2)
    parser.add_argument("--play-heads", type=int, default=4)
    parser.add_argument("--guess-cards", type=int, default=8, help="hand-guess tokens the player sees")
    parser.add_argument("--deck-guess-cards", type=int, default=16, help="deck-guess tokens the player sees")


def add_card_flags(parser: argparse.ArgumentParser) -> None:
    # The card encoder's shape. `load_card_table` reads it back from a run's config under these names.
    parser.add_argument("--card-width", type=int, default=64, help="numbers in a card's vector")
    parser.add_argument("--card-layers", type=int, default=1, help="transformer layers that read a card's program")
    parser.add_argument("--card-heads", type=int, default=4, help="attention heads of those layers")


def add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True, help="where the processed 17Lands tensors are cached")
    parser.add_argument("--run-dir", type=Path, required=True, help="the folder that holds run folders")
    parser.add_argument(
        "--set", dest="set_code", default="TLA", help="the card set, already downloaded with `spells add`"
    )
    parser.add_argument("--held-out-share", type=float, default=0.1, help="the share of drafts kept out of training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jace-commit", default="unknown")
    parser.add_argument("--manabot-commit", default="unknown")


def device_named(name: str) -> torch.device:
    # "auto" is the Apple GPU when this machine has one and the CPU otherwise.
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def start_run(
    arguments: RunArguments, *, device: torch.device, inputs: tuple[CheckpointRef, ...], clock: Clock
) -> tuple[Path, Path]:
    """Create a run folder under `arguments.run_dir` and return it with the path of its curve file.

    The run's config is every field of `arguments`, including those of the
    entry point's own record, with paths written as text, plus the device.
    `inputs` are the checkpoints of other runs this run starts from. The
    curve file is `<run id>.curve.json` beside the run folder, and this
    function does not write it.
    """
    config = {name: str(value) if isinstance(value, Path) else value for name, value in asdict(arguments).items()}
    identity = Identity(
        config={**config, "device": device.type},
        seed=arguments.seed,
        jace_commit=arguments.jace_commit,
        manabot_commit=arguments.manabot_commit,
        inputs=inputs,
    )
    folder = create_run(arguments.run_dir, identity, clock=clock)
    return folder, folder.with_name(f"{folder.name}.curve.json")


def playable_decks(decks: Tensor, *, pool: CardPool, count: int) -> list[Deck]:
    # `decks` is uint8 [n, cards] copy counts. Returns the first `count` decks that contain only cards the engine can
    # play, each as a tuple of card ids.
    playable = torch.tensor([bool(pool[card].program) for card in range(len(pool))])
    usable = decks[(decks[:, ~playable] == 0).all(dim=1)][:count]
    return [tuple(torch.repeat_interleave(torch.arange(len(pool)), row.long()).tolist()) for row in usable]


def starting_meta(arguments: RunArguments, cards: SetCards) -> Meta:
    """Fit the meta a run's guess model starts from, to the public human decks of the training split.

    The training split is the 17Lands drafts that are not held out, which
    the drafter also trains on. The scoring harness deals only held-out
    decks, so none of its decks are among these.

    A card that no training deck contains takes its copies from the day-0
    meta of the set's limited format, where a card's copies are in
    proportion to how often the format's boosters contain it. The fitted
    meta treats such a card as independent of every other card. With no
    training decks at all, the result is the day-0 meta.
    """
    human = load_decks(
        read_set(arguments.set_code).games,
        pool=cards.pool,
        cache_dir=arguments.data_dir,
        held_out_share=arguments.held_out_share,
    )
    day_zero = booster_meta(limited_format(arguments.set_code), cards.pool)
    return fit_meta(human.decks[~human.held_out].float(), fallback=day_zero)


def set_cards(data: SetData) -> SetCards:
    # The card set as jace trains on it, with the engine's program for every card the engine can play.
    programs = {card.name: card.program for card in engine_cards()}
    return card_table(data.attributes, data.statistics, programs=programs)


def card_inputs(data: SetData, cards: SetCards, *, data_dir: Path) -> CardInputs:
    """Build what the card encoder reads about every card of `cards`, in card id order.

    The text vectors are cached in `data_dir`. A run that does not find
    them there loads the text model from Hugging Face, which downloads it
    first if this machine has no copy. The program tokens use the engine's
    own vocabulary, so a card's tokens do not depend on the pool. `data`
    must be the set `cards` was built from.
    """
    text = text_vectors(
        card_texts(data.attributes),
        encode=hugging_face_encoder(TEXT_MODEL, device=torch.device("cpu")),
        model_name=TEXT_MODEL,
        cache_dir=data_dir,
    )
    program = program_tokens(cards.pool, vocabulary=program_vocabulary())
    return CardInputs(text=text, program=program, printed=cards.printed)


def load_card_table(run_folder: Path, *, inputs: CardInputs) -> tuple[CardTable, CheckpointRef]:
    """Load the newest card vectors the run at `run_folder` saved, as a table, and the reference of their checkpoint.

    A drafter pretraining run saves a `CardEncoder`, and its config records
    the encoder's shape under the names that `add_card_flags` gives it. For
    such a run, the table is computed once from that encoder on `inputs`,
    without gradients. Every other kind of run saves the `CardTable` it
    read, which is loaded as it is, and `inputs` is not read.

    Raises `NotCardVectorsError` when the run's embedding checkpoint has no
    table and its config records no encoder shape, as for a run saved
    before the card encoder existed.
    """
    reference = latest_checkpoint(run_folder, "embedding")
    config = json.loads((run_folder / "identity.json").read_text())["config"]
    if "card_width" in config:
        encoder = CardEncoder(
            inputs,
            width=int(config["card_width"]),
            program_vocabulary=FIRST_PROGRAM_TOKEN + len(program_vocabulary()),
            program_layers=int(config["card_layers"]),
            heads=int(config["card_heads"]),
        )
        load_checkpoint(run_folder.parent, reference, into=encoder, keeping=None)
        return frozen_table(encoder), reference
    saved = read_checkpoint(run_folder.parent, reference)
    if "table" not in saved:
        raise NotCardVectorsError(run_folder)
    return CardTable(saved["table"]), reference


def build_play_models(arguments: PlayArguments, embedding: CardTable, *, pool: CardPool) -> PlayModels:
    layout = board_layout()
    engine_to_card = engine_card_ids(pool)
    shape = {"width": arguments.play_width, "layers": arguments.play_layers, "heads": arguments.play_heads}
    return PlayModels(
        player=PlayerModel(
            embedding,
            engine_to_card=engine_to_card,
            layout=layout,
            guess_cards=arguments.guess_cards,
            deck_guess_cards=arguments.deck_guess_cards,
            **shape,
        ),
        guess=GuessModel(embedding, engine_to_card=engine_to_card, layout=layout, **shape),
        win_chance=WinChanceModel(embedding, engine_to_card=engine_to_card, layout=layout, **shape),
        embedding=embedding,
    )


def play_weights(models: PlayModels) -> Weights:
    # Every weight a worker plays with, on the CPU, under the names `play_actor` gives its modules.
    modules = {"embedding": models.embedding, "player": models.player, "guess": models.guess}
    return {
        name: {key: Tensor.cpu(value) for key, value in module.state_dict().items()} for name, module in modules.items()
    }


def play_actor(seed: int, *, arguments: PlayArguments, card_width: int) -> Actor:
    """Build the `Actor` that one worker plays with, inside the worker's process.

    The models get the shapes in `arguments`, a card table of zeros
    `card_width` wide, and the weights PyTorch starts any model with. No
    game is played with these values: at the start of every round, the
    worker replaces the weights and the table with the training process's.
    The actor's opponent is a second player, which a round loads with a past
    player's weights under the name "past_player".

    Each round's games start from a new engine seed, drawn from a generator
    seeded with `seed`. The engine reports only the board arrays that the
    models read.
    """
    cards = set_cards(read_set(arguments.set_code))
    models = build_play_models(arguments, CardTable(torch.zeros(len(cards.pool), card_width)), pool=cards.pool)
    rng = random.Random(seed)

    def start(matchups: Sequence[Matchup]) -> GameBatch:
        # Every round needs its own seed, so this cannot be a partial application of `start_batch`.
        # ast-grep-ignore: no-routing-functions
        return start_batch(
            matchups,
            pool=cards.pool,
            games=arguments.games_per_worker,
            seed=rng.getrandbits(_ENGINE_SEED_BITS),
            boards=BOARD_ARRAYS,
        )

    past_player = copy.deepcopy(models.player)
    return Actor(
        start=start,
        policy=models.player.policy,
        opponent=past_player.policy,
        guesser=models.guess.guess,
        modules={
            "embedding": models.embedding,
            "player": models.player,
            "guess": models.guess,
            "past_player": past_player,
        },
        cards=len(cards.pool),
        history=arguments.history,
    )


def scoring_harness(arguments: PlayArguments, cards: SetCards) -> Harness:
    # The `Harness` a run scores its player with. Its deals are drawn from the first `evaluate_decks` held-out human
    # decks that the engine can play.
    data = read_set(arguments.set_code)
    human = load_decks(
        data.games, pool=cards.pool, cache_dir=arguments.data_dir, held_out_share=arguments.held_out_share
    )
    decks = playable_decks(human.decks[human.held_out], pool=cards.pool, count=arguments.evaluate_decks)
    return Harness(
        deals=draw_deals(decks, count=arguments.evaluate_deals, seed=arguments.seed),
        names=tuple(cards.pool[card].name for card in range(len(cards.pool))),
        workers=arguments.workers,
        game_seconds=_GAME_SECONDS,
        resamples=_RANGE_RESAMPLES,
    )


def score_player(
    run: Running, checkpoint: CheckpointRef, *, models: PlayModels, harness: Harness, history: int
) -> dict[str, float]:
    # Scores the player as it is now against random play, scripted greedy and the heuristic player, once
    # sampling its actions and once taking its top action, and records each result under `checkpoint`.
    # Returns the win rates by opponent name, with "_top" after the name for the top action.
    scores = score_against_scripted(
        run,
        checkpoint,
        players={
            action: model_player(models.player, models.guess, history=history, action=action)
            for action in ("sampled", "top")
        },
        harness=harness,
        seed=checkpoint.update,
    )
    suffixes = {"sampled": "", "top": "_top"}
    return {
        f"{name}{suffixes[action]}": score.rate.rate
        for action, by_name in scores.items()
        for name, score in by_name.items()
    }


def land_and_spell_rates(rollout: Rollout) -> dict[str, float]:
    """Measure how often a seat played a land, and cast a spell, in a turn where it could.

    Here a turn groups the decisions that one seat of one game made during
    one turn of that game. The land rate is the share of turns with a legal
    land play in which the seat played a land. The spell rate is the same
    share for casting a spell.

    The rates count turns and not decisions, so a seat that casts a spell
    and then plays its land still counts as having played a land that turn.
    A rate is zero when no turn of the rollout had its action among the
    legal ones.
    """
    turns = torch.stack((rollout.games.double(), rollout.seats.double(), turn_numbers(rollout.boards).double()), dim=1)
    _, turn = turns.unique(dim=0, return_inverse=True)  # int64 [n], the turn each decision belongs to
    per_turn = torch.zeros(int(turn.max()) + 1 if len(turn) else 0)
    rates: dict[str, float] = {}
    for name, offered in zip(("land_rate", "spell_rate"), land_and_spell_actions(rollout.boards), strict=True):
        took = offered.gather(1, rollout.actions[:, None]).squeeze(1)
        # The max over a turn's decisions is 1 when any one of them offered the action, or took it.
        turn_offered = per_turn.scatter_reduce(0, turn, offered.any(dim=-1).float(), reduce="amax")
        turn_took = per_turn.scatter_reduce(0, turn, took.float(), reduce="amax")
        rates[name] = float(turn_took.sum() / turn_offered.sum().clamp_min(1.0))
    return rates

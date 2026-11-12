"""What the command-line entry points of `jace.loops` share, as plain functions.

`pretrain`, `pretrain_decks` and `train_limited` each parse their flags
into a frozen record, start a run folder and write a curve file beside it.
The flags all three take, the record that holds them, the choice of device
and the start of the run live here, so the three agree on what a run
folder and its config look like. This is not a framework: an entry point
calls what it needs and does the rest itself.
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from jace.cards import CardPool, Deck, Rarity
from jace.data import SetCards
from jace.decks import BuildRules
from jace.tracking import CheckpointRef, Clock, Identity, create_run

__all__ = (
    "ADAM_BETAS",
    "RunArguments",
    "add_run_flags",
    "device_named",
    "limited_rules",
    "playable_decks",
    "start_run",
)

# Adam's decay rates for pretraining on human data, as an earlier human-trained drafter set them.
ADAM_BETAS = (0.9, 0.98)

# A limited deck is 40 cards. 99.8% of the human Avatar decks hold 15 to 18 lands.
_DECK_SIZE = 40
_MIN_LANDS = 15
_MAX_LANDS = 18


@dataclass(frozen=True, slots=True)
class RunArguments:
    """The flags every entry point takes, which `add_run_flags` adds to a parser.

    An entry point's own record extends this one with its other flags, so
    `start_run` can read the fields below with their types and still record
    every flag in the run's config. `run_dir` holds run folders, and
    `data_dir` is the cache of processed 17Lands tensors for `set_code`.
    """

    data_dir: Path
    run_dir: Path
    set_code: str
    held_out_share: float
    seed: int
    jace_commit: str
    manabot_commit: str


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
    # decks: uint8 [n, cards]. The first `count` decks whose every card the engine can play.
    playable = torch.tensor([bool(pool[card].program) for card in range(len(pool))])
    usable = decks[(decks[:, ~playable] == 0).all(dim=1)][:count]
    return [tuple(torch.repeat_interleave(torch.arange(len(pool)), row.long()).tolist()) for row in usable]


def limited_rules(cards: SetCards) -> BuildRules:
    is_land = cards.features[:, cards.feature_names.index("type_land")] > 0  # bool [cards]
    basics = cards.pool.of_rarity(Rarity.BASIC)
    return BuildRules(
        size=_DECK_SIZE,
        basics=basics,
        nonbasic_lands=frozenset(is_land.nonzero().squeeze(1).tolist()) - frozenset(basics),
        min_lands=_MIN_LANDS,
        max_lands=_MAX_LANDS,
    )

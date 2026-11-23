"""Shared CLI flags, card-pool loading, and run initialization for pretraining loops.

Pretraining entry points (`pretrain` and `pretrain_decks`) share run-folder
creation, device selection, set card-pool loading, and saved card-table
loading. Keeping these helpers here keeps each pretraining script focused on
its own model and loss.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from jace.cards import FIRST_PROGRAM_TOKEN, CardEncoder, CardPool, CardTable, frozen_table
from jace.data import SetCards, SetData, card_table, program_tokens
from jace.data.catalog import engine_cards, program_vocabulary
from jace.tracking import (
    CheckpointRef,
    Clock,
    Identity,
    create_run,
    latest_checkpoint,
    load_checkpoint,
    read_checkpoint,
)

__all__ = (
    "ADAM_BETAS",
    "NotCardVectorsError",
    "RunArguments",
    "add_card_flags",
    "add_run_flags",
    "device_named",
    "load_card_table",
    "set_cards",
    "start_run",
)

# Adam's decay rates for pretraining on human 17Lands data.
ADAM_BETAS = (0.9, 0.98)


@dataclass(frozen=True, slots=True)
class RunArguments:
    data_dir: Path
    run_dir: Path
    set_code: str
    held_out_share: float
    seed: int
    jace_commit: str
    manabot_commit: str


class NotCardVectorsError(ValueError):
    def __init__(self, run_folder: Path) -> None:
        super().__init__(
            f"the embedding checkpoint of run {run_folder} is not a card table, and the run's config records "
            "no card encoder shape"
        )


def add_card_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--card-width", type=int, default=64, help="numbers in a card's vector")
    parser.add_argument("--card-layers", type=int, default=1, help="transformer layers that read a card's program")
    parser.add_argument("--card-heads", type=int, default=4, help="attention heads of those layers")


def add_run_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, required=True, help="where processed 17Lands tensors are cached")
    parser.add_argument("--run-dir", type=Path, required=True, help="folder containing run folders")
    parser.add_argument("--set", dest="set_code", default="TLA", help="card set downloaded with `spells add`")
    parser.add_argument("--held-out-share", type=float, default=0.1, help="share of drafts kept out of training")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jace-commit", default="unknown")
    parser.add_argument("--manabot-commit", default="unknown")


def device_named(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def start_run(
    arguments: RunArguments, *, device: torch.device, inputs: tuple[CheckpointRef, ...], clock: Clock
) -> tuple[Path, Path]:
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


def set_cards(data: SetData) -> SetCards:
    programs = {card.name: card.program for card in engine_cards()}
    return card_table(data.attributes, data.statistics, programs=programs)


def load_card_table(run_folder: Path, *, pool: CardPool) -> tuple[CardTable, CheckpointRef]:
    reference = latest_checkpoint(run_folder, "embedding")
    config = json.loads((run_folder / "identity.json").read_text())["config"]
    if "card_width" in config:
        vocabulary = program_vocabulary()
        encoder = CardEncoder(
            program_tokens(pool, vocabulary=vocabulary),
            width=int(config["card_width"]),
            program_vocabulary=FIRST_PROGRAM_TOKEN + len(vocabulary),
            program_layers=int(config["card_layers"]),
            heads=int(config["card_heads"]),
        )
        load_checkpoint(run_folder.parent, reference, into=encoder, keeping=None)
        return frozen_table(encoder), reference
    saved = read_checkpoint(run_folder.parent, reference)
    if "table" not in saved:
        raise NotCardVectorsError(run_folder)
    return CardTable(saved["table"]), reference

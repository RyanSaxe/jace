"""Check the card text writer, the text vectors and the program vectors through `jace.data`.

No test downloads or runs a language model. The encoder is a small function
that turns a text into two numbers and counts its calls, which is enough to
show what the cache does. The cards are a hand-made table with the columns
`read_set` returns, and the cache files go to a temporary folder.
"""

from collections.abc import Sequence
from pathlib import Path

import polars as pl
import torch
from torch import Tensor

from jace.cards import CardPool, CardRecord, Rarity
from jace.data import card_texts, program_vectors, text_vectors

__all__ = ()


def _cards() -> pl.DataFrame:
    creature = {
        "name": "Aang, the Last Airbender",
        "mana_cost": "{3}{W}",
        "card_type": "Creature",
        "subtype": "Human Avatar Ally",
        "power": "3",
        "toughness": "2",
        "rarity": "uncommon",
        "oracle_text": "Flying\nWhen Aang enters, draw a card. Aang, the Last Airbender gains lifelink.",
    }
    land = {
        "name": "Ba Sing Se",
        "mana_cost": "",
        "card_type": "Land",
        "subtype": "",
        "power": None,
        "toughness": None,
        "rarity": "rare",
        "oracle_text": "{T}: Add {G}.",
    }
    return pl.DataFrame([creature, land])


def test_a_card_is_written_as_its_printed_lines_with_its_own_name_replaced() -> None:
    creature, land = card_texts(_cards())

    assert creature == (
        "Mana cost: {3}{W}\n"
        "Type: Creature - Human Avatar Ally\n"
        "Power/toughness: 3/2\n"
        "Rarity: uncommon\n"
        "Flying\n"
        "When this card enters, draw a card. this card gains lifelink."
    )
    assert land == "Type: Land\nRarity: rare\n{T}: Add {G}."


def test_text_vectors_are_standardised_and_the_encoder_runs_once_until_a_text_changes(tmp_path: Path) -> None:
    calls: list[int] = []

    def encode(texts: Sequence[str]) -> Tensor:
        calls.append(len(texts))
        return torch.tensor([[float(len(text)), 7.0] for text in texts])

    first = text_vectors(["a", "bbb", "ccccc"], encode=encode, model_name="fake", cache_dir=tmp_path)
    again = text_vectors(["a", "bbb", "ccccc"], encode=encode, model_name="fake", cache_dir=tmp_path)
    text_vectors(["a", "bbb", "cc"], encode=encode, model_name="fake", cache_dir=tmp_path)

    assert torch.allclose(first[:, 0], torch.tensor([-1.0, 0.0, 1.0]) * 1.5**0.5)
    assert first[:, 1].eq(0).all()
    assert torch.equal(first, again)
    assert calls == [3, 3]


def test_program_vectors_match_for_equal_programs_and_mark_a_card_without_one() -> None:
    programs = {
        "Ant": (1, 5, 2, 9, 1, 7),
        "Bat": (1, 5, 2, 9, 1, 7),
        "Cat": (1, 5, 3, 4),
        "Dog": (),
        "Eel": (2, 9, 4, 1),
    }
    records = [
        CardRecord(name=name, set_code="TST", rarity=Rarity.COMMON, text="", program=program)
        for name, program in programs.items()
    ]

    ant, bat, cat, dog, _ = program_vectors(CardPool(records), width=2)

    assert torch.allclose(ant, bat, atol=1e-5)
    assert not torch.allclose(ant, cat, atol=1e-5)
    assert dog.tolist() == [0.0, 0.0, 1.0]
    assert ant[-1] == 0

"""Read card definitions and engine program tokens from managym's catalog.

This is the one module in `jace.data` allowed to import `managym`. It exposes
each playable card's printed name, set code, Scryfall id, rules text, and
tokenized engine program so that `jace.data.card_table` and `jace.cards` can
build shared card inputs without depending on `managym` directly.
"""

import json
from dataclasses import dataclass

import managym

__all__ = (
    "EngineCard",
    "UnreadableDefinitionsError",
    "engine_cards",
    "program_vocabulary",
)


@dataclass(frozen=True, slots=True)
class EngineCard:
    name: str
    set_code: str
    scryfall_id: str
    text: str
    program: tuple[int, ...]


class UnreadableDefinitionsError(TypeError):
    def __init__(self) -> None:
        super().__init__("the engine's card definitions JSON is not a list of objects with a name and a text_box")


def _rules_texts() -> dict[str, str]:
    raw = json.loads(managym.card_definitions_json())
    if not isinstance(raw, list):
        raise UnreadableDefinitionsError
    texts: dict[str, str] = {}
    for item in raw:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("text_box"), str)
        ):
            raise UnreadableDefinitionsError
        texts[item["name"]] = item["text_box"]
    return texts


def engine_cards() -> tuple[EngineCard, ...]:
    """List every deck-eligible card in engine definition order.

    Token cards, back faces, and rules fixtures are omitted because no deck may
    contain them. Raises `UnreadableDefinitionsError` if `managym`'s card
    definitions JSON does not have the expected schema.
    """
    catalog = managym.catalog_tokens()
    kinds, values, offsets = catalog["token_kind"], catalog["token_value"], catalog["definition_offsets"]
    texts = _rules_texts()
    definitions = {row["card_def_id"]: row for row in managym.content_pool(managym.WORLD_SETUP)["definitions"]}
    cards: list[EngineCard] = []
    for definition_id in sorted(definitions):
        definition = definitions[definition_id]
        key, scryfall_id = definition["key"], definition["scryfall_id"]
        if definition["eligibility"] != "eligible" or key is None or scryfall_id is None:
            continue
        tokens = range(offsets[definition_id], offsets[definition_id + 1])
        cards.append(
            EngineCard(
                name=definition["registry_name"],
                set_code=key.split(".", maxsplit=1)[0].upper(),
                scryfall_id=scryfall_id,
                text=texts[definition["registry_name"]],
                program=tuple(part for token in tokens for part in (kinds[token], values[token])),
            )
        )
    return tuple(cards)


def program_vocabulary() -> dict[tuple[int, int], int]:
    catalog = managym.catalog_tokens()
    pairs = sorted(set(zip(catalog["token_kind"], catalog["token_value"], strict=True)))
    return {pair: number for number, pair in enumerate(pairs)}

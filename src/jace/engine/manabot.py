"""`GameBatch` on manabot's engine, and the engine's card catalog as plain data.

This is the only module that imports manabot, so a change in manabot's API is
a change to this file alone. Decklists are closed, which means a seat's board
never shows the opponent's deck and the guess model has to infer it. Boards
are encoded at policy input version 8 because versions 6 and 7 refuse boards
that hold certain continuous effects. Version 8 adds three arrays and leaves
unchanged every array `jace.play.tokens` reads. The engine has no card rarity,
so `engine_cards` cannot build a `CardPool` without data from another source.

Games run in manabot's `VectorEnv`, which steps every slot in one native call
and writes the boards into arrays this module owns. It reports hidden hands
but not libraries, so `hidden_cards` works a library out as the deck minus the
hand and minus the cards the acting seat's board shows in other zones. Two
cases make a library wrong by one card. The board hides a card the other seat
foretold, so that card still counts as in its library. The board files a
permanent under its controller, and when both decks hold that card the owner
cannot be told from the controller.
"""

import collections
import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from manabot import managym
from torch import Tensor

from jace.cards import CardPool, Deck
from jace.engine.batch import SEATS, Decisions, Finished, GameBatch, HiddenCards, Outcome

__all__ = (
    "POLICY_INPUT_VERSION",
    "DeckRejectedError",
    "EngineCard",
    "UnreadableDefinitionsError",
    "engine_card_ids",
    "engine_cards",
    "start_batch",
)

POLICY_INPUT_VERSION = 8

# Columns of a row of the `agent_cards` and `opponent_cards` boards. The first seven are one flag per zone.
_LIBRARY_COLUMN = int(managym.ZoneEnum.LIBRARY)
_HAND_COLUMN = int(managym.ZoneEnum.HAND)
_TOKEN_COLUMN = 29
# The step results the engine writes beside the boards, which no player reads.
_STEP_RESULTS = ("rewards", "terminated", "truncated")


@dataclass(frozen=True, slots=True)
class EngineCard:
    name: str
    # The upper-case prefix of the engine's card key, for example "TLA".
    set_code: str
    scryfall_id: str
    text: str
    # The card's learning tokens as alternating kind and value, so its length is twice the token count.
    program: tuple[int, ...]


class DeckRejectedError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"the engine rejected a deck: {reason}")


class UnreadableDefinitionsError(TypeError):
    def __init__(self) -> None:
        super().__init__("the engine's card definitions JSON is not a list of objects with a name and a text_box")


def engine_cards() -> tuple[EngineCard, ...]:
    """List every card the engine accepts in a deck, in engine definition order.

    The engine supplies a printed name, a set key, a Scryfall id, rules text
    and learning tokens. It has no rarity, so the caller joins rarity from
    another source before building a `CardPool`. `jace.data.card_table`
    joins by printed name; `scryfall_id` is here for a caller that needs an
    exact printing.
    Tokens, back faces and rules fixtures are left out because no deck may
    hold them.
    """
    catalog = managym.learning_catalog()
    kinds, values, offsets = catalog["token_kind"], catalog["token_value"], catalog["definition_offsets"]
    texts = _rules_texts()
    cards: list[EngineCard] = []
    definitions = {row["card_def_id"]: row for row in managym.content_pool(managym.WORLD_SETUP)["definitions"]}
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


def engine_card_ids(pool: CardPool) -> Tensor:
    # Returns int64 [engine definition ids]: the `CardId` of each engine definition, or -1 when the pool lacks it.
    names = {pool[card].name: card for card in range(len(pool))}
    card_ids = torch.full((managym.CARD_DEFINITION_COUNT,), -1, dtype=torch.int64)
    for definition in managym.content_pool(managym.WORLD_SETUP)["definitions"]:
        card_ids[definition["card_def_id"]] = names.get(definition["registry_name"], -1)
    return card_ids


def start_batch(matchups: Callable[[], tuple[Deck, Deck]], *, pool: CardPool, games: int, seed: int) -> GameBatch:
    names = tuple(pool[card].name for card in range(len(pool)))
    return _ManabotBatch(matchups, names=names, card_ids=engine_card_ids(pool).numpy(), games=games, seed=seed)


def _rules_texts() -> dict[str, str]:
    # Printed name -> rules text. The definitions JSON is the only place the engine publishes rules text.
    texts: dict[str, str] = {}
    definitions: object = json.loads(managym.card_definitions_json())
    if not isinstance(definitions, list):
        raise UnreadableDefinitionsError
    for definition in definitions:
        if not isinstance(definition, dict):
            raise UnreadableDefinitionsError
        name, text = definition.get("name"), definition.get("text_box")
        if not isinstance(name, str) or not isinstance(text, str):
            raise UnreadableDefinitionsError
        texts[name] = text
    return texts


class _ManabotBatch:
    def __init__(
        self,
        matchups: Callable[[], tuple[Deck, Deck]],
        *,
        names: tuple[str, ...],
        card_ids: npt.NDArray[np.int64],
        games: int,
        seed: int,
    ) -> None:
        self._matchups: Callable[[], tuple[Deck, Deck]] = matchups
        self._names: tuple[str, ...] = names
        self._cards_by_name: dict[str, int] = {name: card for card, name in enumerate(names)}
        # card_ids: int64 [engine definition ids], the `CardId` of each engine definition or -1.
        self._card_ids: npt.NDArray[np.int64] = card_ids
        self._decks: list[tuple[Deck, Deck]] = [matchups() for _ in range(games)]
        # deck_counts: int64 [n, 2, cards], copies of each card in each seat's deck.
        self._deck_counts: npt.NDArray[np.int64] = np.zeros((games, SEATS, len(names)), dtype=np.int64)
        # The engine gives slot i the seeds base + i, base + i + n, ... for its successive games, so a
        # slot's games do not change with how fast the other slots finish. The base is hashed so that
        # batches started from neighbouring seeds share no game, and halved so the additions cannot overflow.
        base = int(np.random.SeedSequence(seed).generate_state(1, np.uint64)[0]) >> 1
        self._games: managym.VectorEnv = managym.VectorEnv(games, seed=base, opponent_policy="none")
        setups = [self._setup(slot) for slot in range(games)]
        # The engine sizes its batched boards from the arrays it is given, so one scalar game supplies their shapes.
        template = managym.Env(seed=base)
        try:
            observation, _ = template.reset(setups[0])
            shapes = template.encode_observation(observation, POLICY_INPUT_VERSION)
            self._buffers: dict[str, npt.NDArray[Any]] = {
                name: np.zeros((games, *board.shape), dtype=board.dtype) for name, board in shapes.items()
            }
            self._buffers.update(
                rewards=np.zeros(games, dtype=np.float64),
                terminated=np.zeros(games, dtype=np.uint8),
                truncated=np.zeros(games, dtype=np.uint8),
            )
            self._games.set_buffers(self._buffers)
            self._games.reset_each_into_buffers(setups)
        except managym.AgentError as error:
            raise DeckRejectedError(str(error)) from error

    def decisions(self) -> Decisions:
        # Copies, so a caller may keep a board after later steps overwrite the engine's arrays.
        boards = {name: board.copy() for name, board in self._buffers.items() if name not in _STEP_RESULTS}
        return Decisions(
            seats=np.array(self._games.current_agent_indices(), dtype=np.int64),
            boards=boards,
            action_counts=self._buffers["actions_valid"].sum(axis=1).astype(np.int64),
        )

    def act(self, actions: npt.NDArray[np.int64]) -> tuple[Finished, ...]:
        slots = len(self._decks)
        # An isolated step reports a game the engine cannot continue, such as a board too large
        # for the observation arrays, in that slot's info and lets every other game advance.
        self._games.step_isolated_into_buffers(actions.tolist(), [True] * slots, recover_capacity=True)
        results = self._games.get_last_info()
        failed = ["game_error" in result for result in results]
        ended = self._buffers["terminated"] | self._buffers["truncated"] | np.array(failed)
        if not ended.any():
            return ()
        if any(failed):
            self._games.replace_failed_into_buffers(failed)
        # The engine has already dealt each ended slot a new game from the old decks. The next
        # matchup is dealt over it below.
        finished: list[Finished] = []
        setups: list[list[managym.PlayerConfig] | None] = [None] * slots
        for slot in np.flatnonzero(ended).tolist():
            winner = results[slot].get("winner_index")
            outcome = Outcome.ABANDONED if failed[slot] else _outcome(winner if isinstance(winner, int) else None)
            finished.append(Finished(slot=slot, outcome=outcome, decks=self._decks[slot]))
            self._decks[slot] = self._matchups()
            setups[slot] = self._setup(slot)
        try:
            self._games.reassign_roots_into_buffers(setups)
        except managym.AgentError as error:
            raise DeckRejectedError(str(error)) from error
        return tuple(finished)

    def hidden_cards(self) -> HiddenCards:
        games = len(self._decks)
        # The engine returns each hand as a name-to-copies mapping, so this one loop over games is unavoidable.
        held = [
            (slot, seat, self._cards_by_name[name], copies)
            for seat in range(SEATS)
            for slot, hand in enumerate(self._games.hand_snapshots([seat] * games))
            for name, copies in hand.cards.items()
            if name in self._cards_by_name
        ]
        slots, seats, cards, copies = np.array(held, dtype=np.int64).reshape(-1, 4).T
        hands = np.zeros_like(self._deck_counts)
        hands[slots, seats, cards] = copies
        public = self._cards_in_public_zones()
        # The board files a permanent under its controller. A side that shows more copies of a card
        # than its seat's deck holds must control the other seat's permanents, so the extra copies move back.
        borrowed = (hands + public - self._deck_counts).clip(min=0)
        libraries = self._deck_counts - hands - public + borrowed - borrowed[:, ::-1]
        return HiddenCards(hands=hands, libraries=libraries.clip(min=0))

    def _cards_in_public_zones(self) -> npt.NDArray[np.int64]:
        # Returns int64 [n, 2, cards]: copies of each card that the acting seat's board lists on each
        # seat's side outside the hand and the library, which is the battlefield, graveyard, exile and stack.
        counts = np.zeros_like(self._deck_counts)
        acting = np.array(self._games.current_agent_indices(), dtype=np.int64)
        # definitions: int64 [n, 2, card rows], the engine definition id of each listed card, acting seat first.
        definitions = self._buffers["semantic_cards"].astype(np.int64) - 1
        for side, name in enumerate(("agent_cards", "opponent_cards")):
            rows = self._buffers[name]  # float32 [n, card rows, features]
            # The valid mask also drops the sideboard cards that `semantic_cards` lists after the acting seat's cards.
            hidden_zone_or_token = rows[:, :, [_LIBRARY_COLUMN, _HAND_COLUMN, _TOKEN_COLUMN]].any(axis=-1)
            slots, card_rows = np.nonzero((self._buffers[f"{name}_valid"] > 0) & ~hidden_zone_or_token)
            cards = self._card_ids[definitions[slots, side, card_rows]]
            in_pool = cards >= 0
            seats = np.abs(side - acting[slots])
            np.add.at(counts, (slots[in_pool], seats[in_pool], cards[in_pool]), 1)
        return counts

    def _setup(self, slot: int) -> list[managym.PlayerConfig]:
        players: list[managym.PlayerConfig] = []
        for seat, deck in enumerate(self._decks[slot]):
            copies = collections.Counter(self._names[card] for card in deck)
            players.append(
                managym.PlayerConfig(f"seat {seat}", dict(copies), world=managym.WORLD_SETUP, disclosure="closed")
            )
            self._deck_counts[slot, seat] = np.bincount(deck, minlength=len(self._names))
        return players


def _outcome(winner: int | None) -> Outcome:
    if winner is None:
        return Outcome.DRAW
    return Outcome.SEAT_1_WON if winner else Outcome.SEAT_0_WON

"""`GameBatch` on manabot's engine, and the engine's card catalog as plain data.

This is the only module that imports manabot, so a change in manabot's API is
a change to this file alone. Decklists are closed, which means a seat's board
never shows the opponent's deck and the guess model has to infer it. Boards
are encoded at policy input version 8 because versions 6 and 7 refuse boards
that hold certain continuous effects. Version 8 adds three arrays and leaves
unchanged every array `jace.play.tokens` reads. The engine has no card rarity,
so `engine_cards` cannot build a `CardPool` without data from another source.
"""

import collections
import json
from collections.abc import Callable
from dataclasses import dataclass

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
    and learning tokens. It has no rarity, so the caller must join rarity
    from Scryfall or 17Lands by `scryfall_id` before building a `CardPool`.
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
    # Temporary: one scalar `Env` per slot, stepped in a Python loop, because only the scalar `Env`
    # can report both seats' hands. manabot's author is adding hidden hands to the batched env. When
    # that lands, switch this class to `VectorEnv` with `set_buffers` and `step_into_buffers`.

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
        # card_ids: int64 [engine definition ids], the `CardId` of each engine definition or -1.
        self._card_ids: npt.NDArray[np.int64] = card_ids
        self._seed: int = seed
        self._envs: list[managym.Env] = []
        self._decks: list[tuple[Deck, Deck]] = []
        self._games_started: list[int] = [0] * games
        self._seats: npt.NDArray[np.int64] = np.zeros(games, dtype=np.int64)
        self._action_counts: npt.NDArray[np.int64] = np.zeros(games, dtype=np.int64)
        # deck_counts: int64 [n, 2, cards], copies of each card in each seat's deck.
        self._deck_counts: npt.NDArray[np.int64] = np.zeros((games, SEATS, len(names)), dtype=np.int64)
        self._boards: dict[str, npt.NDArray[np.generic]] = {}
        for slot in range(games):
            decks = matchups()
            env, observation = self._deal(slot, decks=decks)
            self._envs.append(env)
            self._decks.append(decks)
            if not self._boards:
                first = env.encode_observation(observation, POLICY_INPUT_VERSION)
                self._boards = {
                    name: np.zeros((games, *board.shape), dtype=board.dtype) for name, board in first.items()
                }
            self._record(slot, observation=observation)

    def decisions(self) -> Decisions:
        # Copies, so a caller may keep a board after later steps overwrite the slot's row.
        boards = {name: board.copy() for name, board in self._boards.items()}
        return Decisions(seats=self._seats.copy(), boards=boards, action_counts=self._action_counts.copy())

    def act(self, actions: npt.NDArray[np.int64]) -> tuple[Finished, ...]:
        finished: list[Finished] = []
        for slot, action in enumerate(actions.tolist()):
            env = self._envs[slot]
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                finished.append(Finished(slot=slot, outcome=_outcome(env.winner_index()), decks=self._decks[slot]))
                self._decks[slot] = self._matchups()
                self._envs[slot], observation = self._deal(slot, decks=self._decks[slot])
            self._record(slot, observation=observation)
        return tuple(finished)

    def hidden_cards(self) -> HiddenCards:
        hands = np.zeros_like(self._deck_counts)
        outside_library = np.zeros_like(self._deck_counts)
        for slot, env in enumerate(self._envs):
            for seat in range(SEATS):
                view = env.observation_for_player(seat)
                hands[slot, seat], outside_library[slot, seat] = self._own_card_counts(view)
        # A seat's own view lists its cards in every zone but the library, so the library is the rest of its deck.
        return HiddenCards(hands=hands, libraries=(self._deck_counts - outside_library).clip(min=0))

    def _own_card_counts(self, view: managym.Observation) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        # Returns (hand, outside_library), both int64 [cards]: copies of each card the viewing seat
        # owns in its hand, and in any zone but its library.
        hand = np.zeros(len(self._names), dtype=np.int64)
        outside_library = np.zeros(len(self._names), dtype=np.int64)
        for card in (*view.agent_cards, *view.opponent_cards):
            card_id = int(self._card_ids[card.registry_key])
            if card.owner_id != view.agent.id or card.is_token or card_id < 0:
                continue
            if card.zone == managym.ZoneEnum.HAND:
                hand[card_id] += 1
            if card.zone != managym.ZoneEnum.LIBRARY:
                outside_library[card_id] += 1
        return hand, outside_library

    def _deal(self, slot: int, *, decks: tuple[Deck, Deck]) -> tuple[managym.Env, managym.Observation]:
        # Each game's seed depends only on the batch seed, its slot and how many games the slot has
        # started, so a slot's games do not change with how fast the other slots finish.
        entropy = (self._seed, slot, self._games_started[slot])
        self._games_started[slot] += 1
        env = managym.Env(seed=int(np.random.SeedSequence(entropy).generate_state(1, np.uint64)[0]))
        players: list[managym.PlayerConfig] = []
        for seat in range(SEATS):
            copies = collections.Counter(self._names[card] for card in decks[seat])
            players.append(
                managym.PlayerConfig(f"seat {seat}", dict(copies), world=managym.WORLD_SETUP, disclosure="closed")
            )
            self._deck_counts[slot, seat] = np.bincount(decks[seat], minlength=len(self._names))
        try:
            observation, _ = env.reset(players)
        except managym.AgentError as error:
            raise DeckRejectedError(str(error)) from error
        return env, observation

    def _record(self, slot: int, *, observation: managym.Observation) -> None:
        self._seats[slot] = observation.agent.player_index
        self._action_counts[slot] = len(observation.action_space.actions)
        for name, board in self._envs[slot].encode_observation(observation, POLICY_INPUT_VERSION).items():
            self._boards[name][slot] = board


def _outcome(winner: int | None) -> Outcome:
    if winner is None:
        return Outcome.DRAW
    return Outcome.SEAT_1_WON if winner else Outcome.SEAT_0_WON

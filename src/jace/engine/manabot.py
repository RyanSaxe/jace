"""`GameBatch` on manabot's engine, and the engine's card catalog as plain data.

Only this module and `jace.engine.scoring` import manabot or its engine,
`managym`. manabot provides the `ObservationSpace` that describes the sizes
of the board arrays. A change in either package therefore affects only these
two files.

Decklists are closed. A seat's board never shows the opponent's deck, so the
guess model has to infer it.

`board_layout` reads the engine's own description of the board arrays and
checks it against what `jace.play.tokens` assumes. A change of layout
therefore raises an error when a model is built, not while it plays.

Games run in the engine's `VectorEnv`. It has a fixed number of slots, each
a place for one game in play, and it steps every slot in one native call.
It writes the boards into arrays that this module allocates.

The engine reports each seat's hidden hand but not its library. So
`hidden_cards` works out a library as the deck, minus the hand, minus the
cards that the acting seat's board shows in other zones. Two cases make a
library wrong by one card:

- The acting seat's board hides a card that the other seat foretold, so
  that card still counts as in the other seat's library.
- The board lists a permanent under its controller, not its owner. When the
  controller's deck also contains that card, `hidden_cards` cannot always
  tell the owner from the controller.
"""

import collections
import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import managym
import numpy as np
import numpy.typing as npt
import torch
from manabot.env.observation import ObservationSpace
from manabot.infra.config import ObservationConfig
from torch import Tensor

from jace.cards import CardPool
from jace.engine.batch import SEATS, BoardLayout, Decisions, Finished, GameBatch, HiddenCards, Matchup, Outcome

__all__ = (
    "POLICY_INPUT_VERSION",
    "BoardLayoutError",
    "DeckRejectedError",
    "EngineCard",
    "UnreadableDefinitionsError",
    "board_layout",
    "engine_card_ids",
    "engine_cards",
    "land_and_spell_actions",
    "observation_space",
    "program_vocabulary",
    "start_batch",
    "turn_numbers",
)

# Version 9 of the board encoding has no place for what a player did earlier in the turn, such as the life
# they lost. Boards with Avatar: The Last Airbender cards need it, so the engine refuses to encode them at version 9.
POLICY_INPUT_VERSION = 10
# How many cards and permanents of one seat a board has room for. The engine's defaults are 60 and
# 40, which Avatar games outgrow. A game that outgrows these larger sizes too is abandoned.
_MAX_CARDS = 80
_MAX_PERMANENTS = 64

# The engine's description of every board array: its axes, and the position of each named column.
_ARRAYS = managym.observation_layout(turn_state=True)
# Columns of a row of the `agent_cards` and `opponent_cards` boards.
_LIBRARY_COLUMN = int(_ARRAYS["agent_cards"]["columns"]["zone_library"]["index"])
_HAND_COLUMN = int(_ARRAYS["agent_cards"]["columns"]["zone_hand"]["index"])
_TOKEN_COLUMN = int(_ARRAYS["agent_cards"]["columns"]["is_token"]["index"])
# Columns of a row of the `actions` board.
_PLAY_LAND_COLUMN = int(_ARRAYS["actions"]["columns"]["play_land"]["index"])
_CAST_SPELL_COLUMN = int(_ARRAYS["actions"]["columns"]["cast_spell"]["index"])
# The column of a player's row that contains the number of the turn being played.
_TURN_COLUMN = int(_ARRAYS["agent_player"]["columns"]["turn"]["index"])
# What `jace.play.tokens` assumes about the arrays, in the engine's own words.
_DEFINITION_ENCODING = "CardDefId + 1; 0 means absent"
_FOCUS_ORDER = "agent player, opponent player, agent cards, opponent cards, agent permanents, opponent permanents"
# The engine marks a column that names something, rather than counting it, as kind "id" or "reference".
# The encodings below are the ones such columns have in the arrays that `jace.play.tokens` reads.
_NAMING_KINDS = ("id", "reference")
_LINK_ENCODING = f"Padded object table index + 1; 0 absent. Order: {_FOCUS_ORDER}"
_SELECTOR_ENCODING = "Zero-based selector + 1; 0 absent"
_SEAT_ENCODING = "Raw player seat; -1 absent"
_EVENT_OBJECT_ENCODING = (
    "Raw event entity id in the accompanying kind namespace; -1 absent; not a padded object-table index"
)
# An action's continuation has the selectors' encoding, but it numbers a frame of the `relations` array, which
# jace does not read, so `board_layout` leaves it out of the action categories. manabot's own model joins it to
# its encoding of that frame.
_ACTION_FRAME_COLUMN = int(_ARRAYS["actions"]["columns"]["continuation"]["index"])
# In a player's row, the size of each zone is divided by this.
_ZONE_COUNT_SCALE = 60
_ZONE_COUNT_ENCODING = f"Value / {_ZONE_COUNT_SCALE}"


@dataclass(frozen=True, slots=True)
class EngineCard:
    name: str
    # The upper-case prefix of the engine's card key, for example "TLA".
    set_code: str
    scryfall_id: str
    text: str
    # The card's rules text as the engine tokenizes it: (kind, value) pairs, flattened so kind and value alternate.
    program: tuple[int, ...]


class DeckRejectedError(ValueError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"the engine rejected a deck: {reason}")


class BoardLayoutError(RuntimeError):
    def __init__(self, changed: Sequence[str]) -> None:
        super().__init__(
            "the engine's board arrays no longer match what jace.play.tokens assumes about: " + ", ".join(changed)
        )


class UnreadableDefinitionsError(TypeError):
    def __init__(self) -> None:
        super().__init__("the engine's card definitions JSON is not a list of objects with a name and a text_box")


def engine_cards() -> tuple[EngineCard, ...]:
    """List every card the engine accepts in a deck, in engine definition order.

    Token cards, back faces and rules fixtures are left out, because no deck
    may contain them.

    The engine has no card rarity, so the caller joins rarity from another
    source before building a `CardPool`. `jace.data.card_table` joins by
    printed name. `scryfall_id` is there for a caller that needs an exact
    printing.

    Raises `UnreadableDefinitionsError` when the engine's card definitions
    JSON is not a list of objects with a name and a text_box.
    """
    catalog = managym.catalog_tokens()
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


def program_vocabulary() -> dict[tuple[int, int], int]:
    # Every distinct (kind, value) pair in the engine's card programs, numbered from 0 in sorted order.
    # It is read from the whole catalog, so a pair's number does not depend on which cards a pool contains.
    catalog = managym.catalog_tokens()
    pairs = sorted(set(zip(catalog["token_kind"], catalog["token_value"], strict=True)))
    return {pair: number for number, pair in enumerate(pairs)}


def engine_card_ids(pool: CardPool) -> Tensor:
    # Returns int64 [managym.CARD_DEFINITION_COUNT], indexed by engine definition id. Each entry is the
    # `CardId` of the pool's card with that definition's name, or -1 when the pool has no such card.
    names = {pool[card].name: card for card in range(len(pool))}
    card_ids = torch.full((managym.CARD_DEFINITION_COUNT,), -1, dtype=torch.int64)
    for definition in managym.content_pool(managym.WORLD_SETUP)["definitions"]:
        card_ids[definition["card_def_id"]] = names.get(definition["registry_name"], -1)
    return card_ids


def observation_space() -> ObservationSpace:
    # manabot's description of every board array at this module's version and sizes. The engine encodes boards by it
    # here and in `jace.engine.scoring`, so a model reads the same arrays in training and in scoring.
    config = ObservationConfig(
        policy_input_version=POLICY_INPUT_VERSION,
        max_cards_per_player=_MAX_CARDS,
        max_permanents_per_player=_MAX_PERMANENTS,
    )
    return ObservationSpace(config)


def board_layout() -> BoardLayout:
    """Describe the boards for `jace.play`, after checking them against what it assumes.

    `jace.play.tokens` assumes these things about the engine's arrays:

    - The acting seat's and the opponent's arrays have the same columns, and
      the engine numbers an action's objects in the order that
      `jace.play.tokens` joins them.
    - A card's identity in `semantic_cards` is its definition id plus one.
    - The engine can append each seat's turn state to its player row, and
      each card's turn state to its card row.
    - Every column of a player's row and of a card's row is a number.
    - The only columns of permanent, action and event rows that name
      something are the expected ones, with the expected encodings.
    - The decklist and known-hand rows have the expected axes and columns.
    - A player's hand and library sizes are divided by `_ZONE_COUNT_SCALE`.

    Raises `BoardLayoutError`, naming each assumption that fails, when the
    engine's description of its arrays no longer fits one of them.
    """
    columns = {name: list(array["columns"]) for name, array in _ARRAYS.items()}
    permanent_names, action_names, event_names = (
        _naming_columns(name) for name in ("agent_permanents", "actions", "events")
    )
    decklist = _naming_columns("decklists")
    player_columns = _ARRAYS["agent_player"]["columns"]
    holds = {
        "the columns of the two seats' players": columns["agent_player"] == columns["opponent_player"],
        "the columns of the two seats' cards": columns["agent_cards"] == columns["opponent_cards"],
        "the columns of the two seats' permanents": columns["agent_permanents"] == columns["opponent_permanents"],
        "the order of action_focus": str(_ARRAYS["action_focus"]["element"]["encoding"]).endswith(_FOCUS_ORDER),
        "the ids in semantic_cards": _ARRAYS["semantic_cards"]["element"]["encoding"] == _DEFINITION_ENCODING,
        "the axes of semantic_cards": _ARRAYS["semantic_cards"]["axes"] == ["viewer_relative_seat", "card"],
        "the axes of player_turn_state": _ARRAYS["player_turn_state"]["axes"] == ["viewer_relative_seat", "feature"],
        "the axes of card_turn_state": _ARRAYS["card_turn_state"]["axes"] == ["viewer_relative_seat", "card"],
        "the columns that name something in players and cards": not any(
            _naming_columns(name) for name in ("agent_player", "player_turn_state", "agent_cards")
        ),
        "the columns that name something in permanents": set(permanent_names) == {_DEFINITION_ENCODING, _LINK_ENCODING}
        and len(permanent_names[_DEFINITION_ENCODING]) == 1,
        "the columns that name something in actions": set(action_names) == {_SELECTOR_ENCODING},
        "the columns that name something in events": set(event_names) == {_SEAT_ENCODING, _EVENT_OBJECT_ENCODING},
        "the axes of decklists": _ARRAYS["decklists"]["axes"] == ["viewer_relative_seat", "definition", "feature"],
        "the columns of decklists": set(decklist) == {_DEFINITION_ENCODING}
        and {"definition", "unseen", "main"} <= set(columns["decklists"]),
        "the axes of known_hand": _ARRAYS["known_hand"]["axes"] == ["viewer_relative_seat", "definition", "feature"],
        "the columns of known_hand": set(_naming_columns("known_hand")) == {_DEFINITION_ENCODING}
        and {"definition", "count"} <= set(columns["known_hand"]),
        "the hand and library counts of players": all(
            player_columns[name]["encoding"] == _ZONE_COUNT_ENCODING for name in ("hand_count", "library_count")
        ),
    }
    changed = [name for name, held in holds.items() if not held]
    if changed:
        raise BoardLayoutError(changed)
    decklist_columns = _ARRAYS["decklists"]["columns"]
    known_hand_columns = _ARRAYS["known_hand"]["columns"]
    return BoardLayout(
        player_features=len(columns["agent_player"]) + len(columns["player_turn_state"]),
        card_features=len(columns["agent_cards"]) + 1,
        permanent_features=_number_columns("agent_permanents"),
        action_features=_number_columns("actions"),
        event_features=_number_columns("events"),
        permanent_definition=permanent_names[_DEFINITION_ENCODING][0],
        permanent_links=permanent_names[_LINK_ENCODING],
        action_categories=tuple(
            column for column in action_names[_SELECTOR_ENCODING] if column != _ACTION_FRAME_COLUMN
        ),
        event_categories=event_names[_SEAT_ENCODING],
        decklist_definition=decklist[_DEFINITION_ENCODING][0],
        decklist_copies=(int(decklist_columns["unseen"]["index"]), int(decklist_columns["main"]["index"])),
        card_unshown_flags=(_LIBRARY_COLUMN, _HAND_COLUMN, _TOKEN_COLUMN),
        hand_count=int(player_columns["hand_count"]["index"]),
        library_count=int(player_columns["library_count"]["index"]),
        zone_count_scale=float(_ZONE_COUNT_SCALE),
        known_hand_definition=int(known_hand_columns["definition"]["index"]),
        known_hand_copies=int(known_hand_columns["count"]["index"]),
    )


def _number_columns(array: str) -> tuple[int, ...]:
    # The columns of one row of `array` that the engine does not mark as naming something.
    described = _ARRAYS[array]["columns"].values()
    return tuple(int(column["index"]) for column in described if column["kind"] not in _NAMING_KINDS)


def _naming_columns(array: str) -> dict[str, tuple[int, ...]]:
    # The columns of one row of `array` that name something, grouped by the engine's encoding of them.
    grouped: dict[str, list[int]] = collections.defaultdict(list)
    for column in _ARRAYS[array]["columns"].values():
        if column["kind"] in _NAMING_KINDS:
            grouped[str(column["encoding"])].append(int(column["index"]))
    return {encoding: tuple(indices) for encoding, indices in grouped.items()}


def start_batch(
    matchups: Sequence[Matchup], *, pool: CardPool, games: int, seed: int, boards: Collection[str]
) -> GameBatch:
    """Start the games of `matchups` on the engine, at most `games` of them at once.

    `matchups` must contain at least one matchup.

    `boards` names the board arrays that a decision reports, out of the ones
    `observation_space` lists. The engine encodes every array, but the ones
    no model reads make up more than half of a decision's bytes. A caller
    that stores decisions should therefore name only the arrays it reads,
    such as `jace.play.BOARD_ARRAYS`.

    Raises `DeckRejectedError` when the engine refuses a deck of the first
    games. The batch's `act` raises it too, when the engine refuses a deck
    of a game that starts later.
    """
    names = tuple(pool[card].name for card in range(len(pool)))
    card_ids = engine_card_ids(pool).numpy()
    return _ManabotBatch(matchups, names=names, card_ids=card_ids, games=games, seed=seed, boards=boards)


def land_and_spell_actions(boards: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    # `boards` contains the engine's arrays as tensors, each [n, ...]. Returns two bool [n, max_actions]
    # masks: the legal actions that play a land, then the legal actions that cast a spell.
    legal = boards["actions_valid"] > 0
    kinds = boards["actions"]
    lands = kinds[..., _PLAY_LAND_COLUMN] > 0
    spells = kinds[..., _CAST_SPELL_COLUMN] > 0
    return lands & legal, spells & legal


def turn_numbers(boards: Mapping[str, Tensor]) -> Tensor:
    # `boards` contains the engine's arrays as tensors, each [n, ...]. Returns float [n]: a number that
    # is the same for every decision of one turn of a game and differs between turns.
    return boards["agent_player"][:, 0, _TURN_COLUMN]


def _rules_texts() -> dict[str, str]:
    # Maps each printed name to its rules text, which the engine publishes only in its definitions JSON.
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
        matchups: Sequence[Matchup],
        *,
        names: tuple[str, ...],
        card_ids: npt.NDArray[np.int64],
        games: int,
        seed: int,
        boards: Collection[str],
    ) -> None:
        self._matchups: Sequence[Matchup] = matchups
        self._names: tuple[str, ...] = names
        self._cards_by_name: dict[str, int] = {name: card for card, name in enumerate(names)}
        # The `CardId` of each engine definition, or -1, as `engine_card_ids` returns it.
        self._card_ids: npt.NDArray[np.int64] = card_ids
        self._reported: tuple[str, ...] = tuple(boards)
        slots = min(games, len(matchups))
        # For each slot, the position in `matchups` of its game, or -1 once no matchup is left for it,
        # int64 [slots].
        self._game: npt.NDArray[np.int64] = np.arange(slots, dtype=np.int64)
        self._started: int = slots
        # The copies of each card in each seat's deck, for the game in each slot, int64 [slots, 2, cards].
        self._deck_counts: npt.NDArray[np.int64] = np.zeros((slots, SEATS, len(names)), dtype=np.int64)
        # The engine seeds slot i's successive games with base + i, base + i + n, base + i + 2n and so on,
        # where n is the number of slots, so a slot's games do not change with how fast other slots finish.
        # The base is a hash of `seed`, because batches from neighbouring seeds, such as 1 and 2, would
        # otherwise play nearly the same games. The hash is halved so that the additions cannot overflow.
        base = int(np.random.SeedSequence(seed).generate_state(1, np.uint64)[0]) >> 1
        self._engine: managym.VectorEnv = managym.VectorEnv(slots, seed=base, opponent_policy="none")
        setups = [self._setup(slot) for slot in range(slots)]
        # The engine writes the boards into the arrays it is given, and encodes them to fit the arrays' names and sizes.
        self._buffers: dict[str, npt.NDArray[Any]] = observation_space().encoder.allocate(slots)
        self._buffers.update(
            rewards=np.zeros(slots, dtype=np.float64),
            terminated=np.zeros(slots, dtype=np.uint8),
            truncated=np.zeros(slots, dtype=np.uint8),
        )
        try:
            self._engine.set_buffers(self._buffers)
            self._engine.reset_each_into_buffers(setups)
        except managym.AgentError as error:
            raise DeckRejectedError(str(error)) from error

    def decisions(self) -> Decisions:
        playing = np.flatnonzero(self._game >= 0)
        # Indexing copies each array, so a caller may keep a board after later steps overwrite the engine's arrays.
        return Decisions(
            games=self._game[playing],
            seats=np.array(self._engine.current_agent_indices(), dtype=np.int64)[playing],
            boards={name: self._buffers[name][playing] for name in self._reported},
            action_counts=self._buffers["actions_valid"][playing].sum(axis=1).astype(np.int64),
        )

    def act(self, actions: npt.NDArray[np.int64]) -> tuple[Finished, ...]:
        playing = self._game >= 0
        every_slot = np.zeros(len(playing), dtype=np.int64)
        every_slot[playing] = actions
        # An isolated step lets every other game advance when the engine cannot continue one, such as a
        # game whose board is too large for the arrays. The engine reports that game's error in its
        # slot's info. A slot with no game is not stepped.
        self._engine.step_isolated_into_buffers(every_slot.tolist(), playing.tolist(), recover_capacity=True)
        results = self._engine.get_last_info()
        # A game the engine could not continue, or one it cut short, has no result. `jace.engine.scoring`
        # counts both kinds as abandoned too.
        failed = np.array([result.get("game_error") is not None for result in results]) & playing
        cut_short = self._buffers["truncated"] > 0
        ended = ((self._buffers["terminated"] > 0) | cut_short | failed) & playing
        if not ended.any():
            return ()
        if failed.any():
            self._engine.replace_failed_into_buffers(failed.tolist())
        # The engine has already dealt each ended slot a new game with the old decks. Below, the next
        # matchup of the list replaces that game, and a slot with no matchup left is never stepped again.
        finished: list[Finished] = []
        setups: list[list[managym.PlayerConfig] | None] = [None] * len(playing)
        for slot in np.flatnonzero(ended).tolist():
            winner = results[slot].get("winner_index")
            unfinished = bool(failed[slot] or cut_short[slot])
            outcome = Outcome.ABANDONED if unfinished else _outcome(winner if isinstance(winner, int) else None)
            game = int(self._game[slot])
            finished.append(Finished(game=game, outcome=outcome, decks=self._matchups[game]))
            if self._started == len(self._matchups):
                self._game[slot] = -1
                continue
            self._game[slot] = self._started
            self._started += 1
            setups[slot] = self._setup(slot)
        try:
            self._engine.reassign_roots_into_buffers(setups)
        except managym.AgentError as error:
            raise DeckRejectedError(str(error)) from error
        return tuple(finished)

    def hidden_cards(self) -> HiddenCards:
        slots = len(self._game)
        # The engine returns each hand as a name-to-copies mapping, so this one loop over games is unavoidable.
        held = [
            (slot, seat, self._cards_by_name[name], copies)
            for seat in range(SEATS)
            for slot, hand in enumerate(self._engine.hand_snapshots([seat] * slots))
            for name, copies in hand.cards.items()
            if name in self._cards_by_name
        ]
        rows, seats, cards, copies = np.array(held, dtype=np.int64).reshape(-1, 4).T
        hands = np.zeros_like(self._deck_counts)
        hands[rows, seats, cards] = copies
        public = self._cards_in_public_zones()
        # The board lists a permanent on its controller's side. So when a side shows more copies of a
        # card than its seat's deck contains, the extra copies are borrowed, which means this seat
        # controls them but the other seat owns them.
        borrowed = (hands + public - self._deck_counts).clip(min=0)
        # A borrowed copy was counted among the borrower's public cards and not among the owner's. So it
        # is added back to the borrower's library and taken out of the owner's. `borrowed[:, ::-1]` is
        # what the other seat borrowed from this one.
        libraries = self._deck_counts - hands - public + borrowed - borrowed[:, ::-1]
        playing = self._game >= 0
        return HiddenCards(hands=hands[playing], libraries=libraries.clip(min=0)[playing])

    def _cards_in_public_zones(self) -> npt.NDArray[np.int64]:
        # Returns int64 [slots, 2, cards]: the copies of each card that the acting seat's board lists on
        # each seat's side on the battlefield, in the graveyard, in exile or on the stack. Tokens and cards
        # outside the pool are not counted.
        counts = np.zeros_like(self._deck_counts)
        acting = np.array(self._engine.current_agent_indices(), dtype=np.int64)
        # The engine definition id of each listed card, acting seat's side first, int64 [slots, 2, card rows].
        definitions = self._buffers["semantic_cards"].astype(np.int64) - 1
        for side, name in enumerate(("agent_cards", "opponent_cards")):
            rows = self._buffers[name]  # float32 [slots, card rows, features]
            # The valid mask also drops the sideboard cards that `semantic_cards` lists after the acting seat's cards.
            hidden_zone_or_token = rows[:, :, [_LIBRARY_COLUMN, _HAND_COLUMN, _TOKEN_COLUMN]].any(axis=-1)
            slots, card_rows = np.nonzero((self._buffers[f"{name}_valid"] > 0) & ~hidden_zone_or_token)
            cards = self._card_ids[definitions[slots, side, card_rows]]
            in_pool = cards >= 0
            # Side 0 is the acting seat's, so side s belongs to seat |s - acting|.
            seats = np.abs(side - acting[slots])
            np.add.at(counts, (slots[in_pool], seats[in_pool], cards[in_pool]), 1)
        return counts

    def _setup(self, slot: int) -> list[managym.PlayerConfig]:
        players: list[managym.PlayerConfig] = []
        for seat, deck in enumerate(self._matchups[int(self._game[slot])]):
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

"""`GameBatch` on manabot's engine, and the engine's card catalog as plain data.

Only this module and `jace.engine.scoring` import the engine, `managym`, and
manabot, which holds the description of the board arrays' sizes. A change in
either reaches those two files. Decklists are closed, which means a seat's
board never shows the opponent's deck and the guess model has to infer it.
The engine has no card rarity, so `engine_cards` cannot build a `CardPool`
without data from another source.

Boards are encoded at policy input version 10. Version 9 cannot hold what a
player did earlier in the turn, such as the life it lost, and the engine
refuses to encode a board that needs it, which Avatar cards do. The arrays
hold 80 cards and 64 permanents for each seat where the engine's defaults
are 60 and 40, because Avatar games outgrow the defaults. A game that
outgrows these too is abandoned. `board_layout` reads the engine's own
description of the arrays and checks it against what `jace.play.tokens`
assumes, so a change of layout fails when a model is built and not while it
plays.

Games run in the engine's `VectorEnv`, which steps every slot in one native call
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

POLICY_INPUT_VERSION = 10
# How many cards and permanents of one seat a board can hold. The engine's defaults are 60 and 40.
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
# The column of a player's row that holds the number of the turn being played.
_TURN_COLUMN = int(_ARRAYS["agent_player"]["columns"]["turn"]["index"])
# What `jace.play.tokens` assumes about the arrays, in the engine's own words.
_DEFINITION_ENCODING = "CardDefId + 1; 0 means absent"
_FOCUS_ORDER = "agent player, opponent player, agent cards, opponent cards, agent permanents, opponent permanents"
# The engine marks a column that names something, rather than counting it, as kind "id" or
# "reference". These are the encodings of such columns in the arrays `jace.play.tokens` reads.
_NAMING_KINDS = ("id", "reference")
_LINK_ENCODING = f"Padded object table index + 1; 0 absent. Order: {_FOCUS_ORDER}"
_SELECTOR_ENCODING = "Zero-based selector + 1; 0 absent"
_SEAT_ENCODING = "Raw player seat; -1 absent"
_EVENT_OBJECT_ENCODING = (
    "Raw event entity id in the accompanying kind namespace; -1 absent; not a padded object-table index"
)
# An action's continuation has the selectors' encoding, but it numbers a frame of the `relations`
# array, which jace does not read. manabot's own model joins it to its encoding of that frame.
_ACTION_FRAME_COLUMN = int(_ARRAYS["actions"]["columns"]["continuation"]["index"])


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

    The engine supplies a printed name, a set key, a Scryfall id, rules text
    and learning tokens. It has no rarity, so the caller joins rarity from
    another source before building a `CardPool`. `jace.data.card_table`
    joins by printed name; `scryfall_id` is here for a caller that needs an
    exact printing.
    Tokens, back faces and rules fixtures are left out because no deck may
    hold them.
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
    # It is read from the whole catalog, so a pair's number does not depend on which cards a pool holds.
    catalog = managym.catalog_tokens()
    pairs = sorted(set(zip(catalog["token_kind"], catalog["token_value"], strict=True)))
    return {pair: number for number, pair in enumerate(pairs)}


def engine_card_ids(pool: CardPool) -> Tensor:
    # Returns int64 [engine definition ids]: the `CardId` of each engine definition, or -1 when the pool lacks it.
    names = {pool[card].name: card for card in range(len(pool))}
    card_ids = torch.full((managym.CARD_DEFINITION_COUNT,), -1, dtype=torch.int64)
    for definition in managym.content_pool(managym.WORLD_SETUP)["definitions"]:
        card_ids[definition["card_def_id"]] = names.get(definition["registry_name"], -1)
    return card_ids


def observation_space() -> ObservationSpace:
    # The sizes of every board array, which the engine allocates and encodes by.
    config = ObservationConfig(
        policy_input_version=POLICY_INPUT_VERSION,
        max_cards_per_player=_MAX_CARDS,
        max_permanents_per_player=_MAX_PERMANENTS,
    )
    return ObservationSpace(config)


def board_layout() -> BoardLayout:
    """Describe the boards for `jace.play`, after checking them against what it assumes.

    `jace.play.tokens` joins the acting seat's and the opponent's arrays in
    the order the engine numbers an action's objects, reads a card's
    identity as a definition id plus one, and appends each seat's turn
    state to its player row and each card's to its card row. It reads every
    column of a player's and a card's row as a number, and the columns of
    other rows that name something by what their encoding says they name.
    Each of those is compared with the engine's description of its arrays.
    Raises `BoardLayoutError`, naming what differs, when one no longer holds.
    """
    columns = {name: list(array["columns"]) for name, array in _ARRAYS.items()}
    permanent_names, action_names, event_names = (
        _naming_columns(name) for name in ("agent_permanents", "actions", "events")
    )
    decklist = _naming_columns("decklists")
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
    }
    changed = [name for name, held in holds.items() if not held]
    if changed:
        raise BoardLayoutError(changed)
    decklist_columns = _ARRAYS["decklists"]["columns"]
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

    `matchups` must hold at least one game. `boards` names the board arrays
    a decision reports, out of the ones `observation_space` lists. The
    engine encodes them all, and the ones no model reads are more than half
    of a decision's bytes, so a caller that stores decisions names only
    what it reads, such as `jace.play.BOARD_ARRAYS`. Raises
    `DeckRejectedError` when the engine refuses a deck.
    """
    names = tuple(pool[card].name for card in range(len(pool)))
    card_ids = engine_card_ids(pool).numpy()
    return _ManabotBatch(matchups, names=names, card_ids=card_ids, games=games, seed=seed, boards=boards)


def land_and_spell_actions(boards: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    # boards: the engine's arrays as tensors, each [n, ...]. Returns two bool [n, max_actions]
    # masks: the legal actions that play a land, and the legal actions that cast a spell.
    legal = boards["actions_valid"] > 0
    kinds = boards["actions"]
    lands = kinds[..., _PLAY_LAND_COLUMN] > 0
    spells = kinds[..., _CAST_SPELL_COLUMN] > 0
    return lands & legal, spells & legal


def turn_numbers(boards: Mapping[str, Tensor]) -> Tensor:
    # boards: the engine's arrays as tensors, each [n, ...]. Returns float [n]: a number that is
    # the same for every decision of one turn of a game and differs between turns.
    return boards["agent_player"][:, 0, _TURN_COLUMN]


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
        # card_ids: int64 [engine definition ids], the `CardId` of each engine definition or -1.
        self._card_ids: npt.NDArray[np.int64] = card_ids
        self._reported: tuple[str, ...] = tuple(boards)
        slots = min(games, len(matchups))
        # game: int64 [slots], the position in `matchups` of the game each slot plays, or -1 for none.
        self._game: npt.NDArray[np.int64] = np.arange(slots, dtype=np.int64)
        self._started: int = slots
        # deck_counts: int64 [slots, 2, cards], copies of each card in each seat's deck.
        self._deck_counts: npt.NDArray[np.int64] = np.zeros((slots, SEATS, len(names)), dtype=np.int64)
        # The engine gives slot i the seeds base + i, base + i + n, ... for its successive games, so a
        # slot's games do not change with how fast the other slots finish. The base is hashed so that
        # batches started from neighbouring seeds share no game, and halved so the additions cannot overflow.
        base = int(np.random.SeedSequence(seed).generate_state(1, np.uint64)[0]) >> 1
        self._engine: managym.VectorEnv = managym.VectorEnv(slots, seed=base, opponent_policy="none")
        setups = [self._setup(slot) for slot in range(slots)]
        # The engine writes the boards into the arrays it is given, and encodes by their names and sizes.
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
        # Indexing copies, so a caller may keep a board after later steps overwrite the engine's arrays.
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
        # An isolated step reports a game the engine cannot continue, such as a board too large
        # for the observation arrays, in that slot's info and lets every other game advance. A
        # slot with no game is not stepped.
        self._engine.step_isolated_into_buffers(every_slot.tolist(), playing.tolist(), recover_capacity=True)
        results = self._engine.get_last_info()
        # A game the engine could not continue, or one it cut short, has no result. `jace.engine.scoring`
        # counts the same two as abandoned.
        failed = np.array([result.get("game_error") is not None for result in results]) & playing
        cut_short = self._buffers["truncated"] > 0
        ended = ((self._buffers["terminated"] > 0) | cut_short | failed) & playing
        if not ended.any():
            return ()
        if failed.any():
            self._engine.replace_failed_into_buffers(failed.tolist())
        # The engine has already dealt each ended slot a new game from the old decks. The next
        # matchup of the list is dealt over it below, and a slot with none left is never stepped again.
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
        # The board files a permanent under its controller. A side that shows more copies of a card
        # than its seat's deck holds must control the other seat's permanents, so the extra copies move back.
        borrowed = (hands + public - self._deck_counts).clip(min=0)
        # A borrowed copy was counted among the borrower's public cards, so it is added back to the
        # borrower's library and taken out of the owner's. `borrowed[:, ::-1]` is what the other seat
        # borrowed from this one.
        libraries = self._deck_counts - hands - public + borrowed - borrowed[:, ::-1]
        playing = self._game >= 0
        return HiddenCards(hands=hands[playing], libraries=libraries.clip(min=0)[playing])

    def _cards_in_public_zones(self) -> npt.NDArray[np.int64]:
        # Returns int64 [slots, 2, cards]: copies of each card that the acting seat's board lists on each
        # seat's side outside the hand and the library, which is the battlefield, graveyard, exile and stack.
        counts = np.zeros_like(self._deck_counts)
        acting = np.array(self._engine.current_agent_indices(), dtype=np.int64)
        # definitions: int64 [slots, 2, card rows], the engine definition id of each listed card, acting seat first.
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

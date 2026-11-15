"""Turn the engine's board arrays into the tokens the play models read.

`board_tokens`, `legal_actions` and `opponent_cards` are the only code in this
package that names the engine's arrays. Which columns of each kind of row are
numbers, and which name a card, another board object or a small category, is
not in the array names. A `BoardLayout` lists them, and the engine adapter
builds the layout from the engine's own description of its arrays.

All three play models read `BoardTokens`: the guess model, the player and
the win chance. The guess model also reads `OpponentCards`, its counts of
the opponent's cards. The models read nothing else from the engine, so
after a change of engine, or of how the engine encodes boards, only this
module has to change.

The engine pads every kind of row to a fixed capacity, and in a real game most
rows are padding. The tokens therefore contain each kind of row only up to its
last real row in the batch, and no layer reads the rest.

Each play model builds its own `EntityEncoder` as its first layer. The
encoder projects each kind of token to the model's width, and adds a
projection of the shared card embedding to each card and permanent. It
also adds to each permanent a projection of the object it is attached to
or blocks. `CategoryEncoder` embeds the categories of actions and events.
"""

from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn

from jace.engine.batch import SEATS, BoardLayout
from jace.play.types import Boards

__all__ = (
    "BOARD_ARRAYS",
    "BoardTokens",
    "CategoryEncoder",
    "EntityEncoder",
    "OpponentCards",
    "board_tokens",
    "card_vectors",
    "gather_entities",
    "legal_actions",
    "opponent_cards",
)


# The arrays of board objects, in the order in which the engine numbers objects in `action_focus`.
_ENTITY_ARRAYS = (
    "agent_player",
    "opponent_player",
    "agent_cards",
    "opponent_cards",
    "agent_permanents",
    "opponent_permanents",
)
# Every engine array that `board_tokens`, `legal_actions` and `opponent_cards` read. Code that plays or stores
# games for these models can ask the engine for these arrays alone, which are less than half of a decision's
# bytes.
BOARD_ARRAYS = (
    *_ENTITY_ARRAYS,
    *(f"{name}_valid" for name in _ENTITY_ARRAYS),
    "player_turn_state",
    "card_turn_state",
    "semantic_cards",
    "decklists",
    "known_hand",
    "actions",
    "actions_valid",
    "action_focus",
    "events",
    "events_valid",
)

# The arrays whose rows `board_tokens` trims to the last real row of the batch. Each has a `_valid` array.
_TRIMMED_ARRAYS = (*_ENTITY_ARRAYS, "actions", "events")

# The number of rows in each category column's table. A category's row is its value plus one, so that -1
# for none is row 0, and every value too large for the table shares the last row. Mode and ability
# selectors are the largest categories. In 7,000 decisions of random games, ability selectors stayed
# below 3 and mode selectors reached 17.
_CATEGORY_ROWS = 16


@dataclass(frozen=True, slots=True)
class BoardTokens:
    # The engine pads every kind of row to a fixed capacity, and most rows are padding. Each kind
    # here contains its rows only up to the last real row of any board in the batch. A board with fewer
    # real rows still has padding among the kept rows, and the masks mark which rows are real. The
    # float features are squashed by a signed log1p, so they stay near unit scale.
    #
    # players: float [b, 2, player features]. Row 0 is the acting seat and row 1 its opponent.
    players: Tensor
    # cards: float [b, card rows, card features], the acting seat's visible cards, then the opponent's.
    cards: Tensor
    # permanents: float [b, permanents, permanent features], the acting seat's, then the opponent's.
    permanents: Tensor
    # permanent_links: int64 [b, permanents, links]. A permanent has one link for each column of
    # `BoardLayout.permanent_links`: the object it is attached to and the attacker it blocks. Each link
    # is that object's entity number, or -1 for none.
    permanent_links: Tensor
    # entity_mask: bool [b, 2 + card rows + permanents], true for real rows. An entity's number is its
    # place in the order players, cards, permanents.
    entity_mask: Tensor
    # card_ids: int64 [b, card rows + permanents], the `CardId` of each card and permanent, or -1 for none.
    card_ids: Tensor
    # actions: float [b, kept actions, action features]. The kept actions are the leading action slots, up
    # to the last legal action of any board in the batch.
    actions: Tensor
    # action_categories: int64 [b, kept actions, action categories], each a row of its column's table.
    action_categories: Tensor
    # action_mask: bool [b, kept actions], true for legal actions.
    action_mask: Tensor
    # action_focus: int64 [b, kept actions, 2], the entity numbers an action names, or -1 for none.
    action_focus: Tensor
    # action_slots is max_actions, the board's own number of action slots. The action fields may contain
    # fewer, but a policy scores every slot, so its logits have the board's width.
    action_slots: int
    # events: float [b, events, event features], oldest first.
    events: Tensor
    # event_categories: int64 [b, events, event categories], each a row of its column's table.
    event_categories: Tensor
    # event_mask: bool [b, events], true for real events.
    event_mask: Tensor
    # deck_card_ids: int64 [b, decklist rows], the `CardId` of each card in the acting seat's
    # deck, or -1 for none.
    deck_card_ids: Tensor
    # deck_copies: float [b, decklist rows, 2], each card's copies not yet seen, then its copies in the deck.
    deck_copies: Tensor
    # deck_mask: bool [b, decklist rows], true for a card with copies not yet seen.
    deck_mask: Tensor


@dataclass(frozen=True, slots=True)
class OpponentCards:
    # shown: float [b, cards], how many copies of each card the acting seat sees on the opponent's side
    # of the board: on the battlefield, in the graveyard, in exile or on the stack. The board lists a
    # card under its controller, so a card that changed control counts for the side that controls it.
    shown: Tensor
    # known_in_hand: float [b, cards], how many copies of each card were revealed in the opponent's
    # hand, so the acting seat knows the opponent holds them. `hand` counts these cards too.
    known_in_hand: Tensor
    # hand: float [b], the number of cards in the opponent's hand.
    hand: Tensor
    # hidden: float [b], the number of cards in the opponent's hand and library.
    hidden: Tensor


def legal_actions(boards: Boards) -> Tensor:
    # Returns bool [b, max_actions], true for the actions the acting seat may take.
    return boards["actions_valid"] > 0


def opponent_cards(boards: Boards, *, layout: BoardLayout, engine_to_card: Tensor, cards: int) -> OpponentCards:
    # The opponent's rows are row 1 of `semantic_cards` and of `known_hand`. Each of its permanents is also
    # one of its card rows, on the battlefield, so counting the card rows counts each permanent once.
    # `cards` is the size of the card pool.
    rows = boards["opponent_cards"]
    unshown = (rows[..., list(layout.card_unshown_flags)] > 0).any(dim=-1)
    shown_rows = (boards["opponent_cards_valid"] > 0) & ~unshown
    shown = _copies(_card_ids(boards["semantic_cards"][:, 1], engine_to_card), shown_rows.float(), cards=cards)
    known = boards["known_hand"][:, 1]
    known_ids = _card_ids(known[..., layout.known_hand_definition], engine_to_card)
    known_in_hand = _copies(known_ids, known[..., layout.known_hand_copies], cards=cards)
    # The engine stores each zone count divided by `zone_count_scale`. Scaled back, a count is a whole
    # number up to float error, and rounding removes that error.
    player = (boards["opponent_player"][:, 0] * layout.zone_count_scale).round()
    hand = player[:, layout.hand_count]
    return OpponentCards(
        shown=shown, known_in_hand=known_in_hand, hand=hand, hidden=hand + player[:, layout.library_count]
    )


def _copies(card_ids: Tensor, copies: Tensor, *, cards: int) -> Tensor:
    # card_ids is int64 [b, rows], -1 for none, and copies is float [b, rows]. Returns float [b, cards], each
    # card's copies summed over its rows. A row whose id is -1 adds nothing.
    counted = torch.where(card_ids >= 0, copies, 0.0)
    return torch.zeros(len(card_ids), cards, device=copies.device).scatter_add_(1, card_ids.clamp_min(0), counted)


def _squash(features: Tensor) -> Tensor:
    # Some columns contain raw counts, amounts and ranks, which can be large. After a signed log1p, every
    # input is near unit scale.
    return features.sign() * features.abs().log1p()


def _card_ids(definitions: Tensor, engine_to_card: Tensor) -> Tensor:
    # definitions: float [...], engine definition ids plus one, zero for none. Returns int64 [...]
    # `CardId`s, -1 for none and for a definition the card pool lacks.
    engine_ids = definitions.long() - 1
    return torch.where(engine_ids >= 0, engine_to_card[engine_ids.clamp_min(0)], -1)


def _kept_rows(boards: Boards, *, layout: BoardLayout) -> dict[str, int]:
    # Returns how many leading rows to keep of each trimmed array and of the acting seat's decklist: enough
    # to reach the last real row of any board in the batch. A decklist row is real while the deck has copies
    # of its card that the acting seat has not seen. Every count comes from one read of the device, because
    # each read waits for the device to finish its queued work.
    real = {name: boards[f"{name}_valid"] > 0 for name in _TRIMMED_ARRAYS}
    unseen, _ = layout.decklist_copies
    real["decklists"] = boards["decklists"][:, 0, :, unseen] > 0
    ends = [(rows * torch.arange(1, rows.shape[1] + 1, device=rows.device)).amax() for rows in real.values()]
    return dict(zip(real, torch.stack(ends).tolist(), strict=True))


def _entity_numbers(boards: Boards, kept: dict[str, int]) -> Tensor:
    # Returns int64 [padded entities], which maps a place in the engine's padded order to the entity number
    # of the same object among the kept rows, or to -1 for a dropped row. No focus or link names a dropped
    # row, because every row after a kind's last real one is padding.
    numbers: list[int] = []
    first = 0
    for name in _ENTITY_ARRAYS:
        capacity, count = boards[f"{name}_valid"].shape[1], kept[name]
        numbers += [*range(first, first + count), *[-1] * (capacity - count)]
        first += count
    return torch.tensor(numbers, device=boards["actions"].device)


def _categories(values: Tensor) -> Tensor:
    # values is float [..., columns] of small whole numbers, where none is -1 or 0 depending on the column.
    # Returns the int64 row of each value in its column's table.
    rows = values.long() + 1
    return rows.clamp(0, _CATEGORY_ROWS - 1)


def board_tokens(boards: Boards, *, layout: BoardLayout, engine_to_card: Tensor) -> BoardTokens:
    # Each of the engine's arrays has a leading batch axis. The engine adapter checks what the engine
    # reports about the arrays before it builds `layout`, so these facts hold whenever a layout exists:
    #   - Each kind of row comes as one array for the acting seat ("agent") and one for the opponent,
    #     with the same columns. A `_valid` array marks the real rows.
    #   - Each entry of `action_focus` is an object's index in one padded order: agent player, opponent
    #     player, agent cards, opponent cards, agent permanents, opponent permanents, with each kind
    #     padded to its capacity. Concatenating the arrays in that order puts each object at its focus
    #     index. A permanent's link columns contain the same index plus one. The tokens keep fewer rows of
    #     each kind, so `_entity_numbers` maps each index to the object's token.
    #   - `semantic_cards` contains the definition id plus one of each card row, with the acting seat's
    #     rows first. A permanent's row and a decklist row contain the same in the columns that `layout`
    #     names.
    #   - `player_turn_state` has one row for each seat, and `card_turn_state` has one value for each
    #     card row. Both put the acting seat first, and so does `decklists`.
    # A column that names something, such as a definition or a link, is read only as what it names,
    # never as a number.
    #
    # engine_to_card: int64 [engine definition ids], the `CardId` of each engine definition, or -1 for
    # a definition the card pool lacks.
    kept = _kept_rows(boards, layout=layout)
    rows = {name: boards[name][:, : kept[name]] for name in _TRIMMED_ARRAYS}
    valid = {name: boards[f"{name}_valid"][:, : kept[name]] for name in _TRIMMED_ARRAYS}
    agent_cards, opponent_cards = kept["agent_cards"], kept["opponent_cards"]
    players = torch.cat((rows["agent_player"], rows["opponent_player"]), dim=1)
    players = torch.cat((players, boards["player_turn_state"]), dim=-1)
    turn_state = boards["card_turn_state"]
    card_turn_state = torch.cat((turn_state[:, 0, :agent_cards], turn_state[:, 1, :opponent_cards]), dim=1)
    cards = torch.cat((rows["agent_cards"], rows["opponent_cards"]), dim=1)
    cards = torch.cat((cards, card_turn_state[..., None]), dim=-1)
    permanents = torch.cat((rows["agent_permanents"], rows["opponent_permanents"]), dim=1)
    semantic = boards["semantic_cards"]
    definitions = torch.cat(
        (semantic[:, 0, :agent_cards], semantic[:, 1, :opponent_cards], permanents[..., layout.permanent_definition]),
        dim=1,
    )
    entity_valid = torch.cat([valid[name] for name in _ENTITY_ARRAYS], dim=1)
    entity_numbers = _entity_numbers(boards, kept)
    # Links and focus are both places in the engine's padded order, with -1 for none. A link column contains
    # the place plus one, so one is subtracted.
    links = permanents[..., list(layout.permanent_links)].long() - 1
    focus = boards["action_focus"][:, : kept["actions"]].long()
    actions, events = rows["actions"], rows["events"]
    # The acting seat's decklist: [b, decklist rows, decklist columns].
    decklist = boards["decklists"][:, 0, : kept["decklists"]]
    unseen, _ = layout.decklist_copies
    return BoardTokens(
        players=_squash(players),
        cards=_squash(cards),
        permanents=_squash(permanents[..., list(layout.permanent_features)]),
        permanent_links=torch.where(links >= 0, entity_numbers[links.clamp_min(0)], -1),
        entity_mask=entity_valid > 0,
        card_ids=_card_ids(definitions, engine_to_card),
        actions=_squash(actions[..., list(layout.action_features)]),
        action_categories=_categories(actions[..., list(layout.action_categories)]),
        action_mask=valid["actions"] > 0,
        action_focus=torch.where(focus >= 0, entity_numbers[focus.clamp_min(0)], -1),
        action_slots=boards["actions_valid"].shape[1],
        events=_squash(events[..., list(layout.event_features)]),
        event_categories=_categories(events[..., list(layout.event_categories)]),
        event_mask=valid["events"] > 0,
        deck_card_ids=_card_ids(decklist[..., layout.decklist_definition], engine_to_card),
        deck_copies=_squash(decklist[..., list(layout.decklist_copies)]),
        deck_mask=decklist[..., unseen] > 0,
    )


def card_vectors(card_table: Tensor, card_ids: Tensor) -> Tensor:
    # card_table: float [pool cards, card_width]. card_ids: int64 [...], -1 for none.
    # Returns float [..., card_width], zeros where the id is -1.
    return card_table[card_ids.clamp_min(0)] * (card_ids >= 0)[..., None]


def gather_entities(entities: Tensor, numbers: Tensor) -> Tensor:
    # entities: float [b, entities, width]. numbers: int64 [b, rows, slots], entity numbers or -1 for none.
    # Returns float [b, rows, slots * width]: each row's entities side by side, zeros where a slot names none.
    batch, rows, slots = numbers.shape
    named = (numbers >= 0).reshape(batch, rows * slots, 1)
    index = numbers.clamp_min(0).reshape(batch, rows * slots, 1).expand(-1, -1, entities.shape[-1])
    return (entities.gather(1, index) * named).reshape(batch, rows, slots * entities.shape[-1])


class EntityEncoder(nn.Module):
    """Project players, cards and permanents to one width, in entity order.

    Each kind has its own projection. A card or permanent whose card is known
    also gets a projection of that card's shared embedding, which is how one
    card's token differs from another's. A permanent that is attached to an
    object or blocks an attacker also gets a projection of that object's
    vector, with one projection for each kind of link, so the transformer's
    input shows which attacker a blocker blocks. `embed` returns float
    `[b, entities, width]`, in the order that `BoardTokens.action_focus`
    indexes.
    """

    def __init__(self, layout: BoardLayout, *, card_width: int, width: int) -> None:
        super().__init__()
        self.players: nn.Linear = nn.Linear(layout.player_features, width)
        self.cards: nn.Linear = nn.Linear(layout.card_features, width)
        self.permanents: nn.Linear = nn.Linear(len(layout.permanent_features), width)
        self.card_vectors: nn.Linear = nn.Linear(card_width, width, bias=False)
        # seats: float [seats, width], added to the player rows to tell the acting seat from its opponent.
        self.seats: nn.Parameter = nn.Parameter(torch.zeros(SEATS, width))
        # One linear map of the linked objects side by side is one projection for each kind of link, summed.
        self.links: nn.Linear = nn.Linear(len(layout.permanent_links) * width, width, bias=False)

    def embed(self, tokens: BoardTokens, card_table: Tensor) -> Tensor:
        # card_table: float [pool cards, card_width], the shared embedding of every card in the pool.
        objects = torch.cat((self.cards(tokens.cards), self.permanents(tokens.permanents)), dim=1)
        objects = objects + self.card_vectors(card_vectors(card_table, tokens.card_ids))
        entities = torch.cat((self.players(tokens.players) + self.seats, objects), dim=1)
        # A link reads its object's vector from before any link is added, so a permanent gets the vector of the
        # object it links to, but nothing from that object's own links.
        linked = self.links(gather_entities(entities, tokens.permanent_links))  # [b, permanents, width]
        first_permanent = entities.shape[1] - linked.shape[1]
        return torch.cat((entities[:, :first_permanent], entities[:, first_permanent:] + linked), dim=1)


class CategoryEncoder(nn.Module):
    """Embed the category columns of one kind of row and sum the embeddings.

    Each column has its own table, so an action's mode 2 and its ability 2
    get different vectors. `forward` takes the int64 `[..., columns]`
    categories that `BoardTokens` contains for the kind, and returns float
    `[..., width]`.
    """

    def __init__(self, columns: int, *, width: int) -> None:
        super().__init__()
        self.table: nn.Embedding = nn.Embedding(columns * _CATEGORY_ROWS, width)
        # first_rows: int64 [columns], the row of `table` where each column's own rows start.
        self.first_rows: Tensor
        self.register_buffer("first_rows", torch.arange(columns) * _CATEGORY_ROWS, persistent=False)

    @override
    def forward(self, categories: Tensor) -> Tensor:
        rows = categories + self.first_rows
        return self.table(rows).sum(dim=-2)

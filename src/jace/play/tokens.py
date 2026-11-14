"""Turn the engine's board arrays into the tokens the play models read.

`board_tokens`, `legal_actions` and `opponent_cards` are the only code in
this package that names the engine's arrays. What they cannot know from
the names alone, which columns of each kind of row are numbers and which
name a card, another board object or a small category, comes in a
`BoardLayout` that the engine adapter builds from the engine's own
description of its arrays. The guess model and the player model read
`BoardTokens`, and the guess model also reads `OpponentCards`, its counts
of the opponent's cards. They read nothing else, so a change of engine or
of encoder version stops here. `EntityEncoder` is the first layer both
models share: it projects each kind of token to the model's width, adds
the shared card embedding to the tokens that are a card, and adds to each
permanent the object it is attached to or blocks. `CategoryEncoder` embeds
the categories of actions and events.
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


# Every engine array `board_tokens` and `opponent_cards` read. Whatever plays or stores boards for
# these models needs these and no others, which is less than half of what the engine encodes.
_ENTITY_ARRAYS = (
    "agent_player",
    "opponent_player",
    "agent_cards",
    "opponent_cards",
    "agent_permanents",
    "opponent_permanents",
)
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

# A category's value plus one, so that -1 for none is row 0, is clamped to the rows of its table,
# and larger values share the last row. Mode and ability selectors are the largest categories: in
# 7,000 decisions of random games, ability selectors stayed below 3 and mode selectors reached 17.
_CATEGORY_ROWS = 16


@dataclass(frozen=True, slots=True)
class BoardTokens:
    # players: float [b, 2, player features]. Row 0 is the acting seat, row 1 its opponent.
    players: Tensor
    # cards: float [b, cards, card features], the acting seat's visible cards, then the opponent's.
    cards: Tensor
    # permanents: float [b, permanents, permanent features], the acting seat's, then the opponent's.
    permanents: Tensor
    # permanent_links: int64 [b, permanents, links], the entity number of the object each of a
    # permanent's links names, in the order of `BoardLayout.permanent_links`, or -1 for none.
    permanent_links: Tensor
    # entity_mask: bool [b, 2 + cards + permanents], true for real rows. Entities are
    # numbered in the order players, cards, permanents.
    entity_mask: Tensor
    # card_ids: int64 [b, cards + permanents], the `CardId` of each card and permanent, or -1 for none.
    card_ids: Tensor
    # actions: float [b, max_actions, action features].
    actions: Tensor
    # action_categories: int64 [b, max_actions, action categories], each a row of its column's table.
    action_categories: Tensor
    # action_mask: bool [b, max_actions], true for legal actions.
    action_mask: Tensor
    # action_focus: int64 [b, max_actions, 2], the entity numbers an action names, or -1 for none.
    action_focus: Tensor
    # events: float [b, events, event features], oldest first.
    events: Tensor
    # event_categories: int64 [b, events, event categories], each a row of its column's table.
    event_categories: Tensor
    # event_mask: bool [b, events].
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
    # shown: float [b, cards], copies of each card the acting seat sees on the opponent's side of the
    # board: on the battlefield, in the graveyard, in exile or on the stack. The board files a card
    # under its controller, so a card that changed control counts on its controller's side.
    shown: Tensor
    # known_in_hand: float [b, cards], copies of each card the acting seat knows the opponent holds
    # because they were revealed. They are among the cards that `hand` counts.
    known_in_hand: Tensor
    # hand: float [b], cards in the opponent's hand. hidden: float [b], cards in its hand and library.
    hand: Tensor
    hidden: Tensor


def legal_actions(boards: Boards) -> Tensor:
    # Returns bool [b, max_actions], true for the actions the acting seat may take.
    return boards["actions_valid"] > 0


def opponent_cards(boards: Boards, *, layout: BoardLayout, engine_to_card: Tensor, cards: int) -> OpponentCards:
    # The opponent's cards are row 1 of `semantic_cards` and `known_hand`. Its permanents are also its
    # card rows on the battlefield, so the permanent rows are not counted again. `cards` is the pool size.
    rows = boards["opponent_cards"]
    unshown = (rows[..., list(layout.card_unshown_flags)] > 0).any(dim=-1)
    shown_rows = (boards["opponent_cards_valid"] > 0) & ~unshown
    shown = _copies(_card_ids(boards["semantic_cards"][:, 1], engine_to_card), shown_rows.float(), cards=cards)
    known = boards["known_hand"][:, 1]
    known_ids = _card_ids(known[..., layout.known_hand_definition], engine_to_card)
    known_in_hand = _copies(known_ids, known[..., layout.known_hand_copies], cards=cards)
    # The counts are whole numbers stored divided by the scale, so rounding undoes the float error.
    player = (boards["opponent_player"][:, 0] * layout.zone_count_scale).round()
    hand = player[:, layout.hand_count]
    return OpponentCards(
        shown=shown, known_in_hand=known_in_hand, hand=hand, hidden=hand + player[:, layout.library_count]
    )


def _copies(card_ids: Tensor, copies: Tensor, *, cards: int) -> Tensor:
    # card_ids: int64 [b, rows], -1 for none. copies: float [b, rows]. Returns float [b, cards], each card's copies.
    counted = torch.where(card_ids >= 0, copies, 0.0)
    return torch.zeros(len(card_ids), cards, device=copies.device).scatter_add_(1, card_ids.clamp_min(0), counted)


def _squash(features: Tensor) -> Tensor:
    # Some columns hold unscaled counts, amounts and ranks; this keeps every input near unit scale.
    return features.sign() * features.abs().log1p()


def _card_ids(definitions: Tensor, engine_to_card: Tensor) -> Tensor:
    # definitions: float [...], engine definition ids plus one, zero for none. Returns int64 [...]
    # `CardId`s, -1 for none and for a definition the card pool lacks.
    engine_ids = definitions.long() - 1
    return torch.where(engine_ids >= 0, engine_to_card[engine_ids.clamp_min(0)], -1)


def _categories(values: Tensor) -> Tensor:
    # values: float [..., columns], each -1 or 0 for none. Returns int64 rows of the category tables.
    rows = values.long() + 1
    return rows.clamp(0, _CATEGORY_ROWS - 1)


def board_tokens(boards: Boards, *, layout: BoardLayout, engine_to_card: Tensor) -> BoardTokens:
    # This is the only code in jace.play that names the engine's arrays. Each has a leading
    # batch axis. The engine adapter checks what the engine reports about them before it
    # builds `layout`, so these hold whenever a layout exists:
    #   - each kind of row comes as an array for the acting seat ("agent") and one for the
    #     opponent, with the same columns, and a `_valid` array marks the real rows;
    #   - `action_focus` numbers objects as agent player, opponent player, agent cards,
    #     opponent cards, agent permanents, opponent permanents, each kind padded to its
    #     capacity, so concatenating the arrays in that order makes a focus index the
    #     position of its token. A permanent's link columns hold the same index plus one;
    #   - `semantic_cards` holds the definition id + 1 of each card row, acting seat first,
    #     and a permanent's row and a decklist's row hold the same in the columns `layout` names;
    #   - `player_turn_state` has one row for each seat and `card_turn_state` one value for
    #     each card row, acting seat first, and `decklists` has the acting seat's rows first.
    # A column that names something, such as a definition or a link, is read only as what it
    # names, never as a number.
    #
    # engine_to_card: int64 [engine definition ids], the `CardId` of each engine definition,
    # or -1 for a definition the card pool lacks.
    players = torch.cat((boards["agent_player"], boards["opponent_player"]), dim=1)
    players = torch.cat((players, boards["player_turn_state"]), dim=-1)
    cards = torch.cat((boards["agent_cards"], boards["opponent_cards"]), dim=1)
    cards = torch.cat((cards, boards["card_turn_state"].flatten(1)[..., None]), dim=-1)
    permanents = torch.cat((boards["agent_permanents"], boards["opponent_permanents"]), dim=1)
    definitions = torch.cat((boards["semantic_cards"].flatten(1), permanents[..., layout.permanent_definition]), dim=1)
    entity_valid = torch.cat([boards[f"{name}_valid"] for name in _ENTITY_ARRAYS], dim=1)
    actions, events = boards["actions"], boards["events"]
    decklist = boards["decklists"][:, 0]  # the acting seat's: [b, decklist rows, decklist columns]
    unseen, _ = layout.decklist_copies
    return BoardTokens(
        players=_squash(players),
        cards=_squash(cards),
        permanents=_squash(permanents[..., list(layout.permanent_features)]),
        # A link holds its object's position in the focus order plus one, and the entities are the
        # same rows in the same order, so the link minus one is the object's entity number.
        permanent_links=permanents[..., list(layout.permanent_links)].long() - 1,
        entity_mask=entity_valid > 0,
        card_ids=_card_ids(definitions, engine_to_card),
        actions=_squash(actions[..., list(layout.action_features)]),
        action_categories=_categories(actions[..., list(layout.action_categories)]),
        action_mask=legal_actions(boards),
        action_focus=boards["action_focus"].long(),
        events=_squash(events[..., list(layout.event_features)]),
        event_categories=_categories(events[..., list(layout.event_categories)]),
        event_mask=boards["events_valid"] > 0,
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
    return (entities.gather(1, index) * named).reshape(batch, rows, -1)


class EntityEncoder(nn.Module):
    """Project players, cards and permanents to one width, in entity order.

    Each kind has its own projection. A card or permanent with a known card
    also gets the projection of that card's shared embedding, which is how a
    model tells one card from another. A permanent that is attached to an
    object or blocks an attacker also gets a projection of that object's
    vector, with one projection for each kind of link, so the transformer
    sees which attacker a blocker blocks. `embed` returns
    `[b, entities, width]` in the order that `BoardTokens.action_focus`
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
        # A link reads its object's vector from before any link is added, so it reaches one object away.
        linked = self.links(gather_entities(entities, tokens.permanent_links))  # [b, permanents, width]
        first_permanent = entities.shape[1] - linked.shape[1]
        return torch.cat((entities[:, :first_permanent], entities[:, first_permanent:] + linked), dim=1)


class CategoryEncoder(nn.Module):
    """Embed the category columns of one kind of row and sum the embeddings.

    Each column has its own table, so an action's mode 2 and its ability 2
    get different vectors. `forward` takes the int64 `[..., columns]` that
    `BoardTokens` holds for the kind and returns float `[..., width]`.
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

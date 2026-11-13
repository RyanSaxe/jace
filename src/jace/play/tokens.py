"""Turn the engine's board arrays into the tokens the play models read.

`board_tokens` and `legal_actions` are the only code in this package that
names the engine's arrays. What they cannot know from the names alone, the
width of each kind of row and the column that holds a permanent's card,
comes in a `BoardLayout` that the engine adapter builds from the engine's
own description of its arrays. The guess model and the player model read
`BoardTokens` and nothing else, so a change of engine or of encoder version
stops here. `EntityEncoder` is the first layer both
models share: it projects each kind of token to the model's width and adds
the shared card embedding to the tokens that are a card.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from jace.engine.batch import SEATS, BoardLayout
from jace.play.types import Boards

__all__ = ("BOARD_ARRAYS", "BoardTokens", "EntityEncoder", "board_tokens", "legal_actions")


# Every engine array `board_tokens` reads. Whatever plays or stores boards for these models
# needs these and no others, which is less than half of what the engine encodes.
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
    "actions",
    "actions_valid",
    "action_focus",
    "events",
    "events_valid",
)


@dataclass(frozen=True, slots=True)
class BoardTokens:
    # players: float [b, 2, player features]. Row 0 is the acting seat, row 1 its opponent.
    players: Tensor
    # cards: float [b, cards, card features], the acting seat's visible cards, then the opponent's.
    cards: Tensor
    # permanents: float [b, permanents, permanent features], the acting seat's, then the opponent's.
    permanents: Tensor
    # entity_mask: bool [b, 2 + cards + permanents], true for real rows. Entities are
    # numbered in the order players, cards, permanents.
    entity_mask: Tensor
    # card_ids: int64 [b, cards + permanents], the `CardId` of each card and permanent, or -1 for none.
    card_ids: Tensor
    # actions: float [b, max_actions, action features].
    actions: Tensor
    # action_mask: bool [b, max_actions], true for legal actions.
    action_mask: Tensor
    # action_focus: int64 [b, max_actions, 2], the entity numbers an action names, or -1 for none.
    action_focus: Tensor
    # events: float [b, events, event features], oldest first.
    events: Tensor
    # event_mask: bool [b, events].
    event_mask: Tensor


def legal_actions(boards: Boards) -> Tensor:
    # Returns bool [b, max_actions], true for the actions the acting seat may take.
    return boards["actions_valid"] > 0


def _squash(features: Tensor) -> Tensor:
    # Some columns hold raw ids, indices and ranks in the hundreds; this keeps every input near unit scale.
    return features.sign() * features.abs().log1p()


def board_tokens(boards: Boards, *, layout: BoardLayout, engine_to_card: Tensor) -> BoardTokens:
    # This is the only code in jace.play that names the engine's arrays. Each has a leading
    # batch axis. The engine adapter checks what the engine reports about them before it
    # builds `layout`, so these hold whenever a layout exists:
    #   - each kind of row comes as an array for the acting seat ("agent") and one for the
    #     opponent, with the same columns, and a `_valid` array marks the real rows;
    #   - `action_focus` numbers objects as agent player, opponent player, agent cards,
    #     opponent cards, agent permanents, opponent permanents, each kind padded to its
    #     capacity, so concatenating the arrays in that order makes a focus index the
    #     position of its token;
    #   - `semantic_cards` holds the definition id + 1 of each card row, acting seat first,
    #     and a permanent's row holds the same in column `layout.permanent_definition`;
    #   - `player_turn_state` has one row for each seat and `card_turn_state` one value for
    #     each card row, acting seat first.
    #
    # engine_to_card: int64 [engine definition ids], the `CardId` of each engine definition,
    # or -1 for a definition the card pool lacks.
    players = torch.cat((boards["agent_player"], boards["opponent_player"]), dim=1)
    players = torch.cat((players, boards["player_turn_state"]), dim=-1)
    cards = torch.cat((boards["agent_cards"], boards["opponent_cards"]), dim=1)
    cards = torch.cat((cards, boards["card_turn_state"].flatten(1)[..., None]), dim=-1)
    permanents = torch.cat((boards["agent_permanents"], boards["opponent_permanents"]), dim=1)
    definitions = torch.cat((boards["semantic_cards"].flatten(1), permanents[..., layout.permanent_definition]), dim=1)
    engine_ids = definitions.long() - 1
    # card_ids: int64 [b, cards + permanents], -1 for none.
    card_ids = torch.where(engine_ids >= 0, engine_to_card[engine_ids.clamp_min(0)], -1)
    # The definition id is an identity, read through the card embedding, never a numeric feature.
    permanents[..., layout.permanent_definition] = 0.0
    entity_valid = torch.cat([boards[f"{name}_valid"] for name in _ENTITY_ARRAYS], dim=1)
    return BoardTokens(
        players=_squash(players),
        cards=_squash(cards),
        permanents=_squash(permanents),
        entity_mask=entity_valid > 0,
        card_ids=card_ids,
        actions=_squash(boards["actions"]),
        action_mask=legal_actions(boards),
        action_focus=boards["action_focus"].long(),
        events=_squash(boards["events"]),
        event_mask=boards["events_valid"] > 0,
    )


class EntityEncoder(nn.Module):
    """Project players, cards and permanents to one width, in entity order.

    Each kind has its own projection. A card or permanent with a known card
    also gets the projection of that card's shared embedding, which is how a
    model tells one card from another. `embed` returns `[b, entities, width]`
    in the order that `BoardTokens.action_focus` indexes.
    """

    def __init__(self, layout: BoardLayout, *, card_width: int, width: int) -> None:
        super().__init__()
        self.players: nn.Linear = nn.Linear(layout.player_features, width)
        self.cards: nn.Linear = nn.Linear(layout.card_features, width)
        self.permanents: nn.Linear = nn.Linear(layout.permanent_features, width)
        self.card_vectors: nn.Linear = nn.Linear(card_width, width, bias=False)
        # seats: float [seats, width], added to the player rows to tell the acting seat from its opponent.
        self.seats: nn.Parameter = nn.Parameter(torch.zeros(SEATS, width))

    def embed(self, tokens: BoardTokens, card_table: Tensor) -> Tensor:
        # card_table: float [pool cards, card_width], the shared embedding of every card in the pool.
        known = tokens.card_ids >= 0
        vectors = card_table[tokens.card_ids.clamp_min(0)] * known[..., None]
        objects = torch.cat((self.cards(tokens.cards), self.permanents(tokens.permanents)), dim=1)
        players = self.players(tokens.players) + self.seats
        return torch.cat((players, objects + self.card_vectors(vectors)), dim=1)

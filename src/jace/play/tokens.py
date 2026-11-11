"""Turn the engine's board arrays into the tokens the play models read.

`board_tokens` is the only code in this package that knows the names and
layout of the engine's arrays. The guess model and the player model read its
`BoardTokens` and nothing else, so a change of engine or of encoder version
is a change to that one function. `EntityEncoder` is the first layer both
models share: it projects each kind of token to the model's width and adds
the shared card embedding to the tokens that are a card.
"""

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from jace.engine.batch import SEATS
from jace.modules import run
from jace.play.types import Boards

__all__ = ("BoardTokens", "EntityEncoder", "board_tokens")

# The column of the engine's permanent rows that holds the card definition id + 1.
_PERMANENT_DEFINITION_COLUMN = 24


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


def _squash(features: Tensor) -> Tensor:
    # Some columns hold raw ids, indices and ranks in the hundreds; this keeps every input near unit scale.
    return features.sign() * features.abs().log1p()


def board_tokens(boards: Boards, *, engine_to_card: Tensor) -> BoardTokens:
    # ASSUMPTION, TO BE REVISITED. This function is coupled to the arrays that manabot's
    # observation encoder writes, each with a leading batch axis. The arrays below have the
    # same names and shapes at policy input versions 6 and 8; the adapter encodes at 8:
    #   agent_player, opponent_player          float32 [b, 1, 73]
    #   agent_cards, opponent_cards            float32 [b, 60, 56]
    #   agent_permanents, opponent_permanents  float32 [b, 40, 38]; column 24 is the card definition id + 1
    #   semantic_cards                         float32 [b, 2, 60]; the definition id + 1 of each card row, 0 for none
    #   actions                                float32 [b, 512, 22]
    #   action_focus                           int32   [b, 512, 2]; -1 for none
    #   events                                 float32 [b, 32, 7], oldest first
    #   <name>_valid                           float32 [b, rows] for each array above except semantic_cards
    # action_focus numbers objects as agent player, opponent player, agent cards, opponent cards,
    # agent permanents, opponent permanents, each kind padded to its capacity. Concatenating the
    # arrays in that order therefore makes a focus index the position of its token.
    # No other code in jace.play may name these arrays or their columns.
    #
    # engine_to_card: int64 [engine definition ids], the `CardId` of each engine definition,
    # or -1 for a definition the card pool lacks.
    players = torch.cat((boards["agent_player"], boards["opponent_player"]), dim=1)
    cards = torch.cat((boards["agent_cards"], boards["opponent_cards"]), dim=1)
    permanents = torch.cat((boards["agent_permanents"], boards["opponent_permanents"]), dim=1)
    card_ids = _card_ids(boards["semantic_cards"], permanents, engine_to_card=engine_to_card)
    # The definition id is an identity, read through the card embedding, never a numeric feature.
    permanents[..., _PERMANENT_DEFINITION_COLUMN] = 0.0
    entity_valid = torch.cat(
        (
            boards["agent_player_valid"],
            boards["opponent_player_valid"],
            boards["agent_cards_valid"],
            boards["opponent_cards_valid"],
            boards["agent_permanents_valid"],
            boards["opponent_permanents_valid"],
        ),
        dim=1,
    )
    return BoardTokens(
        players=_squash(players),
        cards=_squash(cards),
        permanents=_squash(permanents),
        entity_mask=entity_valid > 0,
        card_ids=card_ids,
        actions=_squash(boards["actions"]),
        action_mask=boards["actions_valid"] > 0,
        action_focus=boards["action_focus"].long(),
        events=_squash(boards["events"]),
        event_mask=boards["events_valid"] > 0,
    )


def _card_ids(semantic_cards: Tensor, permanents: Tensor, *, engine_to_card: Tensor) -> Tensor:
    # semantic_cards: float [b, seats, cards per seat]. permanents: float [b, permanents, permanent features],
    # read before its definition column is cleared. Returns int64 [b, cards + permanents], -1 for none.
    definitions = torch.cat((semantic_cards.flatten(1), permanents[..., _PERMANENT_DEFINITION_COLUMN]), dim=1)
    engine_ids = definitions.long() - 1
    return torch.where(engine_ids >= 0, engine_to_card[engine_ids.clamp_min(0)], -1)


class EntityEncoder(nn.Module):
    """Project players, cards and permanents to one width, in entity order.

    Each kind has its own projection. A card or permanent with a known card
    also gets the projection of that card's shared embedding, which is how a
    model tells one card from another. `embed` returns `[b, entities, width]`
    in the order that `BoardTokens.action_focus` indexes.
    """

    def __init__(self, sample: BoardTokens, *, card_width: int, width: int) -> None:
        # sample: any tokens of the kind this encoder will read; only its feature widths are used.
        super().__init__()
        self.players: nn.Linear = nn.Linear(sample.players.shape[-1], width)
        self.cards: nn.Linear = nn.Linear(sample.cards.shape[-1], width)
        self.permanents: nn.Linear = nn.Linear(sample.permanents.shape[-1], width)
        self.card_vectors: nn.Linear = nn.Linear(card_width, width, bias=False)
        # seats: float [seats, width], added to the player rows to tell the acting seat from its opponent.
        self.seats: nn.Parameter = nn.Parameter(torch.zeros(SEATS, width))

    def embed(self, tokens: BoardTokens, card_table: Tensor) -> Tensor:
        # card_table: float [pool cards, card_width], the shared embedding of every card in the pool.
        known = tokens.card_ids >= 0
        vectors = card_table[tokens.card_ids.clamp_min(0)] * known[..., None]
        objects = torch.cat((run(self.cards, tokens.cards), run(self.permanents, tokens.permanents)), dim=1)
        players = run(self.players, tokens.players) + self.seats
        return torch.cat((players, objects + run(self.card_vectors, vectors)), dim=1)

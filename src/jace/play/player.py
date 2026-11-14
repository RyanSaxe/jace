"""The model that chooses actions, and a random baseline to play against.

The player reads the board, its own deck and a guess of the opponent's
cards, and scores each legal action from the action's own features, the
board objects it names and a summary of the whole board. One head scores
every kind of decision, so nothing depends on how many actions a decision
offers. It has no win chance of its own: training reads that from
`WinChanceModel`, which sees the opponent's true cards. `random_policy` has
the same call shape and no weights, which makes it the opponent for a first
evaluation.
"""

import torch
from torch import Tensor, nn

from jace.cards import CardVectors
from jace.engine import BoardLayout, Decisions
from jace.modules import encode_real, transformer
from jace.play.tokens import CategoryEncoder, EntityEncoder, board_tokens, card_vectors, gather_entities, legal_actions
from jace.play.types import Boards, Guess

__all__ = ("PlayerModel", "as_boards", "random_policy")

# An action is scored from four vectors of the model's width: its own row, its two focus
# entities and the board summary.
_SCORED_VECTORS = 4


def as_boards(decisions: Decisions, *, device: torch.device) -> Boards:
    return {name: torch.as_tensor(array, device=device) for name, array in decisions.boards.items()}


def random_policy(boards: Boards, _guess: Guess) -> Tensor:
    # A `Policy` that gives every legal action the same score.
    legal = legal_actions(boards)  # bool [b, max_actions]
    return torch.zeros(legal.shape, device=legal.device).masked_fill(~legal, -torch.inf)


class PlayerModel(nn.Module):
    """Score every legal action.

    `policy` satisfies the `Policy` type. The guess enters as tokens:
    `guess_cards` for the cards the opponent is likeliest to hold, and
    `deck_guess_cards` for the cards guessed to make up the largest shares
    of the opponent's unseen cards. The player's own deck enters as one
    token for each card with copies it has not yet seen, made from the card
    and its counts of unseen copies and copies in the deck. No
    weight is sized by the card pool, so the model plays with a larger pool
    once `embedding` and `engine_to_card` are replaced by ones that cover it.
    The card vectors are shared with the other models and are not this
    model's to train: build its optimizer from `own_parameters`.
    """

    def __init__(
        self,
        embedding: CardVectors,
        *,
        engine_to_card: Tensor,
        layout: BoardLayout,
        width: int,
        layers: int,
        heads: int,
        guess_cards: int,
        deck_guess_cards: int,
    ) -> None:
        super().__init__()
        card_width = embedding.width
        self.embedding: CardVectors = embedding
        self.guess_cards: int = guess_cards
        self.deck_guess_cards: int = deck_guess_cards
        # engine_to_card: int64 [engine definition ids] -> CardId, -1 for a definition the pool lacks.
        self.engine_to_card: Tensor
        self.register_buffer("engine_to_card", engine_to_card.detach().clone(), persistent=False)
        self.layout: BoardLayout = layout
        self.entities: EntityEncoder = EntityEncoder(layout, card_width=card_width, width=width)
        self.guessed_card: nn.Linear = nn.Linear(card_width, width)
        self.guessed_chance: nn.Linear = nn.Linear(1, width)
        self.guessed_deck_card: nn.Linear = nn.Linear(card_width, width)
        self.guessed_deck_share: nn.Linear = nn.Linear(1, width)
        self.own_deck_card: nn.Linear = nn.Linear(card_width, width)
        self.own_deck_copies: nn.Linear = nn.Linear(len(layout.decklist_copies), width)
        self.encoder: nn.TransformerEncoder = transformer(width, layers=layers, heads=heads)
        self.actions: nn.Linear = nn.Linear(len(layout.action_features), width)
        self.action_categories: CategoryEncoder = CategoryEncoder(len(layout.action_categories), width=width)
        self.action_head: nn.Sequential = nn.Sequential(
            nn.Linear(_SCORED_VECTORS * width, width), nn.GELU(), nn.Linear(width, 1)
        )

    def policy(self, boards: Boards, guess: Guess) -> Tensor:
        # guess.hand, guess.deck: float [b, cards]. Returns action logits [b, max_actions], illegal ones at -inf.
        tokens = board_tokens(boards, layout=self.layout, engine_to_card=self.engine_to_card)
        pool = torch.arange(self.embedding.cards, device=self.embedding.device)
        card_table = self.embedding(pool)  # [cards, card_width]

        entities = self.entities.embed(tokens, card_table)
        own_deck = self.own_deck_card(card_vectors(card_table, tokens.deck_card_ids))
        own_deck = own_deck + self.own_deck_copies(tokens.deck_copies)
        # TODO: The player reads only the likeliest cards of each guess, so most of an early-game guess is dropped.
        # Revisit adding every card's chance as one token each for hand and deck, `guess.hand @ card_table`: it works
        # on any set, but it blurs the cards into one average vector.
        chances, likeliest = guess.hand.topk(min(self.guess_cards, guess.hand.shape[-1]), dim=-1)
        guessed = self.guessed_card(card_table[likeliest]) + self.guessed_chance(chances[..., None])
        shares, largest = guess.deck.topk(min(self.deck_guess_cards, guess.deck.shape[-1]), dim=-1)
        guessed_deck = self.guessed_deck_card(card_table[largest]) + self.guessed_deck_share(shares[..., None])
        sequence = torch.cat((entities, own_deck, guessed, guessed_deck), dim=1)
        guess_mask = torch.ones((likeliest.shape[0], likeliest.shape[1] + largest.shape[1]), dtype=torch.bool)
        mask = torch.cat((tokens.entity_mask, tokens.deck_mask, guess_mask.to(likeliest.device)), dim=1)
        encoded = encode_real(self.encoder, sequence, mask=mask)

        real = mask[..., None]  # bool [b, entities + own deck + guessed, 1]
        summary = (encoded * real).sum(dim=1) / real.sum(dim=1)  # [b, width]

        # The engine fills action slots from the front, so the slots past the longest list are all empty.
        offered = int(tokens.action_mask.sum(dim=1).max())
        focused = gather_entities(encoded[:, : entities.shape[1]], tokens.action_focus[:, :offered])
        actions = self.actions(tokens.actions[:, :offered])
        actions = actions + self.action_categories(tokens.action_categories[:, :offered])
        scored = torch.cat(
            (actions, focused, summary[:, None].expand(-1, offered, -1)),
            dim=-1,
        )  # [b, offered, scored vectors * width]
        logits = torch.full_like(tokens.action_mask, -torch.inf, dtype=scored.dtype)
        logits[:, :offered] = self.action_head(scored).squeeze(-1)
        return logits.masked_fill(~tokens.action_mask, -torch.inf)

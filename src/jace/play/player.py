"""The model that chooses actions, and a random baseline to play against.

The player reads the board and a guess of the opponent's hand, and scores
each legal action from the action's own features, the board objects it names
and a summary of the whole board. One head scores every kind of decision, so
nothing depends on how many actions a decision offers. `random_policy` has
the same call shape and no weights, which makes it the opponent for a first
evaluation.
"""

import torch
from torch import Tensor, nn

from jace.cards import CardEmbedding
from jace.engine import Decisions
from jace.modules import run, transformer
from jace.play.tokens import EntityEncoder, board_tokens
from jace.play.types import Boards, Guess, Policy

__all__ = ("PlayerModel", "as_boards", "random_policy")

# An action is scored from four vectors of the model's width: its own row, its two focus
# entities and the board summary.
_SCORED_VECTORS = 4
# A policy with no value estimate reports an even chance of winning.
_EVEN_CHANCE = 0.5


def as_boards(decisions: Decisions, *, device: torch.device) -> Boards:
    return {name: torch.as_tensor(array, device=device) for name, array in decisions.boards.items()}


def random_policy(engine_to_card: Tensor) -> Policy:
    # engine_to_card: int64 [engine definition ids] -> CardId, as `board_tokens` takes it.
    def policy(boards: Boards, guess: Guess) -> tuple[Tensor, Tensor]:
        legal = board_tokens(boards, engine_to_card=engine_to_card).action_mask  # bool [b, max_actions]
        logits = torch.zeros(legal.shape, device=legal.device).masked_fill(~legal, -torch.inf)
        return logits, torch.full_like(guess.hand[:, 0], _EVEN_CHANCE)

    return policy


class PlayerModel(nn.Module):
    """Score every legal action and estimate the chance of winning.

    `policy` satisfies the `Policy` type. The guess enters as `guess_cards`
    tokens, one for each of the cards the opponent is likeliest to hold. No
    weight is sized by the card pool, so the model plays with a larger pool
    once `embedding` and `engine_to_card` are replaced by ones that cover it.
    The embedding is shared and trained by its own optimizer: leave
    `embedding.parameters()` out of this model's optimizer.
    """

    def __init__(
        self,
        embedding: CardEmbedding,
        *,
        engine_to_card: Tensor,
        sample: Boards,
        width: int,
        layers: int,
        heads: int,
        guess_cards: int,
    ) -> None:
        super().__init__()
        card_width = embedding.starting.shape[1]
        self.embedding: CardEmbedding = embedding
        self.guess_cards: int = guess_cards
        # engine_to_card: int64 [engine definition ids] -> CardId, -1 for a definition the pool lacks.
        self.engine_to_card: Tensor
        self.register_buffer("engine_to_card", engine_to_card.detach().clone(), persistent=False)
        # sample: any batch of boards from the engine that will be played; only its feature widths are used.
        tokens = board_tokens(sample, engine_to_card=engine_to_card)
        self.entities: EntityEncoder = EntityEncoder(tokens, card_width=card_width, width=width)
        self.guessed_card: nn.Linear = nn.Linear(card_width, width)
        self.guessed_chance: nn.Linear = nn.Linear(1, width)
        self.encoder: nn.TransformerEncoder = transformer(width, layers=layers, heads=heads)
        self.actions: nn.Linear = nn.Linear(tokens.actions.shape[-1], width)
        self.action_head: nn.Sequential = nn.Sequential(
            nn.Linear(_SCORED_VECTORS * width, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.value_head: nn.Sequential = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def policy(self, boards: Boards, guess: Guess) -> tuple[Tensor, Tensor]:
        # guess.hand: float [b, cards]. Returns action logits [b, max_actions], illegal ones at -inf,
        # and the acting seat's chance to win, float [b].
        tokens = board_tokens(boards, engine_to_card=self.engine_to_card)
        pool = torch.arange(self.embedding.starting.shape[0], device=self.embedding.starting.device)
        card_table = run(self.embedding, pool)  # [cards, card_width]

        entities = self.entities.embed(tokens, card_table)
        chances, likeliest = guess.hand.topk(min(self.guess_cards, guess.hand.shape[-1]), dim=-1)
        guessed = run(self.guessed_card, card_table[likeliest]) + run(self.guessed_chance, chances[..., None])
        sequence = torch.cat((entities, guessed), dim=1)
        mask = torch.cat((tokens.entity_mask, torch.ones_like(likeliest, dtype=torch.bool)), dim=1)
        encoded = self.encoder.forward(sequence, src_key_padding_mask=~mask)

        real = mask[..., None]  # bool [b, entities + guessed, 1]
        summary = (encoded * real).sum(dim=1) / real.sum(dim=1)  # [b, width]

        focused = _focus_vectors(encoded[:, : entities.shape[1]], tokens.action_focus)
        max_actions = focused.shape[1]
        scored = torch.cat(
            (run(self.actions, tokens.actions), focused, summary[:, None].expand(-1, max_actions, -1)), dim=-1
        )  # [b, max_actions, scored vectors * width]
        logits = run(self.action_head, scored).squeeze(-1).masked_fill(~tokens.action_mask, -torch.inf)
        return logits, run(self.value_head, summary).squeeze(-1).sigmoid()


def _focus_vectors(entities: Tensor, focus: Tensor) -> Tensor:
    # entities: float [b, entities, width], the encoded board entities.
    # focus: int64 [b, max_actions, slots], the entity numbers each action names, or -1 for none.
    # Returns float [b, max_actions, slots * width]: each action's focus entities side by side,
    # zeros where a slot names none.
    batch, max_actions, slots = focus.shape
    named = (focus >= 0).reshape(batch, max_actions * slots, 1)
    index = focus.clamp_min(0).reshape(batch, max_actions * slots, 1).expand(-1, -1, entities.shape[-1])
    return (entities.gather(1, index) * named).reshape(batch, max_actions, -1)

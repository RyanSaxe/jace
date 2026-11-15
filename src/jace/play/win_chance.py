"""The model that estimates the acting seat's chance to win, seeing the opponent's hidden cards.

Training credits an action by how much the chance to win rose after it, so
it needs an estimate of that chance at every decision. The player cannot
see the opponent's hand or library. In self-play `GameBatch.hidden_cards`
reports both, so this estimate can read the true cards where the player
has to guess.

The model reads the same board and own-deck tokens as the player, and the
opponent's true unseen cards in place of the guess. A critic like this is
called privileged, as in Pinto et al. 2017 (arXiv 1710.06542). Only
training calls it, so the hidden cards never reach a decision.
"""

import torch
from torch import Tensor, nn

from jace.cards import CardVectors
from jace.engine import BoardLayout
from jace.modules import encode_real, transformer
from jace.play.tokens import EntityEncoder, board_tokens, card_vectors
from jace.play.types import Boards, Guess

__all__ = ("WinChanceModel",)

# Besides its card, a hidden card's token is made from two numbers: its share of the opponent's unseen
# cards, and whether the opponent holds it in hand.
_HIDDEN_CARD_NUMBERS = 2


# TODO: Check more deeply whether reading the hidden cards is sound and whether it helps. Two results cast doubt
# on it. First, `--lam-advantage` is below 1 by default, so the player's advantages depend on the win chances of
# later decisions. A critic that reads the true state but not the agent's own history can bias the policy gradient
# (Baisero and Amato 2021, arXiv 2105.11674), and this model reads the player's current view and the hidden
# cards, but not the recent events or earlier guesses. Second, a critic with extra information gives the same
# expected update with variance at least as large (Lyu et al. 2021, arXiv 2102.04402), so it is not sure to help.
# Settle both by comparing this model with a win chance that reads only the player's view, and with one that
# also reads the player's history.
class WinChanceModel(nn.Module):
    """Estimate the acting seat's chance to win from its board and the opponent's true hidden cards.

    `win_chance` satisfies the `WinChance` type. The board and the acting
    seat's own deck are read as the player's tokens are. Each card in the
    opponent's hand or library is read as one token, made from its card vector,
    its share of those cards and whether the opponent holds it in hand.

    No weight is sized by the card pool, so the model works with a larger pool
    once `embedding` and `engine_to_card` are replaced by ones that cover
    it. The card vectors are shared with the other models and are not this
    model's to train, so build its optimizer from
    `jace.cards.own_parameters`, which leaves them out.
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
    ) -> None:
        super().__init__()
        card_width = embedding.width
        self.embedding: CardVectors = embedding
        # engine_to_card: int64 [engine definition ids], the `CardId` of each engine definition, or -1 for a
        # definition the pool lacks.
        self.engine_to_card: Tensor
        self.register_buffer("engine_to_card", engine_to_card.detach().clone(), persistent=False)
        self.layout: BoardLayout = layout
        self.entities: EntityEncoder = EntityEncoder(layout, card_width=card_width, width=width)
        self.own_deck_card: nn.Linear = nn.Linear(card_width, width)
        self.own_deck_copies: nn.Linear = nn.Linear(len(layout.decklist_copies), width)
        self.hidden_card: nn.Linear = nn.Linear(card_width, width)
        self.hidden_numbers: nn.Linear = nn.Linear(_HIDDEN_CARD_NUMBERS, width)
        self.encoder: nn.TransformerEncoder = transformer(width, layers=layers, heads=heads)
        self.head: nn.Sequential = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def win_chance(self, boards: Boards, truth: Guess) -> Tensor:
        # truth.deck: float [b, cards], each card's share of the opponent's hand and library.
        # truth.hand: float [b, cards], 1 where the opponent holds the card in hand. Returns float [b], the
        # acting seat's chance to win.
        tokens = board_tokens(boards, layout=self.layout, engine_to_card=self.engine_to_card)
        pool = torch.arange(self.embedding.cards, device=self.embedding.device)
        card_table = self.embedding(pool)  # [cards, card_width]

        entities = self.entities.embed(tokens, card_table)
        own_deck = self.own_deck_card(card_vectors(card_table, tokens.deck_card_ids))
        own_deck = own_deck + self.own_deck_copies(tokens.deck_copies)
        # Only the cards in the opponent's hand and library have a share above zero. The top `held` shares
        # of each row are therefore those cards, followed by zero shares in a row with fewer of them.
        held = int((truth.deck > 0).sum(dim=-1).max())
        shares, hidden_cards = truth.deck.topk(held, dim=-1)  # [b, held]
        numbers = torch.stack((shares, truth.hand.gather(-1, hidden_cards)), dim=-1)
        hidden = self.hidden_card(card_table[hidden_cards]) + self.hidden_numbers(numbers)
        sequence = torch.cat((entities, own_deck, hidden), dim=1)
        mask = torch.cat((tokens.entity_mask, tokens.deck_mask, shares > 0), dim=1)
        encoded = encode_real(self.encoder, sequence, mask=mask)

        real = mask[..., None]  # bool [b, entities + own deck + hidden cards, 1]
        summary = (encoded * real).sum(dim=1) / real.sum(dim=1)  # [b, width]
        return self.head(summary).squeeze(-1).sigmoid()

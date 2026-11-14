"""The model that guesses the opponent's deck and hand.

The guess is the player's only view of hidden cards, and it is trained on
the true hidden cards the engine reports after the fact. It starts from a
table of what the acting seat can know: the field, which is the decks the
opponent's deck is drawn from, less the opponent's cards the seat has seen
played or revealed. The network learns only corrections on top, and it
starts at none, so an untrained model's guess is the table. The network
reads the board, the recent events and its own earlier guesses in the same
game, so a guess can build on the last one without any state carried
between calls. It scores each card by a dot product with that card's
embedding and has no parameter sized by the card pool, so the same weights
work for a larger pool.
"""

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn

from jace.cards import CardVectors, Deck
from jace.engine import BoardLayout
from jace.modules import encode_real, transformer
from jace.play.tokens import CategoryEncoder, EntityEncoder, board_tokens, opponent_cards
from jace.play.types import Boards, Earlier, Guess

__all__ = ("FieldNotSetError", "GuessModel", "TooManyCopiesError")

# A guess has two parts: the opponent's deck and the opponent's hand.
_GUESS_PARTS = 2
# The base of the wavelengths in the age encoding, the value the original transformer used for positions.
_AGE_ENCODING_BASE = 10000.0
# The field counts up to this many copies of one card in a deck, as many as a 40-card deck can hold.
_MOST_COPIES = 40
# Divided by at least this, a total of zero copies gives shares of zero.
_SMALLEST_TOTAL = 1e-12


class FieldNotSetError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("the guess model has no field: call set_field with the decks the opponent is drawn from")


class TooManyCopiesError(ValueError):
    def __init__(self, copies: int) -> None:
        super().__init__(
            f"a deck of the field holds {copies} copies of one card; the field counts up to {_MOST_COPIES}"
        )


def _age_encoding(age: Tensor, *, width: int) -> Tensor:
    # age: int64 [b, t], how many real rows are newer than each row. Returns float [b, t, width]
    # sinusoids, so the newest real row gets the same encoding however much padding there is.
    # One rate for each sine and cosine pair, so every second position up to the width.
    rates = torch.exp(torch.arange(0, width, 2, device=age.device) * (-math.log(_AGE_ENCODING_BASE) / width))
    angles = age[..., None] * rates  # [b, t, rates]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)[..., :width]


def _shifted_shares(shares: Tensor, shifts: Tensor) -> Tensor:
    # shares, shifts: float [b, cards]. Returns softmax(log shares + shifts), which is `shares` when every
    # shift is zero. The log of no share is -inf, so a card with no share keeps none whatever its shift.
    # A row with no share anywhere would give NaN, so it stays zeros.
    held = shares > 0
    logits = torch.where(held.any(dim=-1, keepdim=True), shares.log() + shifts, 0.0)
    return logits.softmax(dim=-1) * held


class GuessModel(nn.Module):
    """Predict the opponent's unseen cards and hand from one seat's view.

    `guess` satisfies the `Guesser` type. It is `table` plus the network's
    corrections, which start at zero. The table needs a field, which
    `set_field` sets and which travels with the model's state dict, so the
    worker processes that receive the weights guess from the same field.
    No weight is sized by the card pool, so the model scores a larger pool
    once `embedding` and `engine_to_card` are replaced by ones that cover it
    and `set_field` is called again. The card vectors are shared with the
    other models and are not this model's to train: build its optimizer
    from `own_parameters`.
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
        self.width: int = width
        # engine_to_card: int64 [engine definition ids] -> CardId, -1 for a definition the pool lacks.
        self.engine_to_card: Tensor
        self.register_buffer("engine_to_card", engine_to_card.detach().clone(), persistent=False)
        # field: float [most copies + 1, cards], row k the share of the field's decks holding exactly k
        # copies of each card. It is NaN until `set_field` is called, so a guess without a field fails.
        self.field: Tensor
        self.register_buffer("field", torch.full((_MOST_COPIES + 1, embedding.cards), torch.nan))
        self.layout: BoardLayout = layout
        self.entities: EntityEncoder = EntityEncoder(layout, card_width=card_width, width=width)
        self.events: nn.Linear = nn.Linear(len(layout.event_features), width)
        self.event_categories: CategoryEncoder = CategoryEncoder(len(layout.event_categories), width=width)
        self.earlier: nn.Linear = nn.Linear(_GUESS_PARTS * card_width, width)
        # queries: float [guess parts, width], the tokens whose encodings score the deck and the hand.
        self.queries: nn.Parameter = nn.Parameter(torch.randn(_GUESS_PARTS, width) * width**-0.5)
        self.encoder: nn.TransformerEncoder = transformer(width, layers=layers, heads=heads)
        # Keys of zero make every correction zero, so the untrained guess is the table. Their gradient is
        # not zero, so the first step moves them and the rest of the network learns from there.
        self.card_keys: nn.Linear = nn.Linear(card_width, width)
        nn.init.zeros_(self.card_keys.weight)
        nn.init.zeros_(self.card_keys.bias)
        self.hand_bias: nn.Parameter = nn.Parameter(torch.zeros(()))

    def set_field(self, decks: Sequence[Deck]) -> None:
        """Make `decks` the field, the decks the opponent's deck is drawn from.

        In self-play the field is the decks of the games being played, and
        in scoring the decks the harness deals, because those are the decks
        an opponent can hold. A deck that plays several games may be listed
        once for each, which weighs it by its games. Raises
        `TooManyCopiesError` when a deck holds more copies of one card than
        the field counts.
        """
        cards = self.embedding.cards
        copies = torch.stack([torch.bincount(torch.tensor(deck, dtype=torch.long), minlength=cards) for deck in decks])
        most = int(copies.max())
        if most > _MOST_COPIES:
            raise TooManyCopiesError(most)
        decks_holding = torch.zeros(_MOST_COPIES + 1, cards).scatter_add_(0, copies, torch.ones(copies.shape))
        self.field = (decks_holding / len(decks)).to(self.field.device)

    def table(self, boards: Boards) -> Guess:
        """Guess the opponent's cards from the field and what the acting seat has seen, with no network.

        The opponent's deck holds at least the copies of a card it has
        shown, on the battlefield, in the graveyard, in exile or on the
        stack, plus those known to be in its hand. Of the field's decks
        only those that hold as many could be its deck, card by card, and
        the table is the mean over those decks. A card's deck share is its
        known copies plus the mean of the copies those decks hold beyond the
        ones accounted for. Its hand chance is 1 when a copy is known to be
        in the hand, and otherwise the mean over those decks of the chance
        that a copy beyond is. Returns deck and hand, each float `[b, cards]`. Raises
        `FieldNotSetError` before `set_field` has been called.
        """
        if bool(self.field.isnan().any()):
            raise FieldNotSetError
        seen = opponent_cards(boards, layout=self.layout, engine_to_card=self.engine_to_card, cards=self.field.shape[1])
        copies = torch.arange(_MOST_COPIES + 1, device=self.field.device)
        # beyond: float [b, cards, copies], each count of copies a deck may hold less the copies shown or known.
        beyond = copies - (seen.shown + seen.known_in_hand)[..., None]
        # weight: the share of the field's decks holding each count, zero for a deck that holds too few to be
        # the opponent's. A card no deck could be holding has no weight, and the clamp keeps its 0 / 0 at zero.
        weight = self.field.T * (beyond >= 0)
        holding = weight.sum(dim=-1).clamp_min(_SMALLEST_TOTAL)
        unaccounted = (weight * beyond).sum(dim=-1) / holding
        hidden_copies = seen.known_in_hand + unaccounted
        deck = hidden_copies / hidden_copies.sum(dim=-1, keepdim=True).clamp_min(_SMALLEST_TOTAL)
        # Each hidden card that is not known is in the opponent's hand with chance p, its hand size less the
        # known cards over its hidden cards less the known cards, so a deck holding u copies beyond those
        # accounted for has one in hand with chance 1 - (1 - p) ** u. The hand chance is the mean of that over
        # the decks, which is lower than the chance for their mean copies. Both counts are whole cards, and
        # when every hidden card is known p is 0.
        known = seen.known_in_hand.sum(dim=-1)
        in_hand = ((seen.hand - known) / (seen.hidden - known).clamp_min(1.0)).clamp(0.0, 1.0)
        none_in_hand = (1.0 - in_hand)[:, None, None] ** beyond.clamp_min(0.0)
        some_in_hand = (weight * (1.0 - none_in_hand)).sum(dim=-1) / holding
        return Guess(deck=deck, hand=torch.where(seen.known_in_hand > 0, 1.0, some_in_hand))

    def guess(self, boards: Boards, earlier: Earlier) -> Guess:
        # earlier.guesses.deck, .hand: float [b, t, cards], this game's guesses so far, newest first; t may be 0.
        # earlier.mask: bool [b, t]. Returns deck and hand, each float [b, cards].
        table = self.table(boards)
        tokens = board_tokens(boards, layout=self.layout, engine_to_card=self.engine_to_card)
        pool = torch.arange(self.embedding.cards, device=self.embedding.device)
        card_table = self.embedding(pool)  # [cards, card_width]

        entities = self.entities.embed(tokens, card_table)
        # Events are oldest first and earlier guesses newest first, so their ages count in opposite directions.
        # A cumulative sum taken from the newest end counts the real events at or after each one.
        event_age = tokens.event_mask.flip(1).cumsum(1).flip(1) - 1
        events = self.events(tokens.events) + self.event_categories(tokens.event_categories)
        events = events + _age_encoding(event_age, width=self.width)
        # One token per earlier guess: its two probability vectors, each as a weighted sum of card embeddings.
        earlier_cards = torch.cat((earlier.guesses.deck @ card_table, earlier.guesses.hand @ card_table), dim=-1)
        earlier_age = earlier.mask.cumsum(1) - 1
        earlier_tokens = self.earlier(earlier_cards) + _age_encoding(earlier_age, width=self.width)

        batch = entities.shape[0]
        queries = self.queries.expand(batch, -1, -1)
        sequence = torch.cat((queries, entities, events, earlier_tokens), dim=1)
        query_mask = torch.ones(batch, _GUESS_PARTS, dtype=torch.bool, device=sequence.device)
        mask = torch.cat((query_mask, tokens.entity_mask, tokens.event_mask, earlier.mask), dim=1)
        encoded = encode_real(self.encoder, sequence, mask=mask)

        keys = self.card_keys(card_table)  # [cards, width]
        corrections = encoded[:, :_GUESS_PARTS] @ keys.T * self.width**-0.5  # [b, guess parts, cards]
        # A table chance of 0 or 1 has an infinite logit, so those cards keep the table's chance.
        hand = (table.hand.logit() + corrections[:, 1] + self.hand_bias).sigmoid()
        return Guess(deck=_shifted_shares(table.deck, corrections[:, 0]), hand=hand)

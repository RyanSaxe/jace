"""The model that guesses the opponent's deck and hand.

The guess is the only input the player has about the opponent's hidden cards.
The model is trained on the true hidden cards, which the engine reports after
the fact, but it reads only what a player can know. It reads the board, the
acting seat's own deck and the recent events. It reads how many cards are in
the opponent's hand and library, and which cards the opponent has shown or
revealed. It also reads the meta, which describes what decks of the format look
like and is fitted from public decklists or from the decks of earlier rounds.

The model reads its own earlier guesses in the same game as well. That lets
it refine its last guess without keeping any state between calls.

`GuessModel.guess` adds the network's corrections to `GuessModel.meta_guess`,
which guesses from the meta and the cards seen without any network. The
corrections start at zero, so an untrained model's guess is the meta guess.

The network reads each card on the board together with that card's row of
the meta, as the mean card vector of the decks that contain the card. It
scores each card by a dot product with a projection of that card's
embedding. No parameter is sized by the card pool, so the same weights
work for a larger pool.
"""

import math

import torch
from torch import Tensor, nn

from jace.cards import CardVectors
from jace.engine import BoardLayout
from jace.modules import encode_real, transformer
from jace.play.meta import Meta
from jace.play.tokens import CategoryEncoder, EntityEncoder, board_tokens, card_vectors, opponent_cards
from jace.play.types import Boards, Earlier, Guess

__all__ = ("GuessModel", "MetaNotSetError")

# A guess has two parts: the opponent's deck and the opponent's hand.
_GUESS_PARTS = 2
# The base of the wavelengths in the age encoding, the value the original transformer used for positions.
_AGE_ENCODING_BASE = 10000.0
# The meta guess expects at least this many unseen copies of every legal card. So while some of the
# opponent's hidden cards are unaccounted for, every legal card's deck share is above zero. In the lifts,
# a mean of zero copies, such as for a pair of cards that no deck contains together, counts as this many,
# so the log of every lift is finite.
_FEWEST_COPIES = 1e-3
# Totals are divided by at least this, so that a total of zero copies gives shares of zero, not NaN.
_SMALLEST_TOTAL = 1e-12


class MetaNotSetError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("the guess model has no meta: call set_meta with what decks of the format look like")


def _age_encoding(age: Tensor, *, width: int) -> Tensor:
    # age is int64 [b, t], how many real rows are newer than each row. Returns float [b, t, width]
    # sinusoids. Because age counts from the newest real row, that row gets the same encoding however
    # much padding there is. Each rate is used for one sine column and one cosine column, which is why
    # the exponents step by 2.
    rates = torch.exp(torch.arange(0, width, 2, device=age.device) * (-math.log(_AGE_ENCODING_BASE) / width))
    angles = age[..., None] * rates  # [b, t, rates]
    return torch.cat((angles.sin(), angles.cos()), dim=-1)[..., :width]


def _shifted_shares(shares: Tensor, shifts: Tensor) -> Tensor:
    # shares and shifts are float [b, cards]. Returns softmax(log shares + shifts), which is `shares` when
    # every shift is zero and the row sums to 1. A share of zero has a log of -inf, so a card with no share
    # gets none whatever its shift. A row of all zeros would give NaN, so it is returned as all zeros.
    held = shares > 0
    logits = torch.where(held.any(dim=-1, keepdim=True), shares.log() + shifts, 0.0)
    return logits.softmax(dim=-1) * held


class GuessModel(nn.Module):
    """Predict the opponent's unseen cards and hand from one seat's view.

    `guess` satisfies the `Guesser` type. It starts from `meta_guess` and
    applies the network's corrections, which start at zero.

    `set_meta` sets the meta, and `meta` reads it back. Until `set_meta` is
    first called, `meta`, `meta_guess` and `guess` raise `MetaNotSetError`.
    The model stores the meta in buffers of its state dict. The worker
    processes that load its weights therefore guess from the same meta, and
    a checkpoint of the model contains the meta it guessed from.

    No weight is sized by the card pool. To score a larger pool, replace
    `embedding` and `engine_to_card` with ones that cover it, and call
    `set_meta` with a meta of that pool.

    The card vectors are shared with the other models and are not this
    model's to train, so build its optimizer from `jace.cards.own_parameters`,
    which leaves them out.
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
        cards = embedding.cards
        self.embedding: CardVectors = embedding
        self.width: int = width
        # engine_to_card: int64 [engine definition ids], the `CardId` of each engine definition, or -1 for a
        # definition the pool lacks.
        self.engine_to_card: Tensor
        self.register_buffer("engine_to_card", engine_to_card.detach().clone(), persistent=False)
        # The fields of the last meta passed to `set_meta`. Until the first call `meta_copies` is NaN, which
        # makes `meta` raise `MetaNotSetError`.
        self.meta_copies: Tensor
        self.meta_conditional: Tensor
        self.meta_holding: Tensor
        self.meta_decks: Tensor
        self.meta_legal: Tensor
        self.register_buffer("meta_copies", torch.full((cards,), torch.nan))
        self.register_buffer("meta_conditional", torch.zeros(cards, cards))
        self.register_buffer("meta_holding", torch.zeros(cards))
        self.register_buffer("meta_decks", torch.zeros(()))
        self.register_buffer("meta_legal", torch.zeros(cards, dtype=torch.bool))
        # meta_lifts is float [cards, cards], the log of each pair's lift. The lift of card j given card i is
        # j's mean copies in the decks that contain i, divided by j's mean copies in all decks, with both
        # raised to at least `_FEWEST_COPIES`. A lift above 1 means that the decks containing i have more
        # copies of j than decks in general. Taking the log of every pair costs more than the rest of the
        # meta guess, so `set_meta` computes the lifts once and the model stores them beside the meta.
        self.meta_lifts: Tensor
        self.register_buffer("meta_lifts", torch.zeros(cards, cards))
        self.layout: BoardLayout = layout
        self.entities: EntityEncoder = EntityEncoder(layout, card_width=card_width, width=width)
        self.meta_rows: nn.Linear = nn.Linear(card_width, width, bias=False)
        self.own_deck_card: nn.Linear = nn.Linear(card_width, width)
        self.own_deck_copies: nn.Linear = nn.Linear(len(layout.decklist_copies), width)
        self.events: nn.Linear = nn.Linear(len(layout.event_features), width)
        self.event_categories: CategoryEncoder = CategoryEncoder(len(layout.event_categories), width=width)
        self.earlier: nn.Linear = nn.Linear(_GUESS_PARTS * card_width, width)
        # queries is float [guess parts, width]: one learned token for the deck and one for the hand. The
        # encoder's output at each of them is used to score every card for that part of the guess.
        self.queries: nn.Parameter = nn.Parameter(torch.randn(_GUESS_PARTS, width) * width**-0.5)
        self.encoder: nn.TransformerEncoder = transformer(width, layers=layers, heads=heads)
        # Zero keys make every correction zero, so an untrained model's guess is the meta guess. The keys'
        # own gradient is not zero, so the first update moves them, and from then on the rest of the
        # network gets a gradient too.
        self.card_keys: nn.Linear = nn.Linear(card_width, width)
        nn.init.zeros_(self.card_keys.weight)
        nn.init.zeros_(self.card_keys.bias)
        self.hand_bias: nn.Parameter = nn.Parameter(torch.zeros(()))

    def set_meta(self, meta: Meta) -> None:
        # The self-play loops blend a round's decks into the meta only after that round's games are over,
        # and scoring uses the meta its checkpoint was saved with. So no game is guessed with a meta fitted
        # from that game's own decks.
        device = self.meta_copies.device
        self.meta_copies = meta.copies.to(device)
        self.meta_conditional = meta.conditional.to(device)
        self.meta_holding = meta.holding.to(device)
        self.meta_decks = meta.decks.to(device)
        self.meta_legal = meta.legal.to(device)
        lifts = meta.conditional.clamp_min(_FEWEST_COPIES) / meta.copies.clamp_min(_FEWEST_COPIES)
        self.meta_lifts = lifts.log().to(device)

    @property
    def meta(self) -> Meta:
        # The meta the model guesses from, on the model's device. Raises `MetaNotSetError` before `set_meta`.
        if bool(self.meta_copies.isnan().any()):
            raise MetaNotSetError
        return Meta(
            copies=self.meta_copies,
            conditional=self.meta_conditional,
            holding=self.meta_holding,
            decks=self.meta_decks,
            legal=self.meta_legal,
        )

    def meta_guess(self, boards: Boards) -> Guess:
        """Guess the opponent's deck and hand from the meta and the cards seen, with no network.

        The guess reads the cards the opponent has shown and the cards it
        has revealed in hand. A card is shown while a copy is on the
        battlefield, in the graveyard, in exile or on the stack.

        It returns a `Guess` of two float `[b, cards]` tensors. `deck` is
        each card's share of the opponent's unseen cards, its hand and
        library. While some of those cards are still unaccounted for, every
        legal card's share is above zero, so the guess rules out no card the
        opponent might hold. `hand` is the chance that the opponent's hand
        contains at least one copy of each card.

        Raises `MetaNotSetError` before `set_meta` has been called.
        """
        meta = self.meta
        seen = opponent_cards(boards, layout=self.layout, engine_to_card=self.engine_to_card, cards=len(meta.copies))
        accounted = seen.shown + seen.known_in_hand
        known = (accounted > 0).float()
        # The known cards' log lifts are summed and divided by the square root of their number. A plain sum
        # would treat the known cards as independent evidence, but the cards of one deck are not independent.
        # A mean would let twenty known cards move the guess no more than two do. Dividing by the square root
        # gives a result between the two. On held-out human decks with some of their cards shown, the guess
        # predicted the hidden cards better this way than with the sum or the mean.
        lift = (known @ self.meta_lifts) / known.sum(dim=-1, keepdim=True).clamp_min(1.0).sqrt()
        expected = meta.copies.clamp_min(_FEWEST_COPIES) * lift.exp()
        # The opponent's deck contains every known card. So for a known card, the guess expects at least its
        # mean copies in the decks that contain it, which is its entry on the diagonal of `conditional`.
        expected = torch.where(known > 0, expected.maximum(meta.conditional.diagonal()), expected)
        beyond = torch.where(meta.legal, (expected - accounted).clamp_min(_FEWEST_COPIES), 0.0)
        known_count = seen.known_in_hand.sum(dim=-1)
        unknown = (seen.hidden - known_count).clamp_min(0.0)
        unknown_copies = beyond * (unknown / beyond.sum(dim=-1).clamp_min(_SMALLEST_TOTAL))[:, None]
        deck = (seen.known_in_hand + unknown_copies) / seen.hidden.clamp_min(1.0)[:, None]
        # Each unknown hidden card is in the hand with chance p. A card's u unknown copies are an expected
        # number, not a count, so the guess treats the card's copies as a Poisson number with mean u. Then none
        # of them is in the hand with chance exp(-p * u). The formula for a whole number of copies,
        # (1 - p) ** u, fails once the library is empty. p is then 1, and every card would be in the hand
        # however few copies the meta gives it.
        in_hand = ((seen.hand - known_count) / unknown.clamp_min(1.0)).clamp(0.0, 1.0)
        none_in_hand = torch.exp(-in_hand[:, None] * unknown_copies)
        return Guess(deck=deck, hand=torch.where(seen.known_in_hand > 0, 1.0, 1.0 - none_in_hand))

    def guess(self, boards: Boards, earlier: Earlier) -> Guess:
        # earlier.guesses.deck and earlier.guesses.hand are float [b, t, cards], the acting seat's guesses so
        # far in this game, newest first, and t may be 0. earlier.mask is bool [b, t]. Returns deck and hand,
        # each float [b, cards].
        start = self.meta_guess(boards)
        tokens = board_tokens(boards, layout=self.layout, engine_to_card=self.engine_to_card)
        pool = torch.arange(self.embedding.cards, device=self.embedding.device)
        card_table = self.embedding(pool)  # [cards, card_width]

        # Each card's row of the meta becomes one vector: the mean card vector of the decks that contain it.
        rows = self.meta_conditional
        meta_vectors = (rows / rows.sum(dim=-1, keepdim=True).clamp_min(_SMALLEST_TOTAL)) @ card_table
        entities = self.entities.embed(tokens, card_table)
        # The players come first among the entities, and every entity after them is a card or a permanent.
        players = entities.shape[1] - tokens.card_ids.shape[1]
        meta_rows = self.meta_rows(card_vectors(meta_vectors, tokens.card_ids))
        entities = torch.cat((entities[:, :players], entities[:, players:] + meta_rows), dim=1)
        # One token for each card of the acting seat's own deck with copies it has not yet seen.
        own_deck = self.own_deck_card(card_vectors(card_table, tokens.deck_card_ids))
        own_deck = own_deck + self.own_deck_copies(tokens.deck_copies)
        # Events are oldest first and earlier guesses newest first, so their ages are counted from opposite ends.
        # A cumulative sum taken from the newest end counts the real events at or after each one.
        event_age = tokens.event_mask.flip(1).cumsum(1).flip(1) - 1
        events = self.events(tokens.events) + self.event_categories(tokens.event_categories)
        events = events + _age_encoding(event_age, width=self.width)
        # Each earlier guess becomes one token. Its deck shares and its hand chances each weight a sum of card
        # embeddings.
        earlier_cards = torch.cat((earlier.guesses.deck @ card_table, earlier.guesses.hand @ card_table), dim=-1)
        earlier_age = earlier.mask.cumsum(1) - 1
        earlier_tokens = self.earlier(earlier_cards) + _age_encoding(earlier_age, width=self.width)

        batch = entities.shape[0]
        queries = self.queries.expand(batch, -1, -1)
        sequence = torch.cat((queries, entities, own_deck, events, earlier_tokens), dim=1)
        query_mask = torch.ones(batch, _GUESS_PARTS, dtype=torch.bool, device=sequence.device)
        mask = torch.cat((query_mask, tokens.entity_mask, tokens.deck_mask, tokens.event_mask, earlier.mask), dim=1)
        encoded = encode_real(self.encoder, sequence, mask=mask)

        keys = self.card_keys(card_table)  # [cards, width]
        corrections = encoded[:, :_GUESS_PARTS] @ keys.T * self.width**-0.5  # [b, guess parts, cards]
        # A start chance of 0 or 1 has an infinite logit, so those cards keep the start's chance.
        hand = (start.hand.logit() + corrections[:, 1] + self.hand_bias).sigmoid()
        return Guess(deck=_shifted_shares(start.deck, corrections[:, 0]), hand=hand)

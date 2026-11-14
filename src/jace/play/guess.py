"""The model that guesses the opponent's deck and hand.

The guess is the player's only view of hidden cards, and it is trained on
the true hidden cards the engine reports after the fact. It reads only what
a player can know: the meta, which holds the format's legal cards and which
cards decks of the format hold together, fitted from public decklists or
from decks of earlier rounds; the board, with its own deck; the opponent's
hand and library sizes and every card the opponent has shown or revealed;
the recent events; and its own earlier guesses in the same game, so a guess
can build on the last one without any state carried between calls. It
starts from the meta guess, the shown cards' rows of the meta combined into
expected copies of each card, less the copies shown, with a floor so that
no legal card gets no share or no chance. The network learns only
corrections on top, and it starts at none, so an untrained model's guess is
the meta guess. The network reads each card on the board with that card's
row of the meta, through the card vectors. It scores each card by a dot
product with that card's embedding and has no parameter sized by the card
pool, so the same weights work for a larger pool.
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
# The meta guess expects at least this many unseen copies of every legal card, so none gets no share or no
# chance, and reads a pair of cards no deck holds together as this many copies, so every lift is finite.
_FEWEST_COPIES = 1e-3
# Divided by at least this, a total of zero copies gives shares of zero.
_SMALLEST_TOTAL = 1e-12


class MetaNotSetError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("the guess model has no meta: call set_meta with what decks of the format look like")


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

    `guess` satisfies the `Guesser` type. It is `meta_guess` plus the
    network's corrections, which start at zero. Both need a meta, which
    `set_meta` sets and `meta` reads back. The meta is held in buffers of
    the model's state dict, so the worker processes that receive the
    weights guess from the same meta, and a checkpoint of the model holds
    the meta it played with. No weight is sized by the card pool, so the
    model scores a larger pool once `embedding` and `engine_to_card` are
    replaced by ones that cover it and `set_meta` is called with a meta of
    that pool. The card vectors are shared with the other models and are
    not this model's to train: build its optimizer from `own_parameters`.
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
        # engine_to_card: int64 [engine definition ids] -> CardId, -1 for a definition the pool lacks.
        self.engine_to_card: Tensor
        self.register_buffer("engine_to_card", engine_to_card.detach().clone(), persistent=False)
        # The fields of the meta `set_meta` was last called with. Its copies are NaN until then, so a guess
        # without a meta fails.
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
        # meta_lifts: float [cards, cards], log(conditional / copies) with both at least the fewest copies:
        # how much likelier a deck that holds card i makes each card j. A logarithm of every pair costs more
        # than the rest of the meta guess, so `set_meta` takes them once and they travel with the meta.
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
        # queries: float [guess parts, width], the tokens whose encodings score the deck and the hand.
        self.queries: nn.Parameter = nn.Parameter(torch.randn(_GUESS_PARTS, width) * width**-0.5)
        self.encoder: nn.TransformerEncoder = transformer(width, layers=layers, heads=heads)
        # Keys of zero make every correction zero, so the untrained guess is the meta guess. Their gradient
        # is not zero, so the first step moves them and the rest of the network learns from there.
        self.card_keys: nn.Linear = nn.Linear(card_width, width)
        nn.init.zeros_(self.card_keys.weight)
        nn.init.zeros_(self.card_keys.bias)
        self.hand_bias: nn.Parameter = nn.Parameter(torch.zeros(()))

    def set_meta(self, meta: Meta) -> None:
        # Self-play sets the meta of the decks of earlier rounds before each round, and scoring keeps the
        # meta a checkpoint was saved with, so neither ever reads the decks of the games being played.
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
        """Guess the opponent's cards from the meta and what the acting seat has seen, with no network.

        A card is known to be in the opponent's deck when it has shown a
        copy, on the battlefield, in the graveyard, in exile or on the
        stack, or a copy was revealed in its hand. Each card's expected
        copies are its mean copies in the meta, times the exponential of
        the known cards' log lifts for it summed and divided by the square
        root of their number, and a known card expects at least its mean
        copies in the decks that hold it. Less the copies known, and at
        least the fewest copies for a legal card and none for another, a
        card's expected copies are scaled so that all of them fill the
        hidden cards that are not known. A card's deck share is its copies
        known to be in hand plus its scaled copies, as a share of the
        hidden cards. Its hand chance is 1 when a copy is known to be in
        hand, and otherwise 1 - exp(-p * u), where p is the share of the
        unknown hidden cards that are in hand and u its scaled copies.
        Returns deck and hand, each float `[b, cards]`. Raises
        `MetaNotSetError` before `set_meta` has been called.
        """
        meta = self.meta
        seen = opponent_cards(boards, layout=self.layout, engine_to_card=self.engine_to_card, cards=len(meta.copies))
        accounted = seen.shown + seen.known_in_hand
        known = (accounted > 0).float()
        # Summed, the log lifts would count the known cards as independent evidence, which cards of one deck are
        # not; averaged, they would say no more after twenty cards than after two. Their sum over the square
        # root of their number sits between, and of the three it predicted best which cards held-out human
        # decks were hiding when some of their cards were shown.
        lift = (known @ self.meta_lifts) / known.sum(dim=-1, keepdim=True).clamp_min(1.0).sqrt()
        expected = meta.copies.clamp_min(_FEWEST_COPIES) * lift.exp()
        expected = torch.where(known > 0, expected.maximum(meta.conditional.diagonal()), expected)
        beyond = torch.where(meta.legal, (expected - accounted).clamp_min(_FEWEST_COPIES), 0.0)
        known_count = seen.known_in_hand.sum(dim=-1)
        unknown = (seen.hidden - known_count).clamp_min(0.0)
        unknown_copies = beyond * (unknown / beyond.sum(dim=-1).clamp_min(_SMALLEST_TOTAL))[:, None]
        deck = (seen.known_in_hand + unknown_copies) / seen.hidden.clamp_min(1.0)[:, None]
        # Each unknown hidden card is in hand with chance p. A card's u unknown copies are a mean, not a count, so
        # they are taken as a Poisson number with that mean, which has none in hand with chance exp(-p * u). The
        # chance of whole copies, (1 - p) ** u, would put every card in hand once the library is empty and p is 1,
        # however few copies the meta expects of it.
        in_hand = ((seen.hand - known_count) / unknown.clamp_min(1.0)).clamp(0.0, 1.0)
        none_in_hand = torch.exp(-in_hand[:, None] * unknown_copies)
        return Guess(deck=deck, hand=torch.where(seen.known_in_hand > 0, 1.0, 1.0 - none_in_hand))

    def guess(self, boards: Boards, earlier: Earlier) -> Guess:
        # earlier.guesses.deck, .hand: float [b, t, cards], this game's guesses so far, newest first; t may be 0.
        # earlier.mask: bool [b, t]. Returns deck and hand, each float [b, cards].
        start = self.meta_guess(boards)
        tokens = board_tokens(boards, layout=self.layout, engine_to_card=self.engine_to_card)
        pool = torch.arange(self.embedding.cards, device=self.embedding.device)
        card_table = self.embedding(pool)  # [cards, card_width]

        # Each card's row of the meta as one vector: the mean card vector of the decks that hold it.
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
        sequence = torch.cat((queries, entities, own_deck, events, earlier_tokens), dim=1)
        query_mask = torch.ones(batch, _GUESS_PARTS, dtype=torch.bool, device=sequence.device)
        mask = torch.cat((query_mask, tokens.entity_mask, tokens.deck_mask, tokens.event_mask, earlier.mask), dim=1)
        encoded = encode_real(self.encoder, sequence, mask=mask)

        keys = self.card_keys(card_table)  # [cards, width]
        corrections = encoded[:, :_GUESS_PARTS] @ keys.T * self.width**-0.5  # [b, guess parts, cards]
        # A start chance of 0 or 1 has an infinite logit, so those cards keep the start's chance.
        hand = (start.hand.logit() + corrections[:, 1] + self.hand_bias).sigmoid()
        return Guess(deck=_shifted_shares(start.deck, corrections[:, 0]), hand=hand)

"""Pre-game matchup value predictor `V_0(D_me, P_0 or D_opp)` and opening-hand comparison.

`MatchupPredictor` reuses `CardSetEncoder` over frozen `CardVectors` with zero
per-card ID parameters to predict pre-game win-rate logits between a hero deck
and an opponent deck belief. `score_deck_vs_field` adapts `MatchupPredictor`
against a `FormatPrior` field mean to match `jace.decks.score.Scorer`.
`evaluate_opening_hand_vs_v0` compares an opening-hand value estimate directly
against `V_0` with no special-case mulligan network or hand-coded rules.
"""

import math
from typing import override

import torch
from torch import Tensor, nn

from jace.belief.types import FormatPrior
from jace.cards import CardVectors
from jace.decks.score import CardSetEncoder

__all__ = (
    "MatchupPredictor",
    "evaluate_opening_hand_vs_v0",
    "score_deck_vs_field",
)

_MATRIX_NDIM = 2


class MatchupPredictor(nn.Module):
    def __init__(
        self,
        embedding: CardVectors,
        *,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = width
        self.deck_encoder: CardSetEncoder = CardSetEncoder(
            embedding,
            amounts=1,
            width=width,
            layers=layers,
            heads=heads,
        )
        self.head: nn.Sequential = nn.Sequential(
            nn.Linear(width * 4, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, 1),
        )

    @override
    def forward(self, my_decks: Tensor, opp_deck_beliefs: Tensor) -> Tensor:
        my_2d = _as_2d_decks(my_decks, name="my_decks")
        opp_2d = _broadcast_opp_beliefs(opp_deck_beliefs, batch=my_2d.shape[0], num_cards=my_2d.shape[1])

        my_vec = _encode_decks_safe(self.deck_encoder, my_2d, width=self.width)
        opp_vec = _encode_decks_safe(self.deck_encoder, opp_2d, width=self.width)
        features = torch.cat((my_vec, opp_vec, my_vec - opp_vec, my_vec * opp_vec), dim=-1)
        return self.head(features).squeeze(-1)

    def predict_equity(self, my_decks: Tensor, opp_deck_beliefs: Tensor) -> Tensor:
        return torch.sigmoid(self.forward(my_decks, opp_deck_beliefs))


def score_deck_vs_field(
    decks: Tensor,
    *,
    format_prior: FormatPrior,
    predictor: MatchupPredictor,
) -> Tensor:
    field = format_prior.mean_copies.to(device=decks.device, dtype=torch.float32)
    return predictor(decks, field)


def evaluate_opening_hand_vs_v0(
    hand_value_v: float,
    *,
    v0_equity: float,
    mulligans_taken: int = 0,
) -> dict[str, float | str]:
    if not math.isfinite(hand_value_v) or not math.isfinite(v0_equity):
        msg = "hand_value_v and v0_equity must be finite numbers"
        raise ValueError(msg)
    if mulligans_taken < 0:
        msg = f"mulligans_taken must be non-negative, got {mulligans_taken}"
        raise ValueError(msg)
    keep_v = float(hand_value_v)
    mulligan_v0 = float(v0_equity)
    delta = keep_v - mulligan_v0
    recommendation = "keep" if delta >= 0.0 else "mulligan"
    return {
        "keep_v": keep_v,
        "mulligan_v0": mulligan_v0,
        "delta": delta,
        "mulligans_taken": float(mulligans_taken),
        "recommendation": recommendation,
    }


def _encode_decks_safe(encoder: CardSetEncoder, decks_2d: Tensor, *, width: int) -> Tensor:
    decks_f = decks_2d.to(dtype=torch.float32)
    non_empty = (decks_f > 0.0).any(dim=-1)
    if bool(non_empty.all()):
        return encoder(decks_f.unsqueeze(-1))
    out = torch.zeros((decks_f.shape[0], width), device=decks_f.device, dtype=torch.float32)
    if bool(non_empty.any()):
        out[non_empty] = encoder(decks_f[non_empty].unsqueeze(-1))
    return out


def _as_2d_decks(decks: Tensor, *, name: str) -> Tensor:
    if decks.ndim == _MATRIX_NDIM:
        return decks
    if decks.ndim == 1:
        return decks.unsqueeze(0)
    msg = f"{name} must be 1D or 2D, got shape {tuple(decks.shape)}"
    raise ValueError(msg)


def _broadcast_opp_beliefs(opp_deck_beliefs: Tensor, *, batch: int, num_cards: int) -> Tensor:
    if opp_deck_beliefs.ndim == 1 and opp_deck_beliefs.shape == (num_cards,):
        return opp_deck_beliefs.unsqueeze(0).expand(batch, -1)
    if opp_deck_beliefs.ndim == _MATRIX_NDIM and opp_deck_beliefs.shape == (batch, num_cards):
        return opp_deck_beliefs
    msg = (
        f"opp_deck_beliefs must have shape ({num_cards},) or ({batch}, {num_cards}), "
        f"got {tuple(opp_deck_beliefs.shape)}"
    )
    raise ValueError(msg)

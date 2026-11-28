"""Pure-tensor auxiliary belief head for supervising policy decision-memory representations.

Maps a policy's differentiable multi-turn `DecisionMemory` summary tensor `[B, S, D]`
(or pooled `[B, D]`) together with the fixed format `deck_prior` and frozen `CardVectors`
into exact conserved `(deck_copies, hand_probs)` distributions. Training this head jointly
with PPO shapes `ActorCritic`'s internal memory encoder to track hidden opponent cards
while leaving inference as a standard zero-overhead `ActorCritic` checkpoint.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
from torch import Tensor, nn

from jace.belief.model import _LOGIT_CLAMP, _normalize_deck_copies, _solve_hand_probs

if TYPE_CHECKING:
    from jace.cards import CardVectors

__all__ = ("PolicyBeliefAuxHead",)

_SUMMARY_SEQ_NDIM = 3


class PolicyBeliefAuxHead(nn.Module):
    def __init__(
        self,
        *,
        memory_dim: int,
        embedding: CardVectors,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.memory_dim = memory_dim
        self.hidden_dim = hidden_dim
        self.embedding: CardVectors = embedding
        self.mem_proj = nn.Sequential(
            nn.LayerNorm(memory_dim),
            nn.Linear(memory_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        # Zero-init card keys so initial auxiliary belief matches the calibrated human FormatPrior.
        self.card_keys = nn.Linear(embedding.width, hidden_dim)
        nn.init.zeros_(self.card_keys.weight)
        nn.init.zeros_(self.card_keys.bias)
        self.hand_bias = nn.Parameter(torch.zeros(()))

    @override
    def forward(
        self,
        memory_summaries: Tensor,
        *,
        deck_prior: Tensor,
        revealed_counts: Tensor,
        opp_hand_size: Tensor,
        opp_library_size: Tensor,
        copy_limits: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        pooled = memory_summaries.mean(dim=1) if memory_summaries.ndim == _SUMMARY_SEQ_NDIM else memory_summaries
        queries = self.mem_proj(pooled.to(dtype=torch.float32)).view(pooled.shape[0], 2, self.hidden_dim)

        num_cards = deck_prior.shape[1]
        card_ids = torch.arange(num_cards, device=deck_prior.device)
        keys = self.card_keys(self.embedding(card_ids).to(device=deck_prior.device, dtype=torch.float32))
        scale = self.hidden_dim**-0.5
        deck_delta = (queries[:, 0] @ keys.T * scale).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        hand_logits = (queries[:, 1] @ keys.T * scale + self.hand_bias).clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)

        hand_f = opp_hand_size.to(device=deck_prior.device, dtype=torch.float32)
        lib_f = opp_library_size.to(device=deck_prior.device, dtype=torch.float32)
        unseen_total = hand_f + lib_f
        deck_copies = _normalize_deck_copies(
            deck_prior,
            deck_delta=deck_delta,
            unseen_total=unseen_total,
            revealed_counts=revealed_counts,
            copy_limits=copy_limits,
        )
        hand_probs = _solve_hand_probs(
            deck_copies, hand_logits=hand_logits, hand_size=hand_f, unseen_total=unseen_total
        )
        return deck_copies, hand_probs

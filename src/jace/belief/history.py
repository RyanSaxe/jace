"""Pure-tensor multi-turn game history buffers and sinusoidal recency encoding for `BeliefNet`.

Captures per-step public gameplay tells, newly revealed opponent card identities,
and earlier belief summaries across turns with zero engine (`managym`/`manabot`)
dependencies. Recency encoding (`age_encoding`) counts steps back from the newest
real event so variable-length padding never shifts the representation of recent tells.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = (
    "HISTORY_EVENT_DIM",
    "MAX_EARLIER_BELIEFS",
    "MAX_HISTORY_STEPS",
    "BeliefHistoryBatch",
    "BeliefHistoryBuffer",
    "age_encoding",
    "pack_history_batches",
)

# 16 gameplay signals + 4 per-step deltas:
# 16: log1p(opp_hand_size), 17: log1p(opp_library_size),
# 18: log1p(newly_revealed_total), 19: delta_spells_cast.
HISTORY_EVENT_DIM = 20
MAX_HISTORY_STEPS = 32
MAX_EARLIER_BELIEFS = 8

_AGE_ENCODING_BASE = 10000.0
_SPELLS_CAST_SLOT = 12
_PROB_EPS = 1e-6


def age_encoding(age: Tensor, *, width: int) -> Tensor:
    half = (width + 1) // 2
    step = -math.log(_AGE_ENCODING_BASE) * 2.0 / max(width, 2)
    rates = torch.exp(torch.arange(0, half, device=age.device, dtype=torch.float32) * step)
    angles = age.to(dtype=torch.float32).clamp_min(0.0).unsqueeze(-1) * rates
    return torch.cat((angles.sin(), angles.cos()), dim=-1)[..., :width]


@dataclass(frozen=True, slots=True)
class BeliefHistoryBatch:
    events: Tensor  # float32 [B, T, HISTORY_EVENT_DIM], oldest first
    card_ids: Tensor  # int64 [B, T], -1 if no card revealed on that step
    mask: Tensor  # bool [B, T], True for real steps
    earlier_deck: Tensor | None = None  # float32 [B, E, C], newest first
    earlier_hand: Tensor | None = None  # float32 [B, E, C], newest first
    earlier_mask: Tensor | None = None  # bool [B, E]

    def to(self, device: torch.device | str) -> BeliefHistoryBatch:
        return BeliefHistoryBatch(
            events=self.events.to(device=device),
            card_ids=self.card_ids.to(device=device),
            mask=self.mask.to(device=device),
            earlier_deck=None if self.earlier_deck is None else self.earlier_deck.to(device=device),
            earlier_hand=None if self.earlier_hand is None else self.earlier_hand.to(device=device),
            earlier_mask=None if self.earlier_mask is None else self.earlier_mask.to(device=device),
        )

    def slice_rows(self, indices: Tensor) -> BeliefHistoryBatch:
        return BeliefHistoryBatch(
            events=self.events[indices],
            card_ids=self.card_ids[indices],
            mask=self.mask[indices],
            earlier_deck=None if self.earlier_deck is None else self.earlier_deck[indices],
            earlier_hand=None if self.earlier_hand is None else self.earlier_hand[indices],
            earlier_mask=None if self.earlier_mask is None else self.earlier_mask[indices],
        )


class BeliefHistoryBuffer:
    def __init__(
        self,
        num_cards: int,
        *,
        max_steps: int = MAX_HISTORY_STEPS,
        max_earlier: int = MAX_EARLIER_BELIEFS,
    ) -> None:
        self._num_cards = num_cards
        self._max_steps = max(1, max_steps)
        self._max_earlier = max(1, max_earlier)
        self._events: list[Tensor] = []
        self._card_ids: list[int] = []
        self._earlier_deck: list[Tensor] = []
        self._earlier_hand: list[Tensor] = []
        self._prev_revealed: Tensor | None = None
        self._prev_spells: float = 0.0

    def reset(self) -> None:
        self._events.clear()
        self._card_ids.clear()
        self._earlier_deck.clear()
        self._earlier_hand.clear()
        self._prev_revealed = None
        self._prev_spells = 0.0

    def record_step(
        self,
        *,
        revealed_counts: Tensor,
        gameplay_signals: Tensor,
        opp_hand_size: int,
        opp_library_size: int,
    ) -> None:
        rev = revealed_counts.detach().to(device="cpu", dtype=torch.float32)
        sig = gameplay_signals.detach().to(device="cpu", dtype=torch.float32)
        delta_rev = rev if self._prev_revealed is None else (rev - self._prev_revealed).clamp_min(0.0)
        newly_total = float(delta_rev.sum().item())
        card_id = int(delta_rev.argmax().item()) if newly_total > 0.0 else -1
        cur_spells = float(sig[_SPELLS_CAST_SLOT].item()) if sig.numel() > _SPELLS_CAST_SLOT else 0.0
        delta_spells = max(0.0, cur_spells - self._prev_spells)

        extra = torch.tensor(
            [
                math.log1p(max(0, opp_hand_size)),
                math.log1p(max(0, opp_library_size)),
                math.log1p(max(0.0, newly_total)),
                delta_spells,
            ],
            dtype=torch.float32,
        )
        event_vec = torch.cat((sig[:16], extra), dim=0)
        self._events.append(event_vec)
        self._card_ids.append(card_id)
        if len(self._events) > self._max_steps:
            del self._events[0]
            del self._card_ids[0]
        self._prev_revealed = rev.clone()
        self._prev_spells = cur_spells

    def record_belief(self, *, deck_copies: Tensor, hand_probs: Tensor) -> None:
        deck_f = deck_copies.detach().to(device="cpu", dtype=torch.float32).clamp_min(0.0)
        hand_f = hand_probs.detach().to(device="cpu", dtype=torch.float32).clamp_min(0.0)
        deck_share = deck_f / deck_f.sum().clamp_min(_PROB_EPS)
        hand_share = hand_f / hand_f.sum().clamp_min(_PROB_EPS)
        self._earlier_deck.insert(0, deck_share)
        self._earlier_hand.insert(0, hand_share)
        if len(self._earlier_deck) > self._max_earlier:
            self._earlier_deck.pop()
            self._earlier_hand.pop()

    def snapshot_tensors(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        events = torch.zeros((self._max_steps, HISTORY_EVENT_DIM), dtype=torch.float32)
        card_ids = torch.full((self._max_steps,), -1, dtype=torch.int64)
        mask = torch.zeros((self._max_steps,), dtype=torch.bool)
        if self._events:
            n = len(self._events)
            events[:n] = torch.stack(self._events, dim=0)
            card_ids[:n] = torch.tensor(self._card_ids, dtype=torch.int64)
            mask[:n] = True

        e_deck = torch.zeros((self._max_earlier, self._num_cards), dtype=torch.float32)
        e_hand = torch.zeros((self._max_earlier, self._num_cards), dtype=torch.float32)
        e_mask = torch.zeros((self._max_earlier,), dtype=torch.bool)
        if self._earlier_deck:
            m = len(self._earlier_deck)
            e_deck[:m] = torch.stack(self._earlier_deck, dim=0)
            e_hand[:m] = torch.stack(self._earlier_hand, dim=0)
            e_mask[:m] = True
        return events, card_ids, mask, e_deck, e_hand, e_mask

    def to_batch(self, *, device: torch.device | str = "cpu") -> BeliefHistoryBatch:
        events, card_ids, mask, e_deck, e_hand, e_mask = self.snapshot_tensors()
        return BeliefHistoryBatch(
            events=events.unsqueeze(0).to(device=device),
            card_ids=card_ids.unsqueeze(0).to(device=device),
            mask=mask.unsqueeze(0).to(device=device),
            earlier_deck=e_deck.unsqueeze(0).to(device=device),
            earlier_hand=e_hand.unsqueeze(0).to(device=device),
            earlier_mask=e_mask.unsqueeze(0).to(device=device),
        )


def pack_history_batches(snapshots: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]]) -> BeliefHistoryBatch:
    if not snapshots:
        msg = "pack_history_batches requires at least one snapshot"
        raise ValueError(msg)
    return BeliefHistoryBatch(
        events=torch.stack([s[0] for s in snapshots], dim=0),
        card_ids=torch.stack([s[1] for s in snapshots], dim=0),
        mask=torch.stack([s[2] for s in snapshots], dim=0),
        earlier_deck=torch.stack([s[3] for s in snapshots], dim=0),
        earlier_hand=torch.stack([s[4] for s in snapshots], dim=0),
        earlier_mask=torch.stack([s[5] for s in snapshots], dim=0),
    )

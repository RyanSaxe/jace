"""Learned opponent belief model (`BeliefNet`) and calibration helpers.

`BeliefNet` updates a format deck prior with revealed cards, gameplay signals
(open mana, turn/phase, priority/combat tells, recent spells/lands, life
totals), optional multi-turn public event and earlier-belief history encoded
via `jace.modules.encode_real`, and dynamic opponent hand and library sizes
with no 7-card cap. Its output heads guarantee
`sum(deck_copies) == opp_hand_size + opp_library_size`,
`sum(hand_probs) == opp_hand_size`, and `0 <= hand_probs <= deck_copies`, while
masking out cards exhausted by `copy_limits - revealed_counts <= 0`.
"""

from pathlib import Path
from typing import override

import torch
from torch import Tensor, nn

from jace.belief.history import HISTORY_EVENT_DIM, BeliefHistoryBatch, age_encoding
from jace.belief.prior import BayesianDeckPoolPrior, PriorBackend, SetTransformerPrior
from jace.belief.project import project_belief_overrides
from jace.belief.types import BeliefState, FormatPrior, PriorBackendName
from jace.cards import CardTable, CardVectors, own_parameters
from jace.modules import encode_real, transformer

__all__ = (
    "GAMEPLAY_SIGNAL_DIM",
    "BeliefNet",
    "belief_loss",
    "evaluate_belief_log_loss",
    "infer_belief_state",
    "load_belief_net",
    "project_belief_overrides",
    "save_belief_net",
    "train_belief_net",
)

# Gameplay signal slots:
# 0..4: open mana (W, U, B, R, G), 5: open generic/total mana, 6: turn number,
# 7..8: phase/step encodings, 9: opp passed priority, 10: opp attacked,
# 11: opp blocked, 12: recent spell count, 13: recent land count,
# 14: hero life, 15: opp life.
GAMEPLAY_SIGNAL_DIM = 16

_MATRIX_NDIM = 2
_SIZE_FEATURE_DIM = 4
_CARD_SCALAR_DIM = 4
_LOGIT_CLAMP = 6.0
_BISECTION_STEPS = 32
_PROB_EPS = 1e-6
_ZERO_EPS = 1e-7
_TOLERANCE = 1e-5
_PRIOR_FLOOR = 1e-4
_HISTORY_QUERIES = 2


class BeliefNet(nn.Module):
    def __init__(
        self,
        *,
        signal_dim: int = GAMEPLAY_SIGNAL_DIM,
        hidden_dim: int = 64,
        embedding: CardVectors | None = None,
        dropout: float = 0.0,
        use_history: bool = True,
        history_layers: int = 2,
        history_heads: int = 4,
    ) -> None:
        super().__init__()
        self.signal_dim = signal_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.use_history = use_history
        self.history_layers = history_layers
        self.history_heads = history_heads
        self.embedding: CardVectors | None = embedding
        self.card_project: nn.Linear | None = None if embedding is None else nn.Linear(embedding.width, hidden_dim)
        self.scalar_encoder = nn.Sequential(
            nn.Linear(_CARD_SCALAR_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(signal_dim + _SIZE_FEATURE_DIM, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.history_event_proj: nn.Linear | None = None
        self.earlier_proj: nn.Linear | None = None
        self.history_queries: nn.Parameter | None = None
        self.history_encoder: nn.TransformerEncoder | None = None
        self.history_fuse: nn.Linear | None = None
        if use_history:
            self.history_event_proj = nn.Linear(HISTORY_EVENT_DIM, hidden_dim)
            if embedding is not None:
                self.earlier_proj = nn.Linear(2 * embedding.width, hidden_dim)
            self.history_queries = nn.Parameter(torch.randn(_HISTORY_QUERIES, hidden_dim) * hidden_dim**-0.5)
            self.history_encoder = transformer(hidden_dim, layers=history_layers, heads=history_heads)
            self.history_fuse = nn.Linear(hidden_dim * 2, hidden_dim)
            # Zero-init history residual fusion so untrained history path preserves exact prior behavior.
            nn.init.zeros_(self.history_fuse.weight)
            nn.init.zeros_(self.history_fuse.bias)
        out = nn.Linear(hidden_dim, 2)
        # Zero-init final projection so the untrained network reproduces the calibrated deck prior.
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.card_head = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout), out)

    @override
    def forward(
        self,
        deck_prior: Tensor,
        revealed_counts: Tensor,
        gameplay_signals: Tensor,
        opp_hand_size: Tensor,
        opp_library_size: Tensor,
        copy_limits: Tensor | None = None,
        history: BeliefHistoryBatch | None = None,
    ) -> tuple[Tensor, Tensor]:
        _validate_belief_inputs(
            deck_prior=deck_prior,
            revealed_counts=revealed_counts,
            gameplay_signals=gameplay_signals,
            opp_hand_size=opp_hand_size,
            opp_library_size=opp_library_size,
            signal_dim=self.signal_dim,
        )
        hand_f = opp_hand_size.to(device=deck_prior.device, dtype=torch.float32)
        lib_f = opp_library_size.to(device=deck_prior.device, dtype=torch.float32)
        unseen_total = hand_f + lib_f

        card_table = self._card_table(deck_prior.shape[1], device=deck_prior.device)
        card_repr = self._encode_cards(deck_prior, revealed_counts, card_table=card_table)
        global_ctx = self._encode_global(gameplay_signals, hand_f=hand_f, lib_f=lib_f, unseen_total=unseen_total)
        board_ctx = _summarize_board(card_repr, deck_prior=deck_prior, revealed_counts=revealed_counts)

        if history is not None and self.history_encoder is not None:
            hist_ctx = self._encode_history(
                history,
                global_ctx=global_ctx,
                board_ctx=board_ctx,
                card_table=card_table,
            )
            global_ctx = global_ctx + hist_ctx

        num_cards = deck_prior.shape[1]
        joined = torch.cat(
            (
                card_repr,
                global_ctx.unsqueeze(1).expand(-1, num_cards, -1),
                board_ctx.unsqueeze(1).expand(-1, num_cards, -1),
            ),
            dim=-1,
        )
        logits = self.card_head(joined)
        deck_delta = logits[..., 0].clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)
        hand_logits = logits[..., 1].clamp(-_LOGIT_CLAMP, _LOGIT_CLAMP)

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

    def _card_table(self, num_cards: int, *, device: torch.device) -> Tensor | None:
        if self.embedding is None:
            return None
        card_ids = torch.arange(num_cards, device=device)
        return self.embedding(card_ids).to(device=device, dtype=torch.float32)

    def _encode_cards(self, deck_prior: Tensor, revealed_counts: Tensor, *, card_table: Tensor | None) -> Tensor:
        prior_f = deck_prior.to(dtype=torch.float32).clamp_min(0.0)
        rev_f = revealed_counts.to(dtype=torch.float32).clamp_min(0.0)
        prior_share = prior_f / prior_f.sum(dim=-1, keepdim=True).clamp_min(1.0)
        rev_flag = (rev_f > 0.0).to(dtype=torch.float32)
        encoded = self.scalar_encoder(torch.stack((prior_f.log1p(), rev_f.log1p(), prior_share, rev_flag), dim=-1))
        if card_table is not None and self.card_project is not None:
            encoded = encoded + self.card_project(card_table).unsqueeze(0)
        return encoded

    def _encode_global(
        self, gameplay_signals: Tensor, *, hand_f: Tensor, lib_f: Tensor, unseen_total: Tensor
    ) -> Tensor:
        hand_ratio = hand_f / unseen_total.clamp_min(1.0)
        # Smooth log1p + ratio features generalize across any hand size (0..15+) without a 7-card cap.
        size_features = torch.stack((hand_f.log1p(), lib_f.log1p(), unseen_total.log1p(), hand_ratio), dim=-1)
        return self.global_encoder(torch.cat((gameplay_signals.to(dtype=torch.float32), size_features), dim=-1))

    def _encode_history(
        self,
        history: BeliefHistoryBatch,
        *,
        global_ctx: Tensor,
        board_ctx: Tensor,
        card_table: Tensor | None,
    ) -> Tensor:
        if (
            self.history_encoder is None
            or self.history_event_proj is None
            or self.history_queries is None
            or self.history_fuse is None
        ):
            return torch.zeros_like(global_ctx)

        device = global_ctx.device
        batch = global_ctx.shape[0]
        event_mask = history.mask.to(device=device, dtype=torch.bool)
        events = history.events.to(device=device, dtype=torch.float32)
        event_age = event_mask.flip(1).cumsum(1).flip(1) - 1
        event_tokens = self.history_event_proj(events) + age_encoding(event_age, width=self.hidden_dim)
        if card_table is not None and self.card_project is not None:
            card_ids = history.card_ids.to(device=device, dtype=torch.int64)
            has_card = (card_ids >= 0) & (card_ids < card_table.shape[0]) & event_mask
            safe_ids = card_ids.clamp(0, max(0, card_table.shape[0] - 1))
            card_proj = self.card_project(card_table[safe_ids])
            event_tokens = event_tokens + card_proj * has_card.unsqueeze(-1).to(dtype=event_tokens.dtype)

        queries = self.history_queries.unsqueeze(0).expand(batch, -1, -1)
        prefix = torch.cat((queries, global_ctx.unsqueeze(1), board_ctx.unsqueeze(1)), dim=1)
        prefix_mask = torch.ones((batch, prefix.shape[1]), dtype=torch.bool, device=device)
        seq_parts = [prefix, event_tokens]
        mask_parts = [prefix_mask, event_mask]

        if (
            self.earlier_proj is not None
            and card_table is not None
            and history.earlier_deck is not None
            and history.earlier_hand is not None
            and history.earlier_mask is not None
        ):
            e_mask = history.earlier_mask.to(device=device, dtype=torch.bool)
            e_deck = history.earlier_deck.to(device=device, dtype=torch.float32) @ card_table
            e_hand = history.earlier_hand.to(device=device, dtype=torch.float32) @ card_table
            e_age = e_mask.cumsum(1) - 1
            e_tokens = self.earlier_proj(torch.cat((e_deck, e_hand), dim=-1)) + age_encoding(
                e_age, width=self.hidden_dim
            )
            seq_parts.append(e_tokens)
            mask_parts.append(e_mask)

        sequence = torch.cat(seq_parts, dim=1)
        full_mask = torch.cat(mask_parts, dim=1)
        encoded = encode_real(self.history_encoder, sequence, mask=full_mask)
        return self.history_fuse(torch.cat((encoded[:, 0], encoded[:, 1]), dim=-1))


def belief_loss(pred_deck: Tensor, pred_hand: Tensor, *, target_deck: Tensor, target_hand: Tensor) -> Tensor:
    deck_nll = _copy_distribution_nll(pred_deck, target_deck)
    hand_nll = _copy_distribution_nll(pred_hand, target_hand)
    return deck_nll + hand_nll


def evaluate_belief_log_loss(
    pred_deck: Tensor, pred_hand: Tensor, *, target_deck: Tensor, target_hand: Tensor
) -> dict[str, float]:
    with torch.no_grad():
        deck_nll = float(_copy_distribution_nll(pred_deck, target_deck).item())
        hand_nll = float(_copy_distribution_nll(pred_hand, target_hand).item())
    return {"deck_log_loss": deck_nll, "hand_log_loss": hand_nll, "total_log_loss": deck_nll + hand_nll}


def train_belief_net(
    model: BeliefNet,
    *,
    deck_priors: Tensor,
    revealed_counts: Tensor,
    gameplay_signals: Tensor,
    opp_hand_sizes: Tensor,
    opp_library_sizes: Tensor,
    target_deck_copies: Tensor,
    target_hand_copies: Tensor,
    history: BeliefHistoryBatch | None = None,
    steps: int = 100,
    lr: float = 3e-3,
    weight_decay: float = 1e-4,
) -> dict[str, float]:
    params = (
        own_parameters(model, model.embedding) if isinstance(model.embedding, nn.Module) else tuple(model.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        pred_deck, pred_hand = model(
            deck_priors,
            revealed_counts,
            gameplay_signals,
            opp_hand_sizes,
            opp_library_sizes,
            history=history,
        )
        loss = belief_loss(pred_deck, pred_hand, target_deck=target_deck_copies, target_hand=target_hand_copies)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final_deck, final_hand = model(
            deck_priors,
            revealed_counts,
            gameplay_signals,
            opp_hand_sizes,
            opp_library_sizes,
            history=history,
        )
    return evaluate_belief_log_loss(
        final_deck, final_hand, target_deck=target_deck_copies, target_hand=target_hand_copies
    )


def infer_belief_state(
    model: BeliefNet,
    *,
    prior: FormatPrior,
    revealed_counts: Tensor,
    gameplay_signals: Tensor,
    opp_hand_size: int,
    opp_library_size: int,
    prior_backend: PriorBackend | PriorBackendName = "bayesian_pool",
    matchup_equity: float | None = None,
    history: BeliefHistoryBatch | None = None,
) -> BeliefState:
    backend, backend_name = _resolve_prior_backend(prior_backend)
    device = revealed_counts.device
    deck_prior = backend.posterior_deck(revealed_counts, prior)
    if deck_prior.ndim != 1:
        msg = f"infer_belief_state expects 1D revealed_counts, got prior shape {tuple(deck_prior.shape)}"
        raise ValueError(msg)

    hand_tensor = torch.tensor([opp_hand_size], device=device, dtype=torch.int64)
    lib_tensor = torch.tensor([opp_library_size], device=device, dtype=torch.int64)
    signals = gameplay_signals.unsqueeze(0) if gameplay_signals.ndim == 1 else gameplay_signals
    limits = prior.copy_limits.to(device=device).unsqueeze(0) if prior.copy_limits is not None else None
    hist_dev = None if history is None else history.to(device)
    with torch.no_grad():
        deck_copies, hand_probs = model(
            deck_prior.unsqueeze(0),
            revealed_counts.unsqueeze(0),
            signals,
            hand_tensor,
            lib_tensor,
            copy_limits=limits,
            history=hist_dev,
        )
    return BeliefState(
        card_names=prior.card_names,
        deck_copies=deck_copies.squeeze(0),
        hand_probs=hand_probs.squeeze(0),
        opp_hand_size=opp_hand_size,
        opp_library_size=opp_library_size,
        prior_backend=backend_name,
        matchup_equity=matchup_equity,
    )


def save_belief_net(path: Path, model: BeliefNet) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    card_table = model.embedding.table.detach().cpu() if isinstance(model.embedding, CardTable) else None
    payload: dict[str, object] = {
        "signal_dim": model.signal_dim,
        "hidden_dim": model.hidden_dim,
        "dropout": model.dropout,
        "use_history": model.use_history,
        "history_layers": model.history_layers,
        "history_heads": model.history_heads,
        "card_table": card_table,
        "state_dict": model.state_dict(),
    }
    tmp_path = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_belief_net(path: Path) -> BeliefNet:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        msg = f"unexpected BeliefNet checkpoint payload at {path}"
        raise TypeError(msg)
    signal_dim = int(payload.get("signal_dim", GAMEPLAY_SIGNAL_DIM))
    hidden_dim = int(payload.get("hidden_dim", 64))
    dropout = float(payload.get("dropout", 0.0))
    use_history = bool(payload.get("use_history", False))
    history_layers = int(payload.get("history_layers", 2))
    history_heads = int(payload.get("history_heads", 4))
    raw_table = payload.get("card_table")
    embedding = CardTable(raw_table) if isinstance(raw_table, Tensor) else None
    state_dict = payload.get("state_dict", payload)
    model = BeliefNet(
        signal_dim=signal_dim,
        hidden_dim=hidden_dim,
        embedding=embedding,
        dropout=dropout,
        use_history=use_history,
        history_layers=history_layers,
        history_heads=history_heads,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _validate_belief_inputs(
    *,
    deck_prior: Tensor,
    revealed_counts: Tensor,
    gameplay_signals: Tensor,
    opp_hand_size: Tensor,
    opp_library_size: Tensor,
    signal_dim: int,
) -> None:
    if deck_prior.ndim != _MATRIX_NDIM or revealed_counts.shape != deck_prior.shape:
        msg = "deck_prior and revealed_counts must have identical shape (B, C)"
        raise ValueError(msg)
    batch = deck_prior.shape[0]
    if gameplay_signals.shape != (batch, signal_dim):
        msg = f"gameplay_signals must have shape ({batch}, {signal_dim}), got {tuple(gameplay_signals.shape)}"
        raise ValueError(msg)
    if opp_hand_size.shape != (batch,) or opp_library_size.shape != (batch,):
        msg = f"opp_hand_size and opp_library_size must have shape ({batch},)"
        raise ValueError(msg)
    if bool((opp_hand_size < 0).any()) or bool((opp_library_size < 0).any()):
        msg = "opp_hand_size and opp_library_size must be non-negative"
        raise ValueError(msg)


def _summarize_board(card_repr: Tensor, *, deck_prior: Tensor, revealed_counts: Tensor) -> Tensor:
    prior_w = deck_prior.to(dtype=torch.float32).clamp_min(0.0)
    rev_w = revealed_counts.to(dtype=torch.float32).clamp_min(0.0)
    prior_norm = prior_w / prior_w.sum(dim=-1, keepdim=True).clamp_min(_PROB_EPS)
    rev_norm = rev_w / rev_w.sum(dim=-1, keepdim=True).clamp_min(_PROB_EPS)
    return 0.5 * ((card_repr * prior_norm.unsqueeze(-1)).sum(dim=1) + (card_repr * rev_norm.unsqueeze(-1)).sum(dim=1))


def _normalize_deck_copies(
    deck_prior: Tensor,
    *,
    deck_delta: Tensor,
    unseen_total: Tensor,
    revealed_counts: Tensor | None = None,
    copy_limits: Tensor | None = None,
) -> Tensor:
    raw = (deck_prior.to(dtype=torch.float32).clamp_min(0.0) + _PRIOR_FLOOR) * torch.exp(deck_delta)
    if copy_limits is not None and revealed_counts is not None:
        limits_f = copy_limits.to(device=raw.device, dtype=torch.float32)
        if limits_f.ndim == 1:
            limits_f = limits_f.unsqueeze(0).expand_as(raw)
        remaining = (limits_f - revealed_counts.to(device=raw.device, dtype=torch.float32)).clamp_min(0.0)
        legal = remaining > _ZERO_EPS
        masked = torch.where(legal, raw, torch.zeros_like(raw))
        raw = torch.where(masked.sum(dim=-1, keepdim=True) > _ZERO_EPS, masked, raw)
    scaled = raw * (unseen_total.unsqueeze(-1) / raw.sum(dim=-1, keepdim=True).clamp_min(1e-8))
    return torch.where(unseen_total.unsqueeze(-1) > 0.0, scaled, torch.zeros_like(scaled))


def _solve_hand_probs(deck_copies: Tensor, *, hand_logits: Tensor, hand_size: Tensor, unseen_total: Tensor) -> Tensor:
    # Solve sum_c deck_copies_c * sigmoid(hand_logits_c + tau) == hand_size so 0 <= hand_probs <= deck_copies.
    target = hand_size.unsqueeze(-1)
    total = unseen_total.unsqueeze(-1)
    with torch.no_grad():
        lo = -hand_logits.max(dim=-1, keepdim=True).values - 20.0
        hi = -hand_logits.min(dim=-1, keepdim=True).values + 20.0
        for _ in range(_BISECTION_STEPS):
            mid = 0.5 * (lo + hi)
            expected = (deck_copies * torch.sigmoid(hand_logits + mid)).sum(dim=-1, keepdim=True)
            lo = torch.where(expected < target, mid, lo)
            hi = torch.where(expected < target, hi, mid)
        tau_star = 0.5 * (lo + hi)

    # One differentiable Newton step attaches the exact implicit gradient d(tau*)/d(deck_copies, hand_logits).
    probs_star = torch.sigmoid(hand_logits + tau_star)
    residual = (deck_copies * probs_star).sum(dim=-1, keepdim=True) - target
    slope = (deck_copies * probs_star * (1.0 - probs_star)).sum(dim=-1, keepdim=True).clamp_min(_PROB_EPS)
    tau = tau_star - (residual / slope).clamp(-2.0, 2.0)
    interior = deck_copies * torch.sigmoid(hand_logits + tau)
    return torch.where(
        target <= 0.0,
        torch.zeros_like(deck_copies),
        torch.where(target >= total - _TOLERANCE, deck_copies, interior),
    )


def _copy_distribution_nll(predicted: Tensor, target: Tensor) -> Tensor:
    pred_f = predicted.to(dtype=torch.float32).clamp_min(0.0) + _PROB_EPS
    probs = pred_f / pred_f.sum(dim=-1, keepdim=True)
    target_f = target.to(dtype=torch.float32).clamp_min(0.0)
    totals = target_f.sum(dim=-1)
    active = totals > 0.0
    if not bool(active.any()):
        return pred_f.sum() * 0.0
    per_row = -(target_f * probs.log()).sum(dim=-1) / totals.clamp_min(1.0)
    return per_row[active].mean()


def _resolve_prior_backend(prior_backend: PriorBackend | PriorBackendName) -> tuple[PriorBackend, PriorBackendName]:
    if isinstance(prior_backend, str):
        if prior_backend == "bayesian_pool":
            return BayesianDeckPoolPrior(), "bayesian_pool"
        msg = "pass a SetTransformerPrior instance for prior_backend='set_transformer'"
        raise ValueError(msg)
    name: PriorBackendName = "set_transformer" if isinstance(prior_backend, SetTransformerPrior) else "bayesian_pool"
    return prior_backend, name

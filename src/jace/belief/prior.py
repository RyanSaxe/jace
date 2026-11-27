"""Format deck priors, exact Bayesian pool updates, learned set smoothing, and serialization.

`fit_format_prior` builds a `FormatPrior` from a `[M, C]` matrix of deck copy
counts. `BayesianDeckPoolPrior` computes the exact hypergeometric Bayesian
posterior over `prior.deck_pool` given revealed opponent cards, falling back
smoothly to pairwise conditional copies and day-0 booster odds when no pool
deck covers the revealed cards or no pool is attached. `SetTransformerPrior`
reuses `CardSetEncoder` over frozen `CardVectors` with zero per-card ID weights
to smooth unseen/day-0 card copy expectations.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, override

import torch
from torch import Tensor, nn
from torch.nn import functional

from jace.belief.types import FormatPrior
from jace.cards import CardVectors
from jace.decks.score import CardSetEncoder

__all__ = (
    "BayesianDeckPoolPrior",
    "PriorBackend",
    "SetTransformerPrior",
    "fit_format_prior",
    "load_format_prior",
    "save_format_prior",
)

_MATRIX_NDIM = 2
_MAX_UINT8 = 255.0
_PRIOR_FORMAT_VERSION = 1
_FALLBACK_UNCOVERED_FLOOR = 0.05
_REQUIRED_PRIOR_KEYS = frozenset(
    {"version", "format_id", "card_names", "mean_copies", "cond_copies", "booster_fallback", "deck_pool", "copy_limits"}
)


class PriorBackend(Protocol):
    def posterior_deck(self, revealed_counts: Tensor, prior: FormatPrior) -> Tensor: ...


def fit_format_prior(  # noqa: PLR0917 -- public contract takes (format_id, card_names, decks)
    format_id: str,
    card_names: Sequence[str],
    decks: Tensor,
    *,
    booster_fallback: Tensor | None = None,
    copy_limits: Tensor | None = None,
) -> FormatPrior:
    names = tuple(card_names)
    num_cards = len(names)
    _validate_deck_matrix(decks, num_cards)

    decks_f = decks.detach().to(dtype=torch.float32)
    mean_copies = decks_f.mean(dim=0)
    fallback = _resolve_booster_fallback(booster_fallback, mean_copies=mean_copies, num_cards=num_cards)
    cond_copies = _fit_conditional_copies(decks_f, mean_copies=mean_copies)
    pool = _pack_deck_pool(decks_f)
    limits = _resolve_copy_limits(copy_limits, num_cards=num_cards)

    return FormatPrior(
        format_id=format_id,
        card_names=names,
        mean_copies=mean_copies,
        cond_copies=cond_copies,
        booster_fallback=fallback,
        deck_pool=pool,
        copy_limits=limits,
    )


class BayesianDeckPoolPrior:
    def __init__(self, *, temperature: float = 1.0) -> None:
        if temperature <= 0.0:
            msg = f"temperature must be positive, got {temperature}"
            raise ValueError(msg)
        self.temperature = temperature

    def posterior_deck(self, revealed_counts: Tensor, prior: FormatPrior) -> Tensor:
        revealed_2d, was_1d = _as_batched_revealed(revealed_counts, len(prior.card_names))
        device = revealed_2d.device
        fallback = _conditional_fallback(revealed_2d, prior, device=device)

        if prior.deck_pool is None or prior.deck_pool.shape[0] == 0:
            result = _apply_copy_limits(fallback, revealed=revealed_2d, copy_limits=prior.copy_limits)
            return result.squeeze(0) if was_1d else result

        pool = prior.deck_pool.to(device=device, dtype=torch.float32)
        unseen_pool, has_cover = _hypergeometric_pool_posterior(
            revealed_2d,
            pool=pool,
            temperature=self.temperature,
        )
        booster = prior.booster_fallback.to(device=device, dtype=torch.float32).unsqueeze(0)
        uncovered_fallback = (1.0 - _FALLBACK_UNCOVERED_FLOOR) * fallback + _FALLBACK_UNCOVERED_FLOOR * booster
        combined = torch.where(has_cover.unsqueeze(-1), unseen_pool, uncovered_fallback)
        result = _apply_copy_limits(combined, revealed=revealed_2d, copy_limits=prior.copy_limits)
        return result.squeeze(0) if was_1d else result


class SetTransformerPrior(nn.Module):
    def __init__(
        self,
        embedding: CardVectors,
        *,
        width: int = 64,
        layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedding: CardVectors = embedding
        self.set_encoder: CardSetEncoder = CardSetEncoder(embedding, amounts=3, width=width, layers=layers, heads=heads)
        self.card_project: nn.Linear = nn.Linear(embedding.width, width)
        self.amount_project: nn.Linear = nn.Linear(3, width)
        out = nn.Linear(width, 1)
        # Start near the analytical prior so untrained outputs remain well-scaled for unseen cards.
        nn.init.zeros_(out.weight)
        nn.init.zeros_(out.bias)
        self.head: nn.Sequential = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), out)

    @override
    def forward(
        self,
        revealed_counts: Tensor,
        mean_copies: Tensor,
        booster_fallback: Tensor,
    ) -> Tensor:
        num_cards = self.embedding.cards
        revealed_2d, was_1d = _as_batched_revealed(revealed_counts, num_cards)
        batch = revealed_2d.shape[0]
        device = revealed_2d.device

        mean_2d = _broadcast_card_vector(mean_copies, batch=batch, num_cards=num_cards, device=device)
        booster_2d = _broadcast_card_vector(booster_fallback, batch=batch, num_cards=num_cards, device=device)
        amounts = torch.stack((revealed_2d, mean_2d, booster_2d), dim=-1)

        context = _encode_non_empty_rows(self.set_encoder, amounts, width=self.card_project.out_features)
        card_ids = torch.arange(num_cards, device=device)
        card_tokens = self.card_project(self.embedding(card_ids)).unsqueeze(0) + self.amount_project(amounts.log1p())
        joined = torch.cat((card_tokens, context.unsqueeze(1).expand(-1, num_cards, -1)), dim=-1)
        delta = self.head(joined).squeeze(-1)

        base = (mean_2d - revealed_2d).clamp_min(0.0) + _FALLBACK_UNCOVERED_FLOOR * booster_2d.clamp_min(0.0)
        predicted = functional.softplus(delta + base.clamp_min(1e-4).log())
        return predicted.squeeze(0) if was_1d else predicted

    def posterior_deck(self, revealed_counts: Tensor, prior: FormatPrior) -> Tensor:
        device = revealed_counts.device
        mean_copies = prior.mean_copies.to(device=device, dtype=torch.float32)
        booster_fallback = prior.booster_fallback.to(device=device, dtype=torch.float32)
        predicted = self.forward(revealed_counts, mean_copies, booster_fallback)
        revealed_2d, was_1d = _as_batched_revealed(revealed_counts, len(prior.card_names))
        pred_2d = predicted.unsqueeze(0) if was_1d else predicted
        clamped = _apply_copy_limits(pred_2d, revealed=revealed_2d, copy_limits=prior.copy_limits)
        return clamped.squeeze(0) if was_1d else clamped


def save_format_prior(path: Path, prior: FormatPrior) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    payload: dict[str, object] = {
        "version": _PRIOR_FORMAT_VERSION,
        "format_id": prior.format_id,
        "card_names": prior.card_names,
        "mean_copies": prior.mean_copies.detach().cpu(),
        "cond_copies": prior.cond_copies.detach().cpu(),
        "booster_fallback": prior.booster_fallback.detach().cpu(),
        "deck_pool": None if prior.deck_pool is None else prior.deck_pool.detach().cpu(),
        "copy_limits": None if prior.copy_limits is None else prior.copy_limits.detach().cpu(),
    }
    torch.save(payload, partial)
    partial.replace(path)


def load_format_prior(path: Path) -> FormatPrior:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not _REQUIRED_PRIOR_KEYS.issubset(payload.keys()):
        msg = f"invalid FormatPrior file at {path}"
        raise ValueError(msg)
    if payload["version"] != _PRIOR_FORMAT_VERSION:
        msg = f"unsupported FormatPrior version {payload['version']!r}"
        raise ValueError(msg)
    return FormatPrior(
        format_id=str(payload["format_id"]),
        card_names=tuple(payload["card_names"]),
        mean_copies=payload["mean_copies"].to(dtype=torch.float32),
        cond_copies=payload["cond_copies"].to(dtype=torch.float32),
        booster_fallback=payload["booster_fallback"].to(dtype=torch.float32),
        deck_pool=payload["deck_pool"],
        copy_limits=None if payload["copy_limits"] is None else payload["copy_limits"].to(dtype=torch.float32),
    )


def _validate_deck_matrix(decks: Tensor, num_cards: int) -> None:
    if num_cards == 0:
        msg = "card_names must be non-empty"
        raise ValueError(msg)
    if decks.ndim != _MATRIX_NDIM or decks.shape[0] == 0 or decks.shape[1] != num_cards:
        msg = f"decks must have shape (M, {num_cards}) with M > 0, got {tuple(decks.shape)}"
        raise ValueError(msg)
    if bool((decks < 0).any()):
        msg = "decks copy counts must be non-negative"
        raise ValueError(msg)


def _resolve_booster_fallback(booster_fallback: Tensor | None, *, mean_copies: Tensor, num_cards: int) -> Tensor:
    if booster_fallback is None:
        avg_size = float(mean_copies.sum().clamp_min(1.0).item())
        return torch.full((num_cards,), avg_size / num_cards, dtype=torch.float32)
    if booster_fallback.shape != (num_cards,) or bool((booster_fallback < 0).any()):
        msg = f"booster_fallback must be non-negative with shape ({num_cards},)"
        raise ValueError(msg)
    return booster_fallback.detach().to(dtype=torch.float32).clone()


def _resolve_copy_limits(copy_limits: Tensor | None, *, num_cards: int) -> Tensor | None:
    if copy_limits is None:
        return None
    if copy_limits.shape != (num_cards,) or bool((copy_limits < 0).any()):
        msg = f"copy_limits must be non-negative with shape ({num_cards},)"
        raise ValueError(msg)
    return copy_limits.detach().to(dtype=torch.float32).clone()


def _fit_conditional_copies(decks_f: Tensor, *, mean_copies: Tensor) -> Tensor:
    present = (decks_f > 0).to(dtype=torch.float32)
    counts = present.sum(dim=0)
    co_copies = present.T @ decks_f
    safe_counts = counts.clamp_min(1.0).unsqueeze(1)
    cond = co_copies / safe_counts
    unseen_rows = (counts == 0).unsqueeze(1)
    return torch.where(unseen_rows, mean_copies.unsqueeze(0).expand_as(cond), cond)


def _pack_deck_pool(decks_f: Tensor) -> Tensor:
    is_integral = bool(torch.equal(decks_f, decks_f.floor()))
    if is_integral and float(decks_f.max().item()) <= _MAX_UINT8:
        return decks_f.to(dtype=torch.uint8)
    return decks_f.clone()


def _as_batched_revealed(revealed_counts: Tensor, num_cards: int) -> tuple[Tensor, bool]:
    if revealed_counts.ndim not in {1, _MATRIX_NDIM} or revealed_counts.shape[-1] != num_cards:
        msg = f"revealed_counts must have shape ({num_cards},) or (B, {num_cards}), got {tuple(revealed_counts.shape)}"
        raise ValueError(msg)
    if bool((revealed_counts < 0).any()):
        msg = "revealed_counts must be non-negative"
        raise ValueError(msg)
    was_1d = revealed_counts.ndim == 1
    batched = revealed_counts.unsqueeze(0) if was_1d else revealed_counts
    return batched.to(dtype=torch.float32), was_1d


def _log_binom(n: Tensor, k: Tensor) -> Tensor:
    diff = (n - k).clamp_min(0.0)
    return torch.lgamma(n + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(diff + 1.0)


def _hypergeometric_pool_posterior(
    revealed: Tensor,
    *,
    pool: Tensor,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    covers = (pool.unsqueeze(0) >= revealed.unsqueeze(1) - 1e-6).all(dim=-1)
    has_cover = covers.any(dim=-1)

    pool_totals = pool.sum(dim=-1).unsqueeze(0)
    rev_totals = revealed.sum(dim=-1, keepdim=True)
    card_log_ways = _log_binom(pool.unsqueeze(0), revealed.unsqueeze(1)).sum(dim=-1)
    total_log_ways = _log_binom(pool_totals, rev_totals)
    log_lik = (card_log_ways - total_log_ways) / temperature

    masked_log_lik = torch.where(covers, log_lik, torch.full_like(log_lik, -torch.inf))
    safe_log_lik = torch.where(has_cover.unsqueeze(-1), masked_log_lik, torch.zeros_like(log_lik))
    weights = torch.softmax(safe_log_lik, dim=-1)
    expected_full = weights @ pool
    unseen = (expected_full - revealed).clamp_min(0.0)
    return unseen, has_cover


def _conditional_fallback(revealed: Tensor, prior: FormatPrior, *, device: torch.device) -> Tensor:
    mean_copies = prior.mean_copies.to(device=device, dtype=torch.float32)
    cond_copies = prior.cond_copies.to(device=device, dtype=torch.float32)
    booster = prior.booster_fallback.to(device=device, dtype=torch.float32)

    rev_total = revealed.sum(dim=-1, keepdim=True)
    no_revealed = rev_total <= 0.0
    cond_est = (revealed @ cond_copies) / rev_total.clamp_min(1.0)

    unseen_card_mask = (mean_copies <= 0.0).to(dtype=torch.float32)
    unseen_share = (revealed * unseen_card_mask).sum(dim=-1, keepdim=True) / rev_total.clamp_min(1.0)
    blended = (1.0 - unseen_share) * cond_est + unseen_share * booster.unsqueeze(0)

    base_turn0 = mean_copies if float(mean_copies.sum().item()) > 0.0 else booster
    estimated_full = torch.where(no_revealed, base_turn0.unsqueeze(0).expand_as(revealed), blended)
    return (estimated_full - revealed).clamp_min(0.0)


def _apply_copy_limits(unseen: Tensor, *, revealed: Tensor, copy_limits: Tensor | None) -> Tensor:
    if copy_limits is None:
        return unseen.clamp_min(0.0)
    limits = copy_limits.to(device=unseen.device, dtype=torch.float32).unsqueeze(0)
    max_unseen = (limits - revealed).clamp_min(0.0)
    return unseen.clamp_min(0.0).minimum(max_unseen)


def _broadcast_card_vector(vector: Tensor, *, batch: int, num_cards: int, device: torch.device) -> Tensor:
    if vector.ndim == 1 and vector.shape == (num_cards,):
        return vector.to(device=device, dtype=torch.float32).unsqueeze(0).expand(batch, -1)
    if vector.ndim == _MATRIX_NDIM and vector.shape == (batch, num_cards):
        return vector.to(device=device, dtype=torch.float32)
    msg = f"expected tensor of shape ({num_cards},) or ({batch}, {num_cards}), got {tuple(vector.shape)}"
    raise ValueError(msg)


def _encode_non_empty_rows(encoder: CardSetEncoder, amounts: Tensor, *, width: int) -> Tensor:
    active_rows = amounts.gt(0).any(dim=-1).any(dim=-1)
    if bool(active_rows.all()):
        return encoder(amounts)
    context = amounts.new_zeros((amounts.shape[0], width))
    if bool(active_rows.any()):
        context[active_rows] = encoder(amounts[active_rows])
    return context

"""End-to-end real-game training and evaluation CLI for `FormatPrior`, `BeliefNet`, `V_0`, and Search.

Fits `FormatPrior` strictly on training decks, collects real closed-decklist
gameplay transitions and outcomes on separate training and held-out deck splits,
trains `BeliefNet` and `MatchupPredictor` (`V_0`) on GPU/CPU with held-out
calibration metrics, and optionally runs mirrored Arena H2H search sweeps
comparing raw policy, prior-only search, and trained `BeliefNet` search.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from jace.belief import (
    GAMEPLAY_SIGNAL_DIM,
    BeliefNet,
    FormatPrior,
    MatchupPredictor,
    belief_loss,
    evaluate_belief_log_loss,
    fit_format_prior,
    save_belief_net,
    save_format_prior,
)
from jace.cards import CardTable, own_parameters
from jace.collect.decks import load_human_deck_catalog
from jace.eval.arena import SearchConfigSummary
from jace.forks.arena import run_search_arena_match
from jace.modules import default_device_name
from jace.train._belief_data import BeliefDataset, collect_belief_dataset

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = (
    "BeliefExperimentConfig",
    "build_cooccurrence_card_table",
    "evaluate_belief_metrics",
    "evaluate_v0_metrics",
    "main",
    "run_belief_experiment",
)

_CARD_VEC_DIM = 32
_EPS = 1e-6
_MIN_AUC_SAMPLES = 2


@dataclass(frozen=True, slots=True)
class BeliefExperimentConfig:
    decks_path: Path
    out_dir: Path
    format_id: str
    checkpoint: Path | None = None
    train_games: int = 300
    val_games: int = 120
    collect_workers: int = 4
    sample_stride: int = 2
    max_actions: int = 600
    hidden_dim: int = 128
    epochs: int = 25
    batch_size: int = 256
    lr: float = 2e-3
    v0_epochs: int = 30
    v0_lr: float = 2e-3
    device: str = "auto"
    seed: int = 9901
    arena_deals: int = 0
    arena_worlds: tuple[int, ...] = (8, 16)
    arena_horizon: int = 24
    arena_workers: int = 1


def build_cooccurrence_card_table(train_decks: Tensor, *, width: int = _CARD_VEC_DIM) -> CardTable:
    decks_f = train_decks.to(dtype=torch.float32)
    centered = decks_f - decks_f.mean(dim=0, keepdim=True)
    # Use low-rank SVD on transposed deck matrix (C x N) to avoid O(N^2) memory.
    num_cards = decks_f.shape[1]
    q = min(width, num_cards, max(1, decks_f.shape[0]))
    if q < 1:
        return CardTable(torch.zeros((num_cards, width), dtype=torch.float32))
    u, s, _ = torch.svd_lowrank(centered.T, q=q, niter=2)
    raw = u * s.unsqueeze(0)
    if raw.shape[1] < width:
        pad = torch.zeros((num_cards, width - raw.shape[1]), dtype=torch.float32)
        raw = torch.cat((raw, pad), dim=1)
    norms = raw.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    return CardTable(raw / norms)


def _scale_prior_baseline(deck_prior: Tensor, *, hand_sizes: Tensor, lib_sizes: Tensor) -> tuple[Tensor, Tensor]:
    hand_f = hand_sizes.to(dtype=torch.float32).unsqueeze(-1)
    unseen = (hand_sizes + lib_sizes).to(dtype=torch.float32).unsqueeze(-1)
    w = deck_prior.to(dtype=torch.float32).clamp_min(0.0)
    w = torch.where(w.sum(dim=-1, keepdim=True) > _EPS, w, torch.ones_like(w))
    deck = torch.where(unseen > 0.0, w * (unseen / w.sum(dim=-1, keepdim=True)), torch.zeros_like(w))
    hand = torch.where(unseen > 0.0, deck * (hand_f / unseen.clamp_min(1.0)), torch.zeros_like(w))
    return deck, hand


def _hand_recall_at_k(pred_hand: Tensor, target_hand: Tensor, *, k: int) -> float:
    k_eff = min(k, pred_hand.shape[1])
    top_idx = pred_hand.topk(k_eff, dim=-1).indices
    hits = target_hand.gather(dim=-1, index=top_idx).sum(dim=-1)
    totals = target_hand.sum(dim=-1).clamp_min(1.0)
    active = target_hand.sum(dim=-1) > 0.0
    if not bool(active.any()):
        return 0.0
    return float((hits[active] / totals[active]).mean().item())


def _hand_brier(pred_hand: Tensor, target_hand: Tensor) -> float:
    pred_p = pred_hand.clamp(0.0, 1.0)
    targ_p = (target_hand > 0.0).to(dtype=torch.float32)
    return float(((pred_p - targ_p) ** 2).mean().item())


def evaluate_belief_metrics(
    pred_deck: Tensor,
    pred_hand: Tensor,
    *,
    dataset: BeliefDataset,
) -> dict[str, object]:
    losses = evaluate_belief_log_loss(
        pred_deck,
        pred_hand,
        target_deck=dataset.target_deck_copies,
        target_hand=dataset.target_hand_copies,
    )
    by_turn: dict[str, dict[str, float]] = {}
    for label, lo, hi in (("T0_T3", 0, 3), ("T4_T6", 4, 6), ("T7_plus", 7, 999)):
        mask = (dataset.turns >= lo) & (dataset.turns <= hi)
        if bool(mask.any()):
            sub = evaluate_belief_log_loss(
                pred_deck[mask],
                pred_hand[mask],
                target_deck=dataset.target_deck_copies[mask],
                target_hand=dataset.target_hand_copies[mask],
            )
            by_turn[label] = {
                "rows": float(int(mask.sum().item())),
                "deck_log_loss": round(sub["deck_log_loss"], 4),
                "hand_log_loss": round(sub["hand_log_loss"], 4),
                "total_log_loss": round(sub["total_log_loss"], 4),
            }
    return {
        "rows": int(dataset.turns.numel()),
        "deck_log_loss": round(losses["deck_log_loss"], 4),
        "hand_log_loss": round(losses["hand_log_loss"], 4),
        "total_log_loss": round(losses["total_log_loss"], 4),
        "hand_recall_at_5": round(_hand_recall_at_k(pred_hand, dataset.target_hand_copies, k=5), 4),
        "hand_recall_at_10": round(_hand_recall_at_k(pred_hand, dataset.target_hand_copies, k=10), 4),
        "hand_brier": round(_hand_brier(pred_hand, dataset.target_hand_copies), 5),
        "by_turn": by_turn,
    }


def _predict_belief_batched(
    model: BeliefNet,
    dataset: BeliefDataset,
    *,
    copy_limits: Tensor | None,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[Tensor, Tensor]:
    model.eval()
    decks_out: list[Tensor] = []
    hands_out: list[Tensor] = []
    limits_dev = copy_limits.to(device=device) if copy_limits is not None else None
    with torch.no_grad():
        for start in range(0, dataset.revealed_counts.shape[0], batch_size):
            end = start + batch_size
            pd, ph = model(
                dataset.deck_priors[start:end].to(device),
                dataset.revealed_counts[start:end].to(device),
                dataset.gameplay_signals[start:end].to(device),
                dataset.opp_hand_sizes[start:end].to(device),
                dataset.opp_library_sizes[start:end].to(device),
                copy_limits=limits_dev,
            )
            decks_out.append(pd.cpu())
            hands_out.append(ph.cpu())
    return torch.cat(decks_out, dim=0), torch.cat(hands_out, dim=0)


def _log_phase(t0: float, msg: str) -> None:
    sys.stderr.write(f"[belief +{time.perf_counter() - t0:6.1f}s] {msg}\n")
    sys.stderr.flush()


def _v2_meta_guess_baseline(
    revealed: Tensor,
    *,
    prior: FormatPrior,
    hand_sizes: Tensor,
    lib_sizes: Tensor,
) -> tuple[Tensor, Tensor]:
    # Reproduces v1/v2-guess's sqrt(k) log-lift pairwise approximation for apples-to-apples comparison.
    copies = prior.mean_copies.to(dtype=torch.float32).clamp_min(1e-4)
    cond = prior.cond_copies.to(dtype=torch.float32).clamp_min(1e-4)
    lifts = (cond / copies.unsqueeze(0)).log()
    known = (revealed > 0.0).to(dtype=torch.float32)
    lift = (known @ lifts) / known.sum(dim=-1, keepdim=True).clamp_min(1.0).sqrt()
    expected = copies.unsqueeze(0) * lift.exp()
    expected = torch.where(known > 0.0, expected.maximum(cond.diagonal().unsqueeze(0)), expected)
    unseen_raw = (expected - revealed).clamp_min(1e-4)
    return _scale_prior_baseline(unseen_raw, hand_sizes=hand_sizes, lib_sizes=lib_sizes)


def _train_belief_model(
    *,
    train_ds: BeliefDataset,
    val_ds: BeliefDataset,
    card_table: CardTable,
    copy_limits: Tensor | None,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int,
    t0: float,
) -> tuple[BeliefNet, dict[str, object]]:
    torch.manual_seed(seed)
    model = BeliefNet(signal_dim=GAMEPLAY_SIGNAL_DIM, hidden_dim=hidden_dim, embedding=card_table).to(device)
    optimizer = torch.optim.AdamW(own_parameters(model, card_table), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    limits_dev = copy_limits.to(device=device) if copy_limits is not None else None
    n_rows = train_ds.revealed_counts.shape[0]
    gen = torch.Generator().manual_seed(seed)

    best_val_loss = math.inf
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_rows, generator=gen)
        for start in range(0, n_rows, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            pd, ph = model(
                train_ds.deck_priors[idx].to(device),
                train_ds.revealed_counts[idx].to(device),
                train_ds.gameplay_signals[idx].to(device),
                train_ds.opp_hand_sizes[idx].to(device),
                train_ds.opp_library_sizes[idx].to(device),
                copy_limits=limits_dev,
            )
            loss = belief_loss(
                pd,
                ph,
                target_deck=train_ds.target_deck_copies[idx].to(device),
                target_hand=train_ds.target_hand_copies[idx].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        val_d, val_h = _predict_belief_batched(model, val_ds, copy_limits=copy_limits, device=device)
        val_losses = evaluate_belief_log_loss(
            val_d, val_h, target_deck=val_ds.target_deck_copies, target_hand=val_ds.target_hand_copies
        )
        if val_losses["total_log_loss"] < best_val_loss:
            best_val_loss = val_losses["total_log_loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            _log_phase(
                t0,
                f"BeliefNet epoch {epoch}/{epochs}: val deck={val_losses['deck_log_loss']:.4f} "
                f"hand={val_losses['hand_log_loss']:.4f} total={val_losses['total_log_loss']:.4f}",
            )

    model.load_state_dict(best_state)
    model.cpu().eval()
    return model, {"best_epoch": best_epoch, "best_val_total_log_loss": round(best_val_loss, 4)}


def _roc_auc(probs: Tensor, labels: Tensor) -> float:
    mask = (labels == 0.0) | (labels == 1.0)
    if int(mask.sum().item()) < _MIN_AUC_SAMPLES:
        return 0.5
    p = probs[mask]
    y = labels[mask]
    pos = p[y == 1.0]
    neg = p[y == 0.0]
    if pos.numel() == 0 or neg.numel() == 0:
        return 0.5
    diff = pos.unsqueeze(1) - neg.unsqueeze(0)
    return float(((diff > 0).float() + 0.5 * (diff == 0).float()).mean().item())


def evaluate_v0_metrics(probs: Tensor, outcomes: Tensor) -> dict[str, float]:
    if outcomes.numel() == 0:
        return {"rows": 0.0, "bce": math.log(2.0), "brier": 0.25, "roc_auc": 0.5}
    p = probs.clamp(_EPS, 1.0 - _EPS)
    bce = float((-(outcomes * p.log() + (1.0 - outcomes) * (1.0 - p).log())).mean().item())
    brier = float(((p - outcomes) ** 2).mean().item())
    return {
        "rows": float(outcomes.numel()),
        "bce": round(bce, 4),
        "brier": round(brier, 4),
        "roc_auc": round(_roc_auc(p, outcomes), 4),
    }


def _train_v0_model(
    *,
    train_ds: BeliefDataset,
    val_ds: BeliefDataset,
    prior: FormatPrior,
    card_table: CardTable,
    epochs: int,
    lr: float,
    device: torch.device,
    seed: int,
) -> tuple[MatchupPredictor, dict[str, object]]:
    torch.manual_seed(seed)
    predictor = MatchupPredictor(card_table, width=32, layers=2, heads=4).to(device)
    if train_ds.matchup_outcomes.numel() == 0:
        predictor.cpu().eval()
        return predictor, {"matchup_val": evaluate_v0_metrics(torch.empty(0), torch.empty(0))}

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    field_dev = prior.mean_copies.to(device=device, dtype=torch.float32)
    my_tr = train_ds.matchup_my_decks.to(device)
    opp_tr = train_ds.matchup_opp_decks.to(device)
    y_tr = train_ds.matchup_outcomes.to(device)

    for _ in range(epochs):
        predictor.train()
        optimizer.zero_grad(set_to_none=True)
        # Jointly train matchup V_0(D_me, D_opp) and blind field V_0(D_me, P_0).
        loss_match = loss_fn(predictor(my_tr, opp_tr), y_tr)
        loss_field = loss_fn(predictor(my_tr, field_dev), y_tr)
        (loss_match + loss_field).backward()
        optimizer.step()

    predictor.eval()
    with torch.no_grad():
        val_my = val_ds.matchup_my_decks.to(device)
        val_opp = val_ds.matchup_opp_decks.to(device)
        val_match_p = predictor.predict_equity(val_my, val_opp).cpu() if val_my.shape[0] > 0 else torch.empty(0)
        val_field_p = predictor.predict_equity(val_my, field_dev).cpu() if val_my.shape[0] > 0 else torch.empty(0)
    predictor.cpu().eval()
    return predictor, {
        "constant_05_bce": round(math.log(2.0), 4),
        "matchup_val": evaluate_v0_metrics(val_match_p, val_ds.matchup_outcomes),
        "field_v0_val": evaluate_v0_metrics(val_field_p, val_ds.matchup_outcomes),
    }


def _run_arena_h2h(
    cfg: BeliefExperimentConfig,
    *,
    belief_dir: Path,
    device: torch.device,
) -> list[dict[str, object]]:
    if cfg.arena_deals <= 0 or cfg.checkpoint is None:
        return []
    records: list[dict[str, object]] = []
    ckpt_label = cfg.checkpoint.stem
    search_device = "cuda" if device.type == "cuda" else "cpu"
    records.append(
        asdict(
            run_search_arena_match(
                cfg.out_dir,
                format_id=cfg.format_id,
                model_a=ckpt_label,
                model_b="heuristic",
                deals=cfg.arena_deals,
                seed=cfg.seed,
                num_workers=cfg.arena_workers,
                decks_path=cfg.decks_path,
                model_a_spec=cfg.checkpoint,
                device=search_device,
            )
        )
    )
    for worlds in cfg.arena_worlds:
        prior_cfg = SearchConfigSummary(worlds=worlds, horizon=cfg.arena_horizon, belief_id="prior-only")
        net_cfg = SearchConfigSummary(worlds=worlds, horizon=cfg.arena_horizon, belief_id=str(belief_dir))
        for search_a, search_b in ((prior_cfg, None), (net_cfg, None), (net_cfg, prior_cfg)):
            records.append(
                asdict(
                    run_search_arena_match(
                        cfg.out_dir,
                        format_id=cfg.format_id,
                        model_a=ckpt_label,
                        model_b=ckpt_label,
                        deals=cfg.arena_deals,
                        seed=cfg.seed,
                        num_workers=cfg.arena_workers,
                        decks_path=cfg.decks_path,
                        search_config_a=search_a,
                        search_config_b=search_b,
                        model_a_spec=cfg.checkpoint,
                        model_b_spec=cfg.checkpoint,
                        device=search_device,
                    )
                )
            )
    return records


def run_belief_experiment(cfg: BeliefExperimentConfig) -> dict[str, object]:
    t0 = time.perf_counter()
    device = torch.device(default_device_name() if cfg.device == "auto" else cfg.device)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    belief_dir = cfg.out_dir / "belief" / cfg.format_id
    belief_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_human_deck_catalog(cfg.decks_path)
    _log_phase(
        t0,
        f"[1/5] Fitting FormatPrior on {catalog.train_decks.shape[0]} train decks ({len(catalog.card_names)} cards)",
    )
    prior = fit_format_prior(cfg.format_id, catalog.card_names, catalog.train_decks)
    save_format_prior(belief_dir / "prior.pt", prior)
    card_table = build_cooccurrence_card_table(catalog.train_decks)

    _log_phase(t0, f"[2/5] Collecting {cfg.train_games} train + {cfg.val_games} held-out closed-decklist games")
    train_ds = collect_belief_dataset(
        catalog=catalog,
        prior=prior,
        split="train",
        num_games=cfg.train_games,
        seed=cfg.seed,
        checkpoint=cfg.checkpoint,
        num_workers=cfg.collect_workers,
        max_actions=cfg.max_actions,
        sample_stride=cfg.sample_stride,
    )
    val_ds = collect_belief_dataset(
        catalog=catalog,
        prior=prior,
        split="held_out",
        num_games=cfg.val_games,
        seed=cfg.seed + 100_000,
        checkpoint=cfg.checkpoint,
        num_workers=cfg.collect_workers,
        max_actions=cfg.max_actions,
        sample_stride=cfg.sample_stride,
    )
    _log_phase(t0, f"Collected {train_ds.turns.numel()} train rows, {val_ds.turns.numel()} val rows")

    # Evaluate baselines vs trained BeliefNet on strictly unseen held-out decks.
    mean_prior_mat = prior.mean_copies.unsqueeze(0).expand(val_ds.revealed_counts.shape[0], -1)
    uncond_d, uncond_h = _scale_prior_baseline(
        mean_prior_mat, hand_sizes=val_ds.opp_hand_sizes, lib_sizes=val_ds.opp_library_sizes
    )
    v2_meta_d, v2_meta_h = _v2_meta_guess_baseline(
        val_ds.revealed_counts, prior=prior, hand_sizes=val_ds.opp_hand_sizes, lib_sizes=val_ds.opp_library_sizes
    )
    bayes_d, bayes_h = _scale_prior_baseline(
        val_ds.deck_priors, hand_sizes=val_ds.opp_hand_sizes, lib_sizes=val_ds.opp_library_sizes
    )
    _log_phase(t0, f"[3/5] Training BeliefNet (hidden={cfg.hidden_dim}, epochs={cfg.epochs}) on {device}")
    belief_net, train_info = _train_belief_model(
        train_ds=train_ds,
        val_ds=val_ds,
        card_table=card_table,
        copy_limits=prior.copy_limits,
        hidden_dim=cfg.hidden_dim,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        device=device,
        seed=cfg.seed,
        t0=t0,
    )
    save_belief_net(belief_dir / "belief_net.pt", belief_net)
    net_d, net_h = _predict_belief_batched(
        belief_net, val_ds, copy_limits=prior.copy_limits, device=torch.device("cpu")
    )

    _log_phase(t0, f"[4/5] Training Turn-0 MatchupPredictor V_0 ({cfg.v0_epochs} epochs) on {device}")
    v0_model, v0_metrics = _train_v0_model(
        train_ds=train_ds,
        val_ds=val_ds,
        prior=prior,
        card_table=card_table,
        epochs=cfg.v0_epochs,
        lr=cfg.v0_lr,
        device=device,
        seed=cfg.seed,
    )
    torch.save({"card_table": card_table.state_dict(), "state_dict": v0_model.state_dict()}, belief_dir / "v0.pt")

    _log_phase(t0, f"[5/5] Arena H2H ({cfg.arena_deals} deals)")
    arena_records = _run_arena_h2h(cfg, belief_dir=belief_dir, device=device)
    summary: dict[str, object] = {
        "format_id": cfg.format_id,
        "device": str(device),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "train_rows": int(train_ds.turns.numel()),
        "val_rows": int(val_ds.turns.numel()),
        "train_matchup_rows": int(train_ds.matchup_outcomes.numel()),
        "val_matchup_rows": int(val_ds.matchup_outcomes.numel()),
        "belief_train": train_info,
        "belief_val": {
            "unconditioned_mean_prior": evaluate_belief_metrics(uncond_d, uncond_h, dataset=val_ds),
            "v2_guess_log_lift_prior": evaluate_belief_metrics(v2_meta_d, v2_meta_h, dataset=val_ds),
            "bayesian_pool_prior": evaluate_belief_metrics(bayes_d, bayes_h, dataset=val_ds),
            "trained_belief_net": evaluate_belief_metrics(net_d, net_h, dataset=val_ds),
        },
        "v0_val": v0_metrics,
        "arena_matches": arena_records,
    }
    (cfg.out_dir / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> BeliefExperimentConfig:
    parser = argparse.ArgumentParser(
        prog="jace.train.belief",
        description="Train and evaluate BeliefNet, V0, and Search",
    )
    parser.add_argument("--decks-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--format", dest="format_id", required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--train-games", type=int, default=300)
    parser.add_argument("--val-games", type=int, default=120)
    parser.add_argument("--collect-workers", type=int, default=4)
    parser.add_argument("--sample-stride", type=int, default=2)
    parser.add_argument("--max-actions", type=int, default=600)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--v0-epochs", type=int, default=30)
    parser.add_argument("--v0-lr", type=float, default=2e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=9901)
    parser.add_argument("--arena-deals", type=int, default=0)
    parser.add_argument("--arena-worlds", default="8,16")
    parser.add_argument("--arena-horizon", type=int, default=24)
    parser.add_argument("--arena-workers", type=int, default=1)
    args = parser.parse_args(list(argv) if argv is not None else None)
    worlds = tuple(int(p.strip()) for p in str(args.arena_worlds).split(",") if p.strip())
    return BeliefExperimentConfig(
        decks_path=args.decks_path,
        out_dir=args.out_dir,
        format_id=args.format_id,
        checkpoint=args.checkpoint,
        train_games=args.train_games,
        val_games=args.val_games,
        collect_workers=args.collect_workers,
        sample_stride=args.sample_stride,
        max_actions=args.max_actions,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        v0_epochs=args.v0_epochs,
        v0_lr=args.v0_lr,
        device=args.device,
        seed=args.seed,
        arena_deals=args.arena_deals,
        arena_worlds=worlds,
        arena_horizon=args.arena_horizon,
        arena_workers=args.arena_workers,
    )


def main(argv: Sequence[str] | None = None) -> int:
    summary = run_belief_experiment(_parse_args(argv))
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Live RL-stream training for history-aware `BeliefNet`, `V_0`, and policy belief-memory.

Consumes newly recorded closed-decklist self-play games incrementally from
`games/train.jsonl` during `jace.train.full_data` training, replaying each game
in Rust (`managym.Env`) with zero extra policy inference to recover exact rolling
public history, ground-truth hidden opponent hand and unseen deck targets, and
game outcomes. Keeps `FormatPrior` strictly frozen on human `train_decks` (zero
builder-deck drift) while training `BeliefNet`, `MatchupPredictor` (`V_0`), and
an optional differentiable `PolicyBeliefAuxHead` on `ActorCritic.decision_memory`.
"""

from __future__ import annotations

import json
import math
import random
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import managym
import torch
from manabot.backends.torch.arrays import on_device
from managym.enums import ZoneEnum
from torch import Tensor, nn

from jace.belief import (
    GAMEPLAY_SIGNAL_DIM,
    BayesianDeckPoolPrior,
    BeliefHistoryBuffer,
    BeliefNet,
    FormatPrior,
    MatchupPredictor,
    PolicyBeliefAuxHead,
    belief_loss,
    fit_format_prior,
    pack_history_batches,
    save_belief_net,
    save_format_prior,
)
from jace.cards import CardTable, own_parameters
from jace.forks.batch import _engine_error_types, _safe_encode_one
from jace.forks.collect import concat_observation_batches
from jace.forks.search import extract_public_belief_inputs
from jace.storage.gameplay import RunLayout, append_metric
from jace.train._belief_data import BeliefDataset, collect_belief_dataset, deck_count_vector
from jace.train.belief import (
    _predict_belief_batched,
    build_cooccurrence_card_table,
    evaluate_belief_metrics,
    evaluate_v0_metrics,
)

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from manabot.backends.torch.policy import ActorCritic
    from manabot.env.observation import ObservationSpace
    from manabot.policy.prepare import Prepared

    from jace.collect.decks import HumanDeckCatalog
    from jace.rl.loop import AuxLossFn, PrepareAuxFn

__all__ = (
    "BeliefStreamSpec",
    "BeliefStreamTrainer",
    "add_belief_stream_cli_args",
    "compose_prepare_aux",
    "open_belief_stream_trainer",
)

type _HistorySnap = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
type _SampledStep = tuple[Tensor, Tensor, int, int, Tensor, Tensor, _HistorySnap, managym.ObservationBatch | None]

_BASIC_LANDS = frozenset({"Plains", "Island", "Swamp", "Mountain", "Forest"})
_NUM_PLAYERS = 2
_BASIC_LIMIT = 20.0
_DEFAULT_LIMIT = 4.0
_MIN_AUX_ROWS = 4
_MIN_TRAIN_ROWS = 32
_MIN_V0_ROWS = 8
_MAX_REPLAY_ACTIONS = 600


@dataclass(frozen=True, slots=True)
class BeliefStreamSpec:
    enabled: bool = False
    every: int = 1
    steps_per_update: int = 4
    batch_size: int = 256
    hidden_dim: int = 128
    lr: float = 1e-3
    capacity: int = 65_536
    sample_stride: int = 4
    max_games_per_ingest: int = 64
    eval_games: int = 60
    policy_aux_weight: float = 0.0
    policy_aux_batch_size: int = 64

    @property
    def active(self) -> bool:
        return self.enabled or self.policy_aux_weight > 0.0

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> BeliefStreamSpec:
        return cls(
            enabled=bool(ns.belief_stream),
            every=max(1, int(ns.belief_stream_every)),
            steps_per_update=max(1, int(ns.belief_stream_steps)),
            batch_size=max(8, int(ns.belief_stream_batch_size)),
            hidden_dim=max(16, int(ns.belief_stream_hidden_dim)),
            lr=float(ns.belief_stream_lr),
            capacity=max(256, int(ns.belief_stream_capacity)),
            sample_stride=max(1, int(ns.belief_stream_stride)),
            max_games_per_ingest=max(1, int(ns.belief_stream_max_games)),
            eval_games=max(0, int(ns.belief_stream_eval_games)),
            policy_aux_weight=max(0.0, float(ns.belief_aux_weight)),
            policy_aux_batch_size=max(4, int(ns.belief_aux_batch_size)),
        )


def add_belief_stream_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--belief-stream", action="store_true", default=False)
    parser.add_argument("--belief-stream-every", type=int, default=1)
    parser.add_argument("--belief-stream-steps", type=int, default=4)
    parser.add_argument("--belief-stream-batch-size", type=int, default=256)
    parser.add_argument("--belief-stream-hidden-dim", type=int, default=128)
    parser.add_argument("--belief-stream-lr", type=float, default=1e-3)
    parser.add_argument("--belief-stream-capacity", type=int, default=65_536)
    parser.add_argument("--belief-stream-stride", type=int, default=4)
    parser.add_argument("--belief-stream-max-games", type=int, default=64)
    parser.add_argument("--belief-stream-eval-games", type=int, default=60)
    parser.add_argument("--belief-aux-weight", type=float, default=0.0)
    parser.add_argument("--belief-aux-batch-size", type=int, default=64)


def _combine_aux_terms(fn_a: AuxLossFn, fn_b: AuxLossFn) -> AuxLossFn:
    def _sum() -> Tensor | None:
        a_val = fn_a()
        b_val = fn_b()
        if a_val is None or b_val is None:
            return b_val if a_val is None else a_val
        return a_val + b_val.to(device=a_val.device, dtype=a_val.dtype)

    return _sum


def compose_prepare_aux(first: PrepareAuxFn | None, second: PrepareAuxFn | None) -> PrepareAuxFn | None:
    if first is None or second is None:
        return second if first is None else first

    def _combined(agent: ActorCritic, update_idx: int) -> AuxLossFn | None:
        fn_a = first(agent, update_idx)
        fn_b = second(agent, update_idx)
        if fn_a is None or fn_b is None:
            return fn_b if fn_a is None else fn_a
        return _combine_aux_terms(fn_a, fn_b)

    return _combined


@dataclass(frozen=True, slots=True)
class _StreamRow:
    deck_prior: Tensor
    revealed: Tensor
    signals: Tensor
    hand_size: int
    lib_size: int
    target_deck: Tensor
    target_hand: Tensor
    history: _HistorySnap


@dataclass(frozen=True, slots=True)
class _PolicyAuxRow:
    encoded_obs: managym.ObservationBatch
    deck_prior: Tensor
    revealed: Tensor
    hand_size: int
    lib_size: int
    target_deck: Tensor
    target_hand: Tensor


def _copy_limits_for_catalog(catalog: HumanDeckCatalog) -> Tensor:
    max_seen = catalog.train_decks.to(dtype=torch.float32).amax(dim=0)
    base = torch.tensor(
        [_BASIC_LIMIT if name in _BASIC_LANDS else _DEFAULT_LIMIT for name in catalog.card_names],
        dtype=torch.float32,
    )
    return torch.maximum(base, max_seen)


def _extract_games_from_line(raw_line: str) -> list[dict[str, object]]:
    stripped = raw_line.strip()
    if not stripped:
        return []
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    games = payload.get("games")
    return [g for g in games if isinstance(g, dict)] if isinstance(games, list) else []


def _player_configs_from_record(raw_players: object) -> tuple[list[managym.PlayerConfig], list[dict[str, int]]] | None:
    if not isinstance(raw_players, list) or len(raw_players) < _NUM_PLAYERS:
        return None
    configs: list[managym.PlayerConfig] = []
    decks: list[dict[str, int]] = []
    for idx, entry in enumerate(raw_players[:_NUM_PLAYERS]):
        if not isinstance(entry, dict):
            return None
        raw_deck = entry.get("decklist")
        if not isinstance(raw_deck, dict) or not raw_deck:
            return None
        deck = {str(k): int(v) for k, v in raw_deck.items() if isinstance(v, (int, float)) and int(v) > 0}
        raw_sb = entry.get("sideboard")
        sb = (
            {str(k): int(v) for k, v in raw_sb.items() if isinstance(v, (int, float)) and int(v) > 0}
            if isinstance(raw_sb, dict)
            else {}
        )
        configs.append(managym.PlayerConfig(name=str(entry.get("name", f"P{idx}")), decklist=deck, sideboard=sb))
        decks.append(deck)
    return configs, decks


def _read_new_games(path: Path, offset: int) -> tuple[list[dict[str, object]], int]:
    if not path.is_file() or path.stat().st_size <= offset:
        return [], offset
    games: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        handle.seek(offset)
        for line in handle:
            games.extend(_extract_games_from_line(line))
        new_offset = handle.tell()
    return games, new_offset


def _step_replay_samples(
    env: managym.Env,
    obs: managym.Observation,
    *,
    actions: list[object],
    card_names: tuple[str, ...],
    name_to_idx: dict[str, int],
    deck_vecs: tuple[Tensor, Tensor],
    sample_stride: int,
    capture_obs: bool,
    obs_space: ObservationSpace,
) -> list[_SampledStep]:
    buffers = (BeliefHistoryBuffer(len(card_names)), BeliefHistoryBuffer(len(card_names)))
    sampled: list[_SampledStep] = []
    for step_idx, raw_act in enumerate(actions[:_MAX_REPLAY_ACTIONS]):
        if not isinstance(raw_act, int):
            break
        acting = int(obs.agent.player_index)
        opp = 1 - acting
        revealed, signals, hand_sz, lib_sz = extract_public_belief_inputs(obs, card_names=card_names, hero_seat=acting)
        buffers[acting].record_step(
            revealed_counts=revealed, gameplay_signals=signals, opp_hand_size=hand_sz, opp_library_size=lib_sz
        )
        opp_hand_sz = int(obs.opponent.zone_counts[int(ZoneEnum.HAND)])
        if opp_hand_sz > 0 and step_idx % sample_stride == 0:
            target_hand = deck_count_vector(env.hand_snapshot(opp).cards, name_to_idx=name_to_idx)
            target_deck = (deck_vecs[opp] - revealed).clamp_min(0.0)
            encoded = _safe_encode_one(obs, obs_space=obs_space) if capture_obs else None
            hist_snap = buffers[acting].snapshot_tensors()
            sampled.append((revealed, signals, hand_sz, lib_sz, target_deck, target_hand, hist_snap, encoded))
        obs, _, term, trunc, _ = env.step(raw_act)
        if term or trunc:
            break
    return sampled


class BeliefStreamTrainer:
    def __init__(
        self,
        spec: BeliefStreamSpec,
        *,
        catalog: HumanDeckCatalog,
        obs_space: ObservationSpace,
        agent: ActorCritic,
        layout: RunLayout,
        device: str,
        seed: int = 1,
    ) -> None:
        self.spec = spec
        self._catalog = catalog
        self._obs_space = obs_space
        self._layout = layout
        self._device = torch.device(device)
        self._rng = random.Random(seed + 42_001)
        self._name_to_idx = {name: i for i, name in enumerate(catalog.card_names)}
        self._prior_backend = BayesianDeckPoolPrior()
        self._file_offset = 0
        self._games_ingested = 0
        self._aux_grad_pending = False

        # FormatPrior and CardTable stay strictly frozen on human train_decks (zero builder-deck drift).
        self.prior: FormatPrior = fit_format_prior(
            "tla-limited-rl-stream",
            catalog.card_names,
            catalog.train_decks,
            copy_limits=_copy_limits_for_catalog(catalog),
        )
        self.card_table: CardTable = build_cooccurrence_card_table(catalog.train_decks)
        self.belief_net = BeliefNet(
            signal_dim=GAMEPLAY_SIGNAL_DIM,
            hidden_dim=spec.hidden_dim,
            embedding=self.card_table,
            dropout=0.2,
            use_history=True,
        ).to(self._device)
        self.v0 = MatchupPredictor(self.card_table, width=32, layers=2, heads=4, dropout=0.3).to(self._device)
        self._belief_opt = torch.optim.AdamW(
            own_parameters(self.belief_net, self.card_table), lr=spec.lr, weight_decay=1e-2
        )
        self._v0_opt = torch.optim.AdamW(own_parameters(self.v0, self.card_table), lr=spec.lr, weight_decay=5e-2)

        self.policy_aux_head: PolicyBeliefAuxHead | None = None
        self._aux_opt: torch.optim.Optimizer | None = None
        self._aux_uses_decision_memory = agent.decision_memory is not None
        if spec.policy_aux_weight > 0.0:
            if agent.decision_memory is None and agent.policy_film is None:
                msg = "--belief-aux-weight > 0 requires policy_film or decision_memory on ActorCritic"
                raise ValueError(msg)
            self.policy_aux_head = PolicyBeliefAuxHead(
                memory_dim=agent.config.hidden_dim,
                embedding=self.card_table,
                hidden_dim=spec.hidden_dim,
            ).to(self._device)
            self._aux_opt = torch.optim.AdamW(
                own_parameters(self.policy_aux_head, self.card_table), lr=spec.lr, weight_decay=1e-2
            )

        self._rows: deque[_StreamRow] = deque(maxlen=spec.capacity)
        self._aux_rows: deque[_PolicyAuxRow] = deque(maxlen=max(512, spec.policy_aux_batch_size * 16))
        self._outcomes: deque[tuple[Tensor, Tensor, float]] = deque(maxlen=max(1024, spec.capacity // 4))
        self._val_ds: BeliefDataset | None = None
        self._save_prior()

    def _save_prior(self) -> None:
        belief_dir = self._layout.root / "belief"
        belief_dir.mkdir(parents=True, exist_ok=True)
        save_format_prior(belief_dir / "prior.pt", self.prior)

    def ingest_recorded_games(self, jsonl_path: Path | None = None) -> int:
        path = self._layout.train_games_path() if jsonl_path is None else jsonl_path
        games, self._file_offset = _read_new_games(path, self._file_offset)
        if not games:
            return 0
        trimmed = games[-self.spec.max_games_per_ingest :]
        capture_obs = self.policy_aux_head is not None
        ingested = sum(1 for game in trimmed if self._replay_one_game(game, capture_obs=capture_obs))
        self._games_ingested += ingested
        return ingested

    def _store_samples(self, sampled: list[_SampledStep]) -> None:
        if not sampled:
            return
        rev_batch = torch.stack([item[0] for item in sampled], dim=0)
        priors = self._prior_backend.posterior_deck(rev_batch, self.prior)
        for idx, item in enumerate(sampled):
            prior_row = priors[idx]
            self._rows.append(
                _StreamRow(
                    deck_prior=prior_row,
                    revealed=item[0],
                    signals=item[1],
                    hand_size=item[2],
                    lib_size=item[3],
                    target_deck=item[4],
                    target_hand=item[5],
                    history=item[6],
                )
            )
            if item[7] is not None:
                self._aux_rows.append(
                    _PolicyAuxRow(
                        encoded_obs=item[7],
                        deck_prior=prior_row,
                        revealed=item[0],
                        hand_size=item[2],
                        lib_size=item[3],
                        target_deck=item[4],
                        target_hand=item[5],
                    )
                )

    def _replay_one_game(self, game: dict[str, object], *, capture_obs: bool) -> bool:
        parsed = _player_configs_from_record(game.get("players"))
        raw_actions = game.get("actions")
        if parsed is None or not isinstance(raw_actions, list) or not raw_actions:
            return False
        configs, decks = parsed
        raw_seed = game.get("seed")
        seed = raw_seed if isinstance(raw_seed, int) else 0
        deck_vecs = (
            deck_count_vector(decks[0], name_to_idx=self._name_to_idx),
            deck_count_vector(decks[1], name_to_idx=self._name_to_idx),
        )
        env = managym.Env(seed=seed, skip_trivial=True, decision_memory=capture_obs and self._aux_uses_decision_memory)
        try:
            obs, _ = env.reset(managym.GameSetup(players=configs, decklists="closed"))
            sampled = _step_replay_samples(
                env,
                obs,
                actions=raw_actions,
                card_names=self.prior.card_names,
                name_to_idx=self._name_to_idx,
                deck_vecs=deck_vecs,
                sample_stride=self.spec.sample_stride,
                capture_obs=capture_obs,
                obs_space=self._obs_space,
            )
        except _engine_error_types():
            return False
        self._store_samples(sampled)
        raw_end = game.get("end")
        winner = raw_end.get("winner") if isinstance(raw_end, dict) else None
        if isinstance(winner, int) and winner in (0, 1):
            score_a = 1.0 if winner == 0 else 0.0
            self._outcomes.append((deck_vecs[0], deck_vecs[1], score_a))
            self._outcomes.append((deck_vecs[1], deck_vecs[0], 1.0 - score_a))
        return True

    def _step_policy_aux_head(self) -> None:
        if self._aux_grad_pending and self.policy_aux_head is not None and self._aux_opt is not None:
            torch.nn.utils.clip_grad_norm_(self.policy_aux_head.parameters(), 1.0)
            self._aux_opt.step()
            self._aux_opt.zero_grad(set_to_none=True)
            self._aux_grad_pending = False

    def step_after_update(self, update_idx: int) -> dict[str, float] | None:
        self._step_policy_aux_head()
        self.ingest_recorded_games()
        if not self.spec.enabled or update_idx % self.spec.every != 0:
            return None
        if len(self._rows) < _MIN_TRAIN_ROWS:
            return None

        limits_dev = None if self.prior.copy_limits is None else self.prior.copy_limits.to(device=self._device)
        rows_list = list(self._rows)
        self.belief_net.train()
        b_losses: list[float] = []
        for _ in range(self.spec.steps_per_update):
            batch_rows = self._rng.sample(rows_list, k=min(self.spec.batch_size, len(rows_list)))
            deck_p = torch.stack([r.deck_prior for r in batch_rows]).to(self._device)
            rev = torch.stack([r.revealed for r in batch_rows]).to(self._device)
            sig = torch.stack([r.signals for r in batch_rows]).to(self._device)
            h_sz = torch.tensor([r.hand_size for r in batch_rows], dtype=torch.int64, device=self._device)
            l_sz = torch.tensor([r.lib_size for r in batch_rows], dtype=torch.int64, device=self._device)
            t_deck = torch.stack([r.target_deck for r in batch_rows]).to(self._device)
            t_hand = torch.stack([r.target_hand for r in batch_rows]).to(self._device)
            hist = pack_history_batches([r.history for r in batch_rows]).to(self._device)

            self._belief_opt.zero_grad(set_to_none=True)
            pd, ph = self.belief_net(deck_p, rev, sig, h_sz, l_sz, copy_limits=limits_dev, history=hist)
            loss = belief_loss(pd, ph, target_deck=t_deck, target_hand=t_hand)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.belief_net.parameters(), 1.0)
            self._belief_opt.step()
            b_losses.append(float(loss.item()))

        v0_loss_val = self._train_v0_steps()
        metrics = {
            "belief_stream_games": float(self._games_ingested),
            "belief_stream_rows": float(len(self._rows)),
            "belief_stream_train_loss": round(sum(b_losses) / len(b_losses), 4),
            "v0_stream_train_loss": round(v0_loss_val, 4),
        }
        append_metric(self._layout, {"kind": "belief_stream", "update": update_idx, **metrics})
        return metrics

    def _train_v0_steps(self) -> float:
        if len(self._outcomes) < _MIN_V0_ROWS:
            return math.log(2.0)
        out_list = list(self._outcomes)
        loss_fn = nn.BCEWithLogitsLoss()
        field_dev = self.prior.mean_copies.to(device=self._device, dtype=torch.float32)
        self.v0.train()
        v0_losses: list[float] = []
        for _ in range(self.spec.steps_per_update):
            sample_o = self._rng.sample(out_list, k=min(self.spec.batch_size, len(out_list)))
            my_d = torch.stack([o[0] for o in sample_o]).to(self._device)
            opp_d = torch.stack([o[1] for o in sample_o]).to(self._device)
            y = torch.tensor([o[2] for o in sample_o], dtype=torch.float32, device=self._device)
            self._v0_opt.zero_grad(set_to_none=True)
            v_loss = loss_fn(self.v0(my_d, opp_d), y) + loss_fn(self.v0(my_d, field_dev), y)
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.v0.parameters(), 1.0)
            self._v0_opt.step()
            v0_losses.append(float(v_loss.item()) * 0.5)
        return sum(v0_losses) / len(v0_losses)

    def prepare_policy_belief_aux(self, agent: ActorCritic, update_idx: int) -> AuxLossFn | None:
        if self.policy_aux_head is None or self._aux_opt is None:
            return None
        self.ingest_recorded_games()
        if len(self._aux_rows) < _MIN_AUX_ROWS:
            return None
        del update_idx
        aux_list = list(self._aux_rows)
        picked = self._rng.sample(aux_list, k=min(self.spec.policy_aux_batch_size, len(aux_list)))
        batch_obs = concat_observation_batches([r.encoded_obs for r in picked])
        prepared = on_device(agent.prepare(batch_obs), self._device)
        deck_p = torch.stack([r.deck_prior for r in picked]).to(self._device)
        rev = torch.stack([r.revealed for r in picked]).to(self._device)
        h_sz = torch.tensor([r.hand_size for r in picked], dtype=torch.int64, device=self._device)
        l_sz = torch.tensor([r.lib_size for r in picked], dtype=torch.int64, device=self._device)
        t_deck = torch.stack([r.target_deck for r in picked]).to(self._device)
        t_hand = torch.stack([r.target_hand for r in picked]).to(self._device)
        limits_dev = None if self.prior.copy_limits is None else self.prior.copy_limits.to(device=self._device)
        weight = self.spec.policy_aux_weight
        aux_head = self.policy_aux_head
        aux_opt = self._aux_opt
        used = {"done": False}

        def _aux_loss() -> Tensor | None:
            if used["done"]:
                return None
            used["done"] = True
            aux_head.train()
            aux_opt.zero_grad(set_to_none=True)
            summaries = _extract_policy_belief_features(agent, prepared)
            pd, ph = aux_head(
                summaries,
                deck_prior=deck_p,
                revealed_counts=rev,
                opp_hand_size=h_sz,
                opp_library_size=l_sz,
                copy_limits=limits_dev,
            )
            self._aux_grad_pending = True
            return weight * belief_loss(pd, ph, target_deck=t_deck, target_hand=t_hand)

        return _aux_loss

    def evaluate_and_save(self, update_idx: int) -> dict[str, object] | None:
        belief_dir = self._layout.root / "belief"
        belief_dir.mkdir(parents=True, exist_ok=True)
        save_belief_net(belief_dir / "belief_net.pt", self.belief_net)
        torch.save(
            {
                "width": self.v0.width,
                "card_table": self.card_table.table.detach().cpu(),
                "state_dict": self.v0.state_dict(),
            },
            belief_dir / "v0.pt",
        )
        if not self.spec.enabled or self.spec.eval_games <= 0:
            return None
        if self._val_ds is None:
            self._val_ds = collect_belief_dataset(
                catalog=self._catalog,
                prior=self.prior,
                split="held_out",
                num_games=self.spec.eval_games,
                seed=9901,
                checkpoint=None,
                num_workers=1,
                sample_stride=self.spec.sample_stride,
            )
        val_ds = self._val_ds
        pred_d, pred_h = _predict_belief_batched(
            self.belief_net, val_ds, copy_limits=self.prior.copy_limits, device=self._device
        )
        b_metrics = evaluate_belief_metrics(pred_d, pred_h, dataset=val_ds)
        self.v0.eval()
        with torch.no_grad():
            v0_p = (
                self.v0.predict_equity(
                    val_ds.matchup_my_decks.to(self._device),
                    val_ds.matchup_opp_decks.to(self._device),
                ).cpu()
                if val_ds.matchup_outcomes.numel() > 0
                else torch.empty(0)
            )
        v0_metrics = evaluate_v0_metrics(v0_p, val_ds.matchup_outcomes)
        summary: dict[str, object] = {
            "kind": "belief_eval",
            "update": update_idx,
            "belief_stream_games": self._games_ingested,
            "belief_val_deck_log_loss": b_metrics["deck_log_loss"],
            "belief_val_hand_log_loss": b_metrics["hand_log_loss"],
            "belief_val_total_log_loss": b_metrics["total_log_loss"],
            "belief_val_recall_at_5": b_metrics["hand_recall_at_5"],
            "belief_val_recall_at_10": b_metrics["hand_recall_at_10"],
            "v0_val_bce": v0_metrics["bce"],
            "v0_val_roc_auc": v0_metrics["roc_auc"],
        }
        append_metric(self._layout, summary)
        return summary


def _extract_policy_belief_features(agent: ActorCritic, prepared: Prepared[Tensor]) -> Tensor:
    if agent.decision_memory is not None:
        return agent.encode_decision_memory(prepared)
    if agent.policy_film is None:
        msg = "ActorCritic must expose decision_memory or policy_film for policy belief auxiliary loss"
        raise RuntimeError(msg)
    captured: list[Tensor] = []

    def _capture(_mod: nn.Module, args: tuple[object, ...]) -> None:
        if args and isinstance(args[0], Tensor):
            captured.append(args[0])

    handle = agent.policy_film.register_forward_pre_hook(_capture)
    try:
        agent.forward_distribution(prepared)
    finally:
        handle.remove()
    if not captured:
        msg = "failed to capture pooled policy representation from policy_film"
        raise RuntimeError(msg)
    return captured[0]


def open_belief_stream_trainer(
    spec: BeliefStreamSpec,
    *,
    catalog: HumanDeckCatalog,
    obs_space: ObservationSpace,
    agent: ActorCritic,
    layout: RunLayout,
    device: str,
    seed: int,
) -> BeliefStreamTrainer | None:
    if not spec.active:
        return None
    return BeliefStreamTrainer(
        spec,
        catalog=catalog,
        obs_space=obs_space,
        agent=agent,
        layout=layout,
        device=device,
        seed=seed,
    )

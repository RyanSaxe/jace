"""Parallel closed-decklist gameplay data collector for `BeliefNet` and Turn-0 `V_0`.

Plays real closed-decklist games on `train` or `held_out` deck splits using a
policy checkpoint (or `heuristic`), recording exact public belief inputs,
rolling multi-turn event and earlier-belief histories, ground-truth hidden
opponent hand and unseen deck targets, and game outcomes for supervised
training and held-out evaluation.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import managym
import torch
from manabot.backends.torch.checkpoint import load_checkpoint
from managym.enums import ZoneEnum

from jace.belief import (
    BayesianDeckPoolPrior,
    BeliefHistoryBatch,
    BeliefHistoryBuffer,
    FormatPrior,
    pack_history_batches,
)
from jace.collect.decks import HumanDeckCatalog, load_human_deck_catalog
from jace.eval.deals import Deal, PlayerFn, PolicyPlayer, build_human_deals, heuristic_player
from jace.forks.batch import _engine_error_types
from jace.forks.search import extract_public_belief_inputs

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from torch import Tensor

__all__ = (
    "BeliefDataset",
    "collect_belief_dataset",
    "deck_count_vector",
)

_LOGGER = logging.getLogger(__name__)
_PRIOR_CHUNK = 256

type _HistorySnap = tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]
type _GameRow = tuple[Tensor, Tensor, int, int, int, Tensor, Tensor, _HistorySnap]


@dataclass(frozen=True, slots=True)
class BeliefDataset:
    deck_priors: Tensor
    revealed_counts: Tensor
    gameplay_signals: Tensor
    opp_hand_sizes: Tensor
    opp_library_sizes: Tensor
    turns: Tensor
    target_deck_copies: Tensor
    target_hand_copies: Tensor
    matchup_my_decks: Tensor
    matchup_opp_decks: Tensor
    matchup_outcomes: Tensor
    history: BeliefHistoryBatch | None = None


def deck_count_vector(deck: Mapping[str, int], *, name_to_idx: Mapping[str, int]) -> Tensor:
    vec = torch.zeros(len(name_to_idx), dtype=torch.float32)
    for name, count in deck.items():
        if (idx := name_to_idx.get(name)) is not None and count > 0:
            vec[idx] = float(count)
    return vec


def _collect_game_rows(
    deal: Deal,
    *,
    player: PlayerFn,
    card_names: tuple[str, ...],
    name_to_idx: dict[str, int],
    max_actions: int,
    sample_stride: int,
    use_mem: bool,
) -> tuple[list[_GameRow], tuple[Tensor, Tensor, float] | None]:
    env = managym.Env(seed=deal.seed, skip_trivial=True, decision_memory=use_mem)
    deck_vecs = (
        deck_count_vector(deal.deck_a, name_to_idx=name_to_idx),
        deck_count_vector(deal.deck_b, name_to_idx=name_to_idx),
    )
    buffers = (BeliefHistoryBuffer(len(card_names)), BeliefHistoryBuffer(len(card_names)))
    rows: list[_GameRow] = []
    outcome_pair: tuple[Tensor, Tensor, float] | None = None
    try:
        obs, _ = env.reset(deal.game_setup(swapped=False))
        term = trunc = False
        step_idx = 0
        while not term and not trunc and step_idx < max_actions:
            acting = int(obs.agent.player_index)
            opp = 1 - acting
            revealed, signals, hand_sz, lib_sz = extract_public_belief_inputs(
                obs, card_names=card_names, hero_seat=acting
            )
            buffers[acting].record_step(
                revealed_counts=revealed,
                gameplay_signals=signals,
                opp_hand_size=hand_sz,
                opp_library_size=lib_sz,
            )
            opp_hand_sz = int(obs.opponent.zone_counts[int(ZoneEnum.HAND)])
            if opp_hand_sz > 0 and step_idx % sample_stride == 0:
                hand_snap = env.hand_snapshot(opp).cards
                target_hand = deck_count_vector(hand_snap, name_to_idx=name_to_idx)
                target_deck = (deck_vecs[opp] - revealed).clamp_min(0.0)
                hist_snap = buffers[acting].snapshot_tensors()
                rows.append(
                    (revealed, signals, hand_sz, lib_sz, int(obs.turn.turn_number), target_deck, target_hand, hist_snap)
                )
            action = int(player(obs))
            obs, _, term, trunc, _ = env.step(action)
            step_idx += 1
        winner = env.winner_index() if term else None
        score_a = 0.5 if winner is None else (1.0 if int(winner) == 0 else 0.0)
        outcome_pair = (deck_vecs[0], deck_vecs[1], score_a)
    except _engine_error_types() as exc:
        _LOGGER.warning("belief collector aborted deal seed=%d (%s): %s", deal.seed, type(exc).__name__, exc)
    return rows, outcome_pair


def _worker_collect(
    args: tuple[Path | None, list[Deal], tuple[str, ...], int, int, int, bool],
) -> tuple[list[_GameRow], list[tuple[Tensor, Tensor, float]]]:
    ckpt_path, deals, card_names, seed, max_actions, sample_stride, deterministic = args
    logging.getLogger("manabot").setLevel(logging.WARNING)
    torch.set_num_threads(1)
    name_to_idx = {name: i for i, name in enumerate(card_names)}
    rows_all: list[_GameRow] = []
    outcomes_all: list[tuple[Tensor, Tensor, float]] = []

    if ckpt_path is None:
        player = heuristic_player()
        for deal in deals:
            rows, outcome = _collect_game_rows(
                deal,
                player=player,
                card_names=card_names,
                name_to_idx=name_to_idx,
                max_actions=max_actions,
                sample_stride=sample_stride,
                use_mem=False,
            )
            rows_all.extend(rows)
            if outcome is not None:
                outcomes_all.append(outcome)
        return rows_all, outcomes_all

    policy = load_checkpoint(str(ckpt_path))[0].eval()
    player = PolicyPlayer(policy, policy.observation_space, seed=seed, deterministic=deterministic)
    use_mem = player.needs_decision_memory
    with player:
        for deal in deals:
            rows, outcome = _collect_game_rows(
                deal,
                player=player,
                card_names=card_names,
                name_to_idx=name_to_idx,
                max_actions=max_actions,
                sample_stride=sample_stride,
                use_mem=use_mem,
            )
            rows_all.extend(rows)
            if outcome is not None:
                outcomes_all.append(outcome)
    return rows_all, outcomes_all


def _compute_bayesian_priors(revealed_counts: Tensor, prior: FormatPrior) -> Tensor:
    backend = BayesianDeckPoolPrior()
    chunks: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, revealed_counts.shape[0], _PRIOR_CHUNK):
            chunk = revealed_counts[start : start + _PRIOR_CHUNK]
            chunks.append(backend.posterior_deck(chunk, prior))
    return torch.cat(chunks, dim=0) if chunks else torch.zeros_like(revealed_counts)


def _pack_dataset(
    rows: Sequence[_GameRow],
    outcomes: Sequence[tuple[Tensor, Tensor, float]],
    *,
    prior: FormatPrior,
) -> BeliefDataset:
    num_cards = len(prior.card_names)
    if not rows:
        msg = "belief data collection produced zero decision rows"
        raise RuntimeError(msg)
    revealed = torch.stack([r[0] for r in rows])
    signals = torch.stack([r[1] for r in rows])
    hand_sizes = torch.tensor([r[2] for r in rows], dtype=torch.int64)
    lib_sizes = torch.tensor([r[3] for r in rows], dtype=torch.int64)
    turns = torch.tensor([r[4] for r in rows], dtype=torch.int64)
    target_deck = torch.stack([r[5] for r in rows])
    target_hand = torch.stack([r[6] for r in rows])
    history = pack_history_batches([r[7] for r in rows])
    deck_priors = _compute_bayesian_priors(revealed, prior)

    if outcomes:
        my_list: list[Tensor] = []
        opp_list: list[Tensor] = []
        out_list: list[float] = []
        for deck_a, deck_b, score_a in outcomes:
            my_list.extend((deck_a, deck_b))
            opp_list.extend((deck_b, deck_a))
            out_list.extend((score_a, 1.0 - score_a))
        my_decks = torch.stack(my_list)
        opp_decks = torch.stack(opp_list)
        matchup_outcomes = torch.tensor(out_list, dtype=torch.float32)
    else:
        my_decks = torch.zeros((0, num_cards), dtype=torch.float32)
        opp_decks = torch.zeros((0, num_cards), dtype=torch.float32)
        matchup_outcomes = torch.zeros((0,), dtype=torch.float32)

    return BeliefDataset(
        deck_priors=deck_priors,
        revealed_counts=revealed,
        gameplay_signals=signals,
        opp_hand_sizes=hand_sizes,
        opp_library_sizes=lib_sizes,
        turns=turns,
        target_deck_copies=target_deck,
        target_hand_copies=target_hand,
        matchup_my_decks=my_decks,
        matchup_opp_decks=opp_decks,
        matchup_outcomes=matchup_outcomes,
        history=history,
    )


def collect_belief_dataset(
    *,
    catalog: HumanDeckCatalog | Path,
    prior: FormatPrior,
    split: Literal["train", "held_out"],
    num_games: int,
    seed: int,
    checkpoint: Path | None = None,
    num_workers: int = 4,
    max_actions: int = 600,
    sample_stride: int = 2,
    deterministic: bool = False,
) -> BeliefDataset:
    loaded = catalog if isinstance(catalog, HumanDeckCatalog) else load_human_deck_catalog(catalog)
    deals = build_human_deals(loaded, split=split, count=num_games, seed=seed, decklists="closed")
    workers = max(1, min(num_workers, len(deals)))
    shards = [[d for i, d in enumerate(deals) if i % workers == r] for r in range(workers)]
    tasks = [
        (checkpoint, shard, prior.card_names, seed + r * 1009, max_actions, max(1, sample_stride), deterministic)
        for r, shard in enumerate(shards)
        if shard
    ]
    rows: list[_GameRow] = []
    outcomes: list[tuple[Tensor, Tensor, float]] = []
    if len(tasks) == 1:
        r_part, o_part = _worker_collect(tasks[0])
        rows.extend(r_part)
        outcomes.extend(o_part)
    else:
        with mp.get_context("spawn").Pool(len(tasks)) as pool:
            for r_part, o_part in pool.map(_worker_collect, tasks):
                rows.extend(r_part)
                outcomes.extend(o_part)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    shuffled_rows = [rows[i] for i in order]
    return _pack_dataset(shuffled_rows, outcomes, prior=prior)

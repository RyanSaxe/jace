"""Unit tests for `jace.belief` priors, `BeliefNet`, override projection, and `V_0`."""

from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

from jace.belief import (
    GAMEPLAY_SIGNAL_DIM,
    BayesianDeckPoolPrior,
    BeliefNet,
    BeliefOverride,
    BeliefState,
    MatchupPredictor,
    SetTransformerPrior,
    evaluate_belief_log_loss,
    evaluate_opening_hand_vs_v0,
    fit_format_prior,
    infer_belief_state,
    load_belief_net,
    load_format_prior,
    project_belief_overrides,
    save_belief_net,
    save_format_prior,
    score_deck_vs_field,
    train_belief_net,
)
from jace.cards import CardTable, own_parameters

if TYPE_CHECKING:
    from jace.decks.score import Scorer

__all__ = ()

CARD_NAMES = (
    "Mountain",
    "Island",
    "Lightning Bolt",
    "Goblin Guide",
    "Counterspell",
    "Opt",
    "Wrath of God",
    "Day-0 Spoiler",
)


def _card_table(num_cards: int = len(CARD_NAMES), width: int = 8) -> CardTable:
    vectors = torch.randn((num_cards, width), generator=torch.Generator().manual_seed(42))
    return CardTable(vectors)


def _sample_meta_decks() -> Tensor:
    # Two sharp archetypes: Red Aggro (cards 0, 2, 3) and Blue Control (cards 1, 4, 5, 6); card 7 is day-0 unseen.
    aggro = torch.tensor([12.0, 0.0, 4.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    control = torch.tensor([0.0, 12.0, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0])
    azorius = torch.tensor([0.0, 10.0, 0.0, 0.0, 4.0, 2.0, 4.0, 0.0])
    return torch.stack((aggro, aggro, control, azorius))


def _make_belief_state(*, hand_size: int, library_size: int) -> BeliefState:
    unseen_total = float(hand_size + library_size)
    raw = torch.tensor([6.0, 6.0, 4.0, 4.0, 4.0, 4.0, 2.0, 2.0])
    deck_copies = raw * (unseen_total / float(raw.sum().item())) if unseen_total > 0 else torch.zeros_like(raw)
    hand_probs = deck_copies * (hand_size / unseen_total) if unseen_total > 0 else torch.zeros_like(raw)
    return BeliefState(
        card_names=CARD_NAMES,
        deck_copies=deck_copies,
        hand_probs=hand_probs,
        opp_hand_size=hand_size,
        opp_library_size=library_size,
        prior_backend="bayesian_pool",
    )


@pytest.mark.parametrize("hand_size", [0, 4, 7, 11, 15])
def test_belief_net_supports_unconstrained_hand_sizes(hand_size: int) -> None:
    model = BeliefNet(signal_dim=GAMEPLAY_SIGNAL_DIM, hidden_dim=16, embedding=_card_table())
    library_size = 25
    prior = fit_format_prior("TEST", CARD_NAMES, _sample_meta_decks())
    revealed = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    signals = torch.zeros(GAMEPLAY_SIGNAL_DIM)

    state = infer_belief_state(
        model,
        prior=prior,
        revealed_counts=revealed,
        gameplay_signals=signals,
        opp_hand_size=hand_size,
        opp_library_size=library_size,
    )

    assert state.opp_hand_size == hand_size
    assert state.opp_library_size == library_size
    assert state.deck_copies.shape == (len(CARD_NAMES),)
    assert state.hand_probs.shape == (len(CARD_NAMES),)
    assert torch.isclose(state.hand_probs.sum(), torch.tensor(float(hand_size)), atol=1e-3)
    assert torch.isclose(state.deck_copies.sum(), torch.tensor(float(hand_size + library_size)), atol=1e-3)
    assert bool((state.hand_probs >= -1e-6).all())
    assert bool((state.hand_probs <= state.deck_copies + 1e-4).all())


def test_bayesian_pool_prior_shifts_with_revealed_cards_and_metagame_updates(tmp_path: Path) -> None:
    decks = _sample_meta_decks()
    prior = fit_format_prior(
        "TEST",
        CARD_NAMES,
        decks,
        copy_limits=torch.tensor([20.0, 20.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0]),
    )
    backend = BayesianDeckPoolPrior()

    # Revealing Lightning Bolt eliminates Blue Control and Azorius immediately.
    revealed_bolt = torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    post_aggro = backend.posterior_deck(revealed_bolt, prior)
    assert float(post_aggro[3].item()) == pytest.approx(4.0, abs=1e-4)  # Goblin Guide
    assert float(post_aggro[4].item()) == pytest.approx(0.0, abs=1e-4)  # Counterspell
    assert float(post_aggro[2].item()) == pytest.approx(3.0, abs=1e-4)  # 4 - 1 Bolt left

    # Batched query matches single query and falls back smoothly when revealing a day-0 card.
    revealed_day0 = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    batched = backend.posterior_deck(torch.stack((revealed_bolt, revealed_day0)), prior)
    assert torch.allclose(batched[0], post_aggro, atol=1e-5)
    assert torch.isfinite(batched[1]).all()
    assert float(batched[1].sum().item()) > 0.0

    # Zero-retrain metagame shift: swap deck_pool so Mountain now pairs with Wrath of God instead of Goblin Guide.
    shifted_decks = torch.tensor(
        [
            [12.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0],
            [12.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0, 0.0],
        ]
    )
    shifted_prior = fit_format_prior("TEST-META2", CARD_NAMES, shifted_decks)
    post_shifted = backend.posterior_deck(revealed_bolt, shifted_prior)
    assert float(post_shifted[6].item()) == pytest.approx(4.0, abs=1e-4)
    assert float(post_shifted[3].item()) == pytest.approx(0.0, abs=1e-4)

    # Atomic save + weights_only=True load round-trip.
    prior_path = tmp_path / "format_prior.pt"
    save_format_prior(prior_path, prior)
    loaded = load_format_prior(prior_path)
    assert loaded.format_id == prior.format_id
    assert loaded.card_names == prior.card_names
    assert torch.allclose(loaded.mean_copies, prior.mean_copies)
    assert torch.allclose(loaded.cond_copies, prior.cond_copies)
    assert loaded.deck_pool is not None
    assert prior.deck_pool is not None
    assert torch.equal(loaded.deck_pool, prior.deck_pool)


def test_set_transformer_prior_smooths_unseen_day0_cards() -> None:
    embedding = _card_table()
    prior = fit_format_prior("TEST", CARD_NAMES, _sample_meta_decks())
    smoother = SetTransformerPrior(embedding, width=16, layers=1, heads=2)
    revealed = torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

    predicted = smoother(revealed, prior.mean_copies, prior.booster_fallback)
    assert predicted.shape == (2, len(CARD_NAMES))
    assert torch.isfinite(predicted).all()
    assert bool((predicted >= 0).all())
    # Day-0 card (index 7, mean_copies == 0) receives positive smoothed expectation.
    assert float(predicted[0, 7].item()) > 0.0
    assert len(own_parameters(smoother, embedding)) == len(list(smoother.parameters()))

    single_post = smoother.posterior_deck(revealed[0], prior)
    assert single_post.shape == (len(CARD_NAMES),)


def test_project_belief_overrides_enforces_all_four_invariants() -> None:
    belief = _make_belief_state(hand_size=10, library_size=22)
    revealed = torch.tensor([2.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    copy_limits = torch.tensor([20.0, 20.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

    override = BeliefOverride(
        pinned_in_hand={"Lightning Bolt": 2, "Counterspell": 2},
        excluded_cards=frozenset({"Wrath of God", "Day-0 Spoiler"}),
        copy_multipliers={"Goblin Guide": 2.0},
    )
    projected = project_belief_overrides(belief, [override], revealed_counts=revealed, copy_limits=copy_limits)

    # Invariant 1 & 2: pinned fixed, excluded zeroed, hand_probs sums to opp_hand_size, unedited scaled proportionally.
    assert float(projected.hand_probs[2].item()) == pytest.approx(2.0, abs=1e-5)
    assert float(projected.hand_probs[4].item()) == pytest.approx(2.0, abs=1e-5)
    assert float(projected.hand_probs[6].item()) == 0.0
    assert float(projected.hand_probs[7].item()) == 0.0
    assert float(projected.hand_probs.sum().item()) == pytest.approx(10.0, abs=1e-4)
    # Mountain (idx 0) and Island (idx 1) were unedited with equal initial mass -> remain equal after scaling.
    assert float(projected.hand_probs[0].item()) == pytest.approx(float(projected.hand_probs[1].item()), abs=1e-4)

    # Invariant 3: deck_copies >= hand_probs and <= max(0, copy_limits - revealed).
    max_unseen = (copy_limits - revealed).clamp_min(0.0)
    assert bool((projected.deck_copies >= projected.hand_probs - 1e-5).all())
    assert bool((projected.deck_copies <= max_unseen + 1e-5).all())
    assert float(projected.deck_copies[6].item()) == 0.0
    assert float(projected.deck_copies[7].item()) == 0.0

    # Invariant 4: total unseen deck copies equals opp_hand_size + opp_library_size.
    assert float(projected.deck_copies.sum().item()) == pytest.approx(32.0, abs=1e-4)


def test_project_belief_overrides_rejects_invalid_constraints() -> None:
    belief = _make_belief_state(hand_size=3, library_size=10)
    revealed = torch.tensor([0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    copy_limits = torch.tensor([20.0, 20.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])

    # Pinned total > opp_hand_size -> ValueError (HTTP 422 in Studio).
    with pytest.raises(ValueError, match="exceeds opp_hand_size"):
        project_belief_overrides(
            belief,
            [BeliefOverride(pinned_in_hand={"Mountain": 2, "Island": 2})],
            revealed_counts=revealed,
            copy_limits=copy_limits,
        )

    # Pinning more copies than copy_limit - revealed -> ValueError.
    with pytest.raises(ValueError, match="copy limit"):
        project_belief_overrides(
            belief,
            [BeliefOverride(pinned_in_hand={"Lightning Bolt": 2})],
            revealed_counts=revealed,
            copy_limits=copy_limits,
        )

    # Pinning an excluded card -> ValueError.
    with pytest.raises(ValueError, match="excluded_cards"):
        project_belief_overrides(
            belief,
            [BeliefOverride(pinned_in_hand={"Opt": 1}, excluded_cards=frozenset({"Opt"}))],
            revealed_counts=revealed,
            copy_limits=copy_limits,
        )

    # Negative pin count -> ValueError.
    with pytest.raises(ValueError, match="non-negative"):
        project_belief_overrides(
            belief,
            [BeliefOverride(pinned_in_hand={"Opt": -1})],
            revealed_counts=revealed,
            copy_limits=copy_limits,
        )


def test_project_belief_overrides_distributes_uniformly_when_unedited_mass_is_zero() -> None:
    deck_copies = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    hand_probs = torch.tensor([4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    belief = BeliefState(
        card_names=CARD_NAMES,
        deck_copies=deck_copies,
        hand_probs=hand_probs,
        opp_hand_size=4,
        opp_library_size=6,
        prior_backend="bayesian_pool",
    )
    revealed = torch.zeros(len(CARD_NAMES))
    # Excluding Mountain leaves zero mass on unedited cards; projection spreads uniformly across legal cards.
    projected = project_belief_overrides(
        belief,
        [BeliefOverride(excluded_cards=frozenset({"Mountain"}))],
        revealed_counts=revealed,
    )
    assert float(projected.hand_probs[0].item()) == 0.0
    assert float(projected.hand_probs.sum().item()) == pytest.approx(4.0, abs=1e-4)
    assert float(projected.deck_copies.sum().item()) == pytest.approx(10.0, abs=1e-4)
    assert bool((projected.deck_copies >= projected.hand_probs - 1e-5).all())


def test_belief_net_beats_format_prior_on_synthetic_heldout_log_loss() -> None:
    torch.manual_seed(7)
    num_cards = len(CARD_NAMES)
    embedding = _card_table(num_cards, width=8)
    prior = fit_format_prior("TEST", CARD_NAMES, _sample_meta_decks())

    def make_batch(size: int, seed: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        gen = torch.Generator().manual_seed(seed)
        is_blue_open = torch.randint(0, 2, (size,), generator=gen).bool()
        deck_priors = prior.mean_copies.unsqueeze(0).expand(size, -1).clone()
        revealed = torch.zeros((size, num_cards))
        signals = torch.zeros((size, GAMEPLAY_SIGNAL_DIM))
        # Signal 1 = open blue mana, Signal 3 = open red mana, Signal 9 = passed priority.
        signals[is_blue_open, 1] = 2.0
        signals[is_blue_open, 9] = 1.0
        signals[~is_blue_open, 3] = 2.0
        hand_sizes = torch.full((size,), 4, dtype=torch.int64)
        lib_sizes = torch.full((size,), 16, dtype=torch.int64)

        target_deck = torch.zeros((size, num_cards))
        target_hand = torch.zeros((size, num_cards))
        # Blue-open tells -> Counterspell + Opt + Island; Red-open tells -> Bolt + Guide + Mountain.
        target_hand[is_blue_open, 4] = 2.0
        target_hand[is_blue_open, 5] = 1.0
        target_hand[is_blue_open, 1] = 1.0
        target_deck[is_blue_open, 1] = 12.0
        target_deck[is_blue_open, 4] = 4.0
        target_deck[is_blue_open, 5] = 4.0

        target_hand[~is_blue_open, 2] = 2.0
        target_hand[~is_blue_open, 3] = 1.0
        target_hand[~is_blue_open, 0] = 1.0
        target_deck[~is_blue_open, 0] = 12.0
        target_deck[~is_blue_open, 2] = 4.0
        target_deck[~is_blue_open, 3] = 4.0
        return deck_priors, revealed, signals, hand_sizes, lib_sizes, target_deck, target_hand

    train_data = make_batch(64, seed=101)
    val_priors, val_rev, val_sig, val_hand, val_lib, val_tdeck, val_thand = make_batch(32, seed=202)

    # Prior baseline: scale mean_copies to unseen_total and hand_size proportionally.
    unseen = (val_hand + val_lib).to(torch.float32).unsqueeze(-1)
    prior_deck = val_priors * (unseen / val_priors.sum(dim=-1, keepdim=True))
    prior_hand = prior_deck * (val_hand.to(torch.float32).unsqueeze(-1) / unseen)
    baseline = evaluate_belief_log_loss(prior_deck, prior_hand, target_deck=val_tdeck, target_hand=val_thand)

    model = BeliefNet(signal_dim=GAMEPLAY_SIGNAL_DIM, hidden_dim=32, embedding=embedding)
    train_belief_net(
        model,
        deck_priors=train_data[0],
        revealed_counts=train_data[1],
        gameplay_signals=train_data[2],
        opp_hand_sizes=train_data[3],
        opp_library_sizes=train_data[4],
        target_deck_copies=train_data[5],
        target_hand_copies=train_data[6],
        steps=60,
        lr=1e-2,
    )
    with torch.no_grad():
        pred_deck, pred_hand = model(val_priors, val_rev, val_sig, val_hand, val_lib)
    learned = evaluate_belief_log_loss(pred_deck, pred_hand, target_deck=val_tdeck, target_hand=val_thand)

    assert learned["deck_log_loss"] < baseline["deck_log_loss"] * 0.8
    assert learned["hand_log_loss"] < baseline["hand_log_loss"] * 0.8
    assert learned["total_log_loss"] < baseline["total_log_loss"] * 0.8


def test_matchup_predictor_and_scorer_contract_and_v0_mulligan_helper() -> None:
    embedding = _card_table()
    predictor = MatchupPredictor(embedding, width=16, layers=1, heads=2)
    decks = _sample_meta_decks()
    prior = fit_format_prior("TEST", CARD_NAMES, decks)

    logits = predictor(decks, prior.mean_copies)
    equity = predictor.predict_equity(decks, prior.mean_copies)
    assert logits.shape == (decks.shape[0],)
    assert equity.shape == (decks.shape[0],)
    assert bool(((equity >= 0.0) & (equity <= 1.0)).all())
    assert len(own_parameters(predictor, embedding)) == len(list(predictor.parameters()))

    scorer: Scorer = partial(score_deck_vs_field, format_prior=prior, predictor=predictor)
    scored = scorer(decks)
    assert torch.allclose(scored, logits, atol=1e-6)

    keep_cmp = evaluate_opening_hand_vs_v0(0.58, v0_equity=0.51, mulligans_taken=0)
    assert keep_cmp["recommendation"] == "keep"
    assert keep_cmp["keep_v"] == pytest.approx(0.58)
    assert keep_cmp["mulligan_v0"] == pytest.approx(0.51)
    assert keep_cmp["delta"] == pytest.approx(0.07)

    mull_cmp = evaluate_opening_hand_vs_v0(0.41, v0_equity=0.52, mulligans_taken=1)
    assert mull_cmp["recommendation"] == "mulligan"
    assert mull_cmp["delta"] == pytest.approx(-0.11)


def test_edge_cases_empty_library_and_poolless_prior_and_set_transformer_backend() -> None:
    embedding = _card_table()
    decks = _sample_meta_decks()
    prior_with_pool = fit_format_prior("TEST", CARD_NAMES, decks)
    poolless = fit_format_prior("TEST-NOPOOL", CARD_NAMES, decks)
    poolless = type(poolless)(
        format_id=poolless.format_id,
        card_names=poolless.card_names,
        mean_copies=poolless.mean_copies,
        cond_copies=poolless.cond_copies,
        booster_fallback=poolless.booster_fallback,
        deck_pool=None,
        copy_limits=poolless.copy_limits,
    )
    revealed = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    fallback_post = BayesianDeckPoolPrior().posterior_deck(revealed, poolless)
    assert torch.isfinite(fallback_post).all()
    assert float(fallback_post[3].item()) > float(fallback_post[4].item())

    # BeliefNet works without CardVectors and when opp_library_size == 0 (hand_probs == deck_copies).
    model_no_emb = BeliefNet(signal_dim=GAMEPLAY_SIGNAL_DIM, hidden_dim=16, embedding=None)
    state_empty_lib = infer_belief_state(
        model_no_emb,
        prior=prior_with_pool,
        revealed_counts=revealed,
        gameplay_signals=torch.zeros(GAMEPLAY_SIGNAL_DIM),
        opp_hand_size=6,
        opp_library_size=0,
        prior_backend=SetTransformerPrior(embedding, width=16, layers=1, heads=2),
        matchup_equity=0.55,
    )
    assert state_empty_lib.prior_backend == "set_transformer"
    assert state_empty_lib.matchup_equity == pytest.approx(0.55)
    assert torch.allclose(state_empty_lib.hand_probs, state_empty_lib.deck_copies, atol=1e-4)
    assert float(state_empty_lib.hand_probs.sum().item()) == pytest.approx(6.0, abs=1e-4)


def test_belief_net_masks_copy_limit_exhausted_cards_and_roundtrips_checkpoint(tmp_path: Path) -> None:
    copy_limits = torch.tensor([20.0, 20.0, 4.0, 4.0, 4.0, 4.0, 4.0, 4.0])
    prior = fit_format_prior("TEST", CARD_NAMES, _sample_meta_decks(), copy_limits=copy_limits)
    model = BeliefNet(signal_dim=GAMEPLAY_SIGNAL_DIM, hidden_dim=24, embedding=None)
    # All 4 copies of Lightning Bolt (idx 2) are already revealed -> must receive exact 0.0 mass.
    revealed = torch.tensor([1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    state = infer_belief_state(
        model,
        prior=prior,
        revealed_counts=revealed,
        gameplay_signals=torch.zeros(GAMEPLAY_SIGNAL_DIM),
        opp_hand_size=4,
        opp_library_size=16,
    )
    assert float(state.deck_copies[2].item()) == 0.0
    assert float(state.hand_probs[2].item()) == 0.0
    assert float(state.deck_copies.sum().item()) == pytest.approx(20.0, abs=1e-3)

    ckpt_path = tmp_path / "belief_net.pt"
    save_belief_net(ckpt_path, model)
    reloaded = load_belief_net(ckpt_path)
    assert reloaded.hidden_dim == 24
    assert reloaded.signal_dim == GAMEPLAY_SIGNAL_DIM
    assert reloaded.embedding is None

    emb_model = BeliefNet(signal_dim=GAMEPLAY_SIGNAL_DIM, hidden_dim=24, embedding=_card_table())
    emb_path = tmp_path / "belief_net_emb.pt"
    save_belief_net(emb_path, emb_model)
    reloaded_emb = load_belief_net(emb_path)
    assert isinstance(reloaded_emb.embedding, CardTable)
    assert reloaded_emb.embedding.width == 8
    emb_state_a = infer_belief_state(
        emb_model,
        prior=prior,
        revealed_counts=revealed,
        gameplay_signals=torch.zeros(GAMEPLAY_SIGNAL_DIM),
        opp_hand_size=4,
        opp_library_size=16,
    )
    emb_state_b = infer_belief_state(
        reloaded_emb,
        prior=prior,
        revealed_counts=revealed,
        gameplay_signals=torch.zeros(GAMEPLAY_SIGNAL_DIM),
        opp_hand_size=4,
        opp_library_size=16,
    )
    assert torch.allclose(emb_state_a.hand_probs, emb_state_b.hand_probs, atol=1e-6)


def test_matchup_predictor_handles_all_zero_rows_gracefully() -> None:
    embedding = _card_table()
    predictor = MatchupPredictor(embedding, width=16, layers=1, heads=2)
    decks = torch.stack((_sample_meta_decks()[0], torch.zeros(len(CARD_NAMES))))
    opp = torch.zeros(len(CARD_NAMES))
    logits = predictor(decks, opp)
    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()

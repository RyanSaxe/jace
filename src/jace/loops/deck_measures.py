"""How well the deck score and the deck builder fit human decks, as plain functions.

`pretrain_decks` measures its two models with these while they train. The
deck score is measured by `score_fit`: the log loss of the games each deck
won and lost, and the correlation of its predicted and actual win rates.
The builder is measured three ways: `build_log_loss`, the negative
log-probability of the human builds under its sampling, `deck_overlap`,
how much of each human deck the builder's own deck holds, and
`land_total_match`, how often its deck has the human deck's land total.
Only human decks that `allowed_by` accepts teach or test the builder.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from jace.data import HumanDecks
from jace.decks import BuildRules, Builds, DeckBuilder, Scorer, build_log_probs
from jace.train import score_loss

__all__ = (
    "GAMES_FOR_A_RATE",
    "ScoreFit",
    "allowed_by",
    "build_log_loss",
    "deck_overlap",
    "land_total_match",
    "score_fit",
)

# A deck's actual win rate is compared with its predicted one only when it played this many games.
GAMES_FOR_A_RATE = 5
# Decks scored in one call of a model.
_CHUNK = 1024
# Keeps the correlation finite when every prediction is the same.
_SMALLEST_CORRELATION_SCALE = 1e-12


@dataclass(frozen=True, slots=True)
class ScoreFit:
    """How well a scorer predicts the results of `decks` decks.

    `log_loss` is the binomial log loss for each game the decks played, in
    nats, so a scorer that says 0.5 for every deck gets 0.693. `correlation`
    is Pearson's, between the predicted and the actual win rate, over the
    decks that played at least five games. It is zero when every prediction
    is the same.
    """

    log_loss: float
    correlation: float
    decks: int


def allowed_by(rules: BuildRules, decks: HumanDecks) -> Tensor:
    # Returns bool [drafts]: the human decks with exactly `rules.size` cards and a land count the rules allow.
    counts = decks.decks.long()
    lands = counts[:, [*rules.basics, *sorted(rules.nonbasic_lands)]].sum(dim=1)
    return (counts.sum(dim=1) == rules.size) & (lands >= rules.min_lands) & (lands <= rules.max_lands)


def score_fit(score: Scorer, decks: HumanDecks, *, rows: Tensor, device: torch.device) -> ScoreFit:
    # rows: int64 [n], the decks to score. `score` returns the logit of winning a game. No gradients are kept.
    logits: list[Tensor] = []
    with torch.no_grad():
        for start in range(0, len(rows), _CHUNK):
            chosen = rows[start : start + _CHUNK]
            logits.append(score(decks.decks[chosen].to(device).float()).cpu())
    predicted = torch.cat(logits).double()  # [n]
    wins = decks.wins[rows].double()
    games = wins + decks.losses[rows].double()
    log_loss = score_loss(predicted, wins=wins, games=games)
    enough = games >= GAMES_FOR_A_RATE
    predicted_centred = predicted.sigmoid()[enough] - predicted.sigmoid()[enough].mean()
    actual_centred = (wins / games.clamp_min(1.0))[enough] - (wins / games.clamp_min(1.0))[enough].mean()
    # Pearson's correlation: the sum of products of the centred values over this scale.
    scale = (predicted_centred.pow(2).sum() * actual_centred.pow(2).sum()).sqrt().clamp_min(_SMALLEST_CORRELATION_SCALE)
    return ScoreFit(
        log_loss=float(log_loss), correlation=float((predicted_centred * actual_centred).sum() / scale), decks=len(rows)
    )


def build_log_loss(builder: DeckBuilder, builds: Builds, *, rules: BuildRules, spells_per_round: int) -> float:
    # The mean negative log-probability of `builds` under the builder's sampling at temperature 1, in nats a deck.
    # The order of a build's cards is part of it, so score the same builds to compare two measurements.
    surprise = 0.0
    with torch.no_grad():
        for start in range(0, len(builds.pools), _CHUNK):
            part = builds.rows(torch.arange(start, min(start + _CHUNK, len(builds.pools)), device=builds.pools.device))
            log_probs, _ = build_log_probs(
                builder, part, rules=rules, spells_per_round=spells_per_round, temperature=1.0
            )
            surprise -= float(log_probs.sum())
    return surprise / len(builds.pools)


def deck_overlap(built: Tensor, human: Tensor, *, rules: BuildRules) -> float:
    """Measure how much of each human deck the deck built from the same pool holds.

    `built` and `human` are float `[n, cards]`, row by row a written deck
    and the human deck of the same pool. A row's share is the number of the
    human deck's non-basic cards that the written deck also holds, counting
    copies, over the human deck's non-basic cards. Returns the mean share.
    Basics are left out because they are free to add, so matching them
    says little.
    """
    not_basic = torch.ones(human.shape[1], dtype=torch.bool, device=human.device)
    not_basic[list(rules.basics)] = False
    shared = torch.minimum(built, human)[:, not_basic].sum(dim=1)
    return float((shared / human[:, not_basic].sum(dim=1)).mean())


def land_total_match(built: Tensor, human: Tensor, *, rules: BuildRules) -> float:
    # built, human: float [n, cards], as in `deck_overlap`. The share of rows whose two decks hold as many lands.
    lands = [*rules.basics, *sorted(rules.nonbasic_lands)]
    return float((built[:, lands].sum(dim=1) == human[:, lands].sum(dim=1)).float().mean())

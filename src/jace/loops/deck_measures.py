"""How well the deck score and the deck builder fit human decks, as plain functions.

`pretrain_decks` measures its two models with these functions while they
train. `score_fit` measures the deck score by the log loss of the games
each deck won and lost, and by the correlation between each deck's
predicted and actual win rates.

The builder is measured three ways. `build_log_loss` is the negative
log-probability of the human builds under the builder's sampling, so it is
lower when the builder is more likely to write them. `deck_overlap` is how
much of each human deck the builder's own deck for the same pool contains.
`land_total_match` is how often the builder's deck has as many lands as
the human deck. The builder trains and is tested only on the human decks
that `allowed_by` accepts.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from jace.data import HumanDecks
from jace.decks import BuildRules, Builds, DeckBuilder, Scorer, build_log_probs
from jace.loops.supervised import score_loss

__all__ = (
    "GAMES_FOR_A_RATE",
    "ScoreFit",
    "allowed_by",
    "build_log_loss",
    "deck_overlap",
    "land_total_match",
    "score_fit",
)

# A deck's actual win rate is compared with its predicted one only when it played at least this many games.
GAMES_FOR_A_RATE = 5
# The number of decks a model scores in one call.
_CHUNK = 1024
# The correlation divides by at least this, so it stays finite when every prediction is the same.
_SMALLEST_CORRELATION_SCALE = 1e-12


@dataclass(frozen=True, slots=True)
class ScoreFit:
    """How well a scorer predicts the results of `decks` decks.

    `log_loss` is the binomial log loss per game the decks played, in nats.
    A scorer that predicts 0.5 for every deck gets 0.693.

    `correlation` is Pearson's correlation between the predicted and the
    actual win rates of the decks that played at least five games. It is
    zero when every prediction is the same, and when no deck played five
    games.
    """

    log_loss: float
    correlation: float
    decks: int


def allowed_by(rules: BuildRules, decks: HumanDecks) -> Tensor:
    # Returns bool [drafts], true for each human deck that has exactly `rules.size` cards and from
    # `rules.min_lands` to `rules.max_lands` lands.
    counts = decks.decks.long()
    lands = counts[:, [*rules.basics, *sorted(rules.nonbasic_lands)]].sum(dim=1)
    return (counts.sum(dim=1) == rules.size) & (lands >= rules.min_lands) & (lands <= rules.max_lands)


def score_fit(score: Scorer, decks: HumanDecks, *, rows: Tensor, device: torch.device) -> ScoreFit:
    # `rows` is int64 [n], the positions in `decks` of the decks to score, and `score` returns the logit of
    # winning a game. No gradients are kept.
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
    # The denominator of Pearson's correlation.
    scale = (predicted_centred.pow(2).sum() * actual_centred.pow(2).sum()).sqrt().clamp_min(_SMALLEST_CORRELATION_SCALE)
    return ScoreFit(
        log_loss=float(log_loss), correlation=float((predicted_centred * actual_centred).sum() / scale), decks=len(rows)
    )


def build_log_loss(builder: DeckBuilder, builds: Builds, *, rules: BuildRules, spells_per_round: int) -> float:
    # Returns the mean negative log-probability of `builds` under the builder's sampling at temperature 1, in nats
    # per deck. A build's log-probability depends on the order of its cards, so two measurements can be compared
    # only on the same builds.
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
    """Measure how much of each human deck the builder's deck for the same pool contains, as a mean share.

    `built` and `human` are float `[n, cards]` copy counts. Row i of
    `built` is a deck the builder wrote from the pool of the human deck in
    row i of `human`.

    A row's share is the fraction of the human deck's non-basic cards,
    copies included, that the built deck also contains. Basic lands are
    left out because any deck can add them without drafting them, so
    sharing them is little sign that the builder chose as the human did.
    """
    not_basic = torch.ones(human.shape[1], dtype=torch.bool, device=human.device)
    not_basic[list(rules.basics)] = False
    shared = torch.minimum(built, human)[:, not_basic].sum(dim=1)
    return float((shared / human[:, not_basic].sum(dim=1)).mean())


def land_total_match(built: Tensor, human: Tensor, *, rules: BuildRules) -> float:
    # `built` and `human` are as in `deck_overlap`. Returns the share of rows whose two decks have the same number
    # of lands.
    lands = [*rules.basics, *sorted(rules.nonbasic_lands)]
    return float((built[:, lands].sum(dim=1) == human[:, lands].sum(dim=1)).float().mean())

"""Train a learning copy of the deck score on self-play, and let the search read it once it has earned that.

The deck score trained on human decks predicts how humans' decks did in
humans' hands. Self-play decks win in the hands of the current player, so a
copy of the score learns from the decks the loop plays, from the first
iteration, while the human-trained score stays frozen. Before the learning
copy trains on an iteration's decks, both copies predict those decks'
records, and that held-out log loss is the test. Every self-play game has
one winner, so an iteration's decks win half their games on average, while
the human decks the frozen score was fit to won more. Each copy is tested
with its mean logit over the decks moved to zero, so the test compares how
the two rank decks, which is what the search asks of them, and a copy
cannot earn the switch by learning that average alone. The deck search reads the
human-trained score until the learning copy's mean test loss over the last
`switch_iterations` iterations is lower by `switch_margin`, and then reads
the learning copy for good. The switch is one-way so the search cannot
flip between two rankings from one noisy iteration to the next.

The learning copy trains on the decks of the last `window` iterations with
a pull toward the human-trained score: the KL divergence between their
Bernoulli predictions on the same decks, so it keeps what the human decks
taught wherever self-play has said little.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor
from torch.optim import Optimizer

from jace.decks import DeckScore, Scorer
from jace.train import Metrics, apply, kl_to, score_loss

__all__ = ("FIRST_CHECK", "PlayedDecks", "ScoreCheck", "ScoreRule", "check_scores", "train_learning_score")


@dataclass(frozen=True, slots=True)
class ScoreRule:
    """The dials of the learning deck score and of the switch to it.

    The learning copy trains on the decks of the last `window` iterations,
    `passes` gradient steps an iteration over all of them, and `pull`
    weighs its KL divergence from the human-trained score. The search
    switches to it once its mean test loss over the last
    `switch_iterations` iterations is below the human-trained score's by
    at least `switch_margin` nats per game.
    """

    window: int
    passes: int
    pull: float
    switch_iterations: int
    switch_margin: float


@dataclass(frozen=True, slots=True)
class PlayedDecks:
    # The decks one iteration played: decks float [m, cards], wins and games float [m], every deck with a game.
    decks: Tensor
    wins: Tensor
    games: Tensor


@dataclass(frozen=True, slots=True)
class ScoreCheck:
    """What the deck-score check carries from one iteration to the next.

    `recent` holds the played decks of the last iterations, oldest first,
    which the learning copy trains on. `human_losses` and `learning_losses`
    are the two copies' test losses over the last iterations, oldest
    first. `reads_learning` is whether the search reads the learning copy:
    it starts false, turns true once the learning copy has earned it, and
    never turns back.
    """

    recent: tuple[PlayedDecks, ...]
    human_losses: tuple[float, ...]
    learning_losses: tuple[float, ...]
    reads_learning: bool


# A run starts with nothing played, and the search reads the human-trained score.
FIRST_CHECK = ScoreCheck(recent=(), human_losses=(), learning_losses=(), reads_learning=False)


def _centered(logits: Tensor, *, games: Tensor) -> Tensor:
    # The logits less their mean over the decks, each deck weighed by its games.
    return logits - (logits * games).sum() / games.sum()


def check_scores(
    check: ScoreCheck, *, played: PlayedDecks, human: Scorer, learning: Scorer, rule: ScoreRule
) -> tuple[ScoreCheck, Metrics]:
    """Test both deck scores on an iteration's decks and return the check after it, with the losses.

    Call it before the learning copy trains on `played`, so the test is on
    decks it has not seen. `human` and `learning` return the logit of
    winning a game, and each is tested with its mean logit over the decks
    moved to zero. The new check holds `played` in its window and the two
    losses in its history, and reads the learning copy if it already did or
    if the learning copy's mean loss over the last `rule.switch_iterations`
    iterations is below the human-trained one's by `rule.switch_margin`.
    An iteration whose decks played no game leaves the check as it was.
    """
    if len(played.games) == 0:
        nothing: dict[str, float] = {}
        return check, nothing
    with torch.no_grad():
        human_logits = _centered(human(played.decks), games=played.games)
        learning_logits = _centered(learning(played.decks), games=played.games)
        human_loss = float(score_loss(human_logits, wins=played.wins, games=played.games))
        learning_loss = float(score_loss(learning_logits, wins=played.wins, games=played.games))
    human_losses = (*check.human_losses, human_loss)[-rule.switch_iterations :]
    learning_losses = (*check.learning_losses, learning_loss)[-rule.switch_iterations :]
    gain = (sum(human_losses) - sum(learning_losses)) / len(human_losses)
    earned = len(human_losses) == rule.switch_iterations and gain >= rule.switch_margin
    after = ScoreCheck(
        recent=(*check.recent, played)[-rule.window :],
        human_losses=human_losses,
        learning_losses=learning_losses,
        reads_learning=check.reads_learning or earned,
    )
    return after, {
        "deck_score/human_test_loss": human_loss,
        "deck_score/learning_test_loss": learning_loss,
        "deck_score/reads_learning": float(after.reads_learning),
    }


def train_learning_score(
    learning: DeckScore,
    human: DeckScore,
    *,
    recent: Sequence[PlayedDecks],
    rule: ScoreRule,
    optimizer: Optimizer,
    max_grad_norm: float,
) -> Metrics:
    """Take `rule.passes` steps of the learning copy on the recent decks, pulled toward the human-trained score.

    Each step's loss is the binomial log loss of every recent deck's record
    plus `rule.pull` times the mean KL divergence from the learning copy's
    Bernoulli prediction for each deck to the human-trained one's. Returns
    the last step's two parts. `human` is not changed.
    """
    if not recent:
        nothing: dict[str, float] = {}
        return nothing
    decks = torch.cat([played.decks for played in recent])
    wins = torch.cat([played.wins for played in recent])
    games = torch.cat([played.games for played in recent])
    with torch.no_grad():
        human_logits = human(decks)
    # A logit beside a zero is a two-way softmax whose first odds are sigmoid(logit), so `kl_to` between two such
    # pairs is the KL divergence between the two Bernoulli predictions.
    zeros = torch.zeros_like(human_logits)
    human_odds = torch.stack((human_logits, zeros), dim=-1)
    fit = pull = torch.zeros(())
    for _ in range(rule.passes):
        logits = learning(decks)
        fit = score_loss(logits, wins=wins, games=games)
        pull = kl_to(torch.stack((logits, zeros), dim=-1), human_odds).mean()
        apply(fit + rule.pull * pull, [optimizer], max_grad_norm=max_grad_norm)
    return {"deck_score/loss": float(fit.detach()), "deck_score/pull": float(pull.detach())}

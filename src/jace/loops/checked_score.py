"""Train a copy of the deck score on self-play, and switch the deck search to it once it predicts better.

The deck score predicts a deck's chance of winning a game, and the deck
search keeps the builds that the score ranks highest. The score that
pretraining fit to human decks predicts how those decks did in human
hands. Self-play decks win or lose in the hands of the current player
instead, so the loop also trains a copy of the score, the learning copy,
on the decks it plays. The learning copy trains from the first iteration,
and the human-trained score stays frozen.

The loop tests both scores on an iteration's decks before the learning
copy trains on them, so the learning copy is tested on decks it has not
seen. A score's test loss is the log loss of its predictions against each
deck's record of wins and losses. Self-play decks play only each other, so
an iteration's decks win half their games on average, while the human
decks that the frozen score was fit to won more often. Each score's logits
are therefore shifted to a mean of zero over the decks before the test.
The test then compares how the two scores rank the decks, which is all the
search uses. With the mean removed, the learning copy cannot earn the
switch by learning the average win rate alone.

The search reads the human-trained score until the learning copy's mean
test loss over the last `switch_iterations` iterations is lower by at
least `switch_margin`. From then on it reads the learning copy. The switch
is one-way so that one noisy iteration cannot flip the search between two
rankings.

The learning copy trains on the decks of the last `window` iterations. Its
loss adds a pull toward the human-trained score: the KL divergence between
the two scores' Bernoulli predictions for the same decks. The fit to the
records weighs each deck by its games, while the pull weighs every deck
the same. On decks with few games, the learning copy therefore stays close
to the human-trained score.
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
    """The settings of the learning deck score and of the switch to it, fixed for a run.

    The learning copy trains on the decks of the last `window` iterations.
    It takes `passes` gradient steps an iteration, each over all of those
    decks, and `pull` weighs its KL divergence from the human-trained score.

    The search switches to the learning copy once the learning copy's mean
    test loss over the last `switch_iterations` iterations is below the
    human-trained score's by at least `switch_margin` nats per game.
    """

    window: int
    passes: int
    pull: float
    switch_iterations: int
    switch_margin: float


@dataclass(frozen=True, slots=True)
class PlayedDecks:
    # The decks played in one iteration, each with at least one game. `decks` is float [m, cards], each deck's
    # copies of each card, and `wins` and `games` are float [m], each deck's record.
    decks: Tensor
    wins: Tensor
    games: Tensor


@dataclass(frozen=True, slots=True)
class ScoreCheck:
    """What the deck-score check passes from one iteration to the next.

    `recent` contains the played decks of the last `ScoreRule.window`
    iterations, oldest first, and the learning copy trains on them.
    `human_losses` and `learning_losses` are the two scores' test losses
    over the last `ScoreRule.switch_iterations` iterations, oldest first.

    `reads_learning` is whether the deck search reads the learning copy. It
    starts false and turns true once the learning copy earns the switch. It
    never turns back.
    """

    recent: tuple[PlayedDecks, ...]
    human_losses: tuple[float, ...]
    learning_losses: tuple[float, ...]
    reads_learning: bool


# A run starts with nothing played, and the search reads the human-trained score.
FIRST_CHECK = ScoreCheck(recent=(), human_losses=(), learning_losses=(), reads_learning=False)


def _centered(logits: Tensor, *, games: Tensor) -> Tensor:
    # Each deck is weighed by its games, as `score_loss` weighs it, so the logits average zero per game.
    return logits - (logits * games).sum() / games.sum()


def check_scores(
    check: ScoreCheck, *, played: PlayedDecks, human: Scorer, learning: Scorer, rule: ScoreRule
) -> tuple[ScoreCheck, Metrics]:
    """Test both deck scores on an iteration's decks, and return the next check with the iteration's metrics.

    Call it before the learning copy trains on `played`, so that the
    learning copy is tested on decks it has not seen. `human` and
    `learning` each return the logit of winning a game. Each score's logits
    are shifted to a mean of zero over the decks before the test.

    The new check adds `played` to its window and the two test losses to
    its history. If `check.reads_learning` is true, so is the new check's.
    Otherwise it turns true when the history has losses from
    `rule.switch_iterations` iterations and the learning copy's mean loss
    over them is below the human-trained score's by at least
    `rule.switch_margin`.

    The metrics are the two test losses and the new `reads_learning`. When
    `played` contains no deck, it returns `check` unchanged and no metrics.
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

    Every step fits the records of all the decks in `recent` at once, and
    `rule.pull` weighs the pull against that fit, as the module docstring
    describes. `human` is only read, never changed.

    Returns the fit and the pull of the last step. With no recent decks, it
    takes no step and returns no metrics.
    """
    if not recent:
        nothing: dict[str, float] = {}
        return nothing
    decks = torch.cat([played.decks for played in recent])
    wins = torch.cat([played.wins for played in recent])
    games = torch.cat([played.games for played in recent])
    with torch.no_grad():
        human_logits = human(decks)
    # `kl_to` compares softmax odds, so each logit is paired with a zero: the softmax of (logit, 0) gives the first
    # choice the chance sigmoid(logit). `kl_to` on these pairs is therefore the KL divergence between the two
    # Bernoulli predictions.
    zeros = torch.zeros_like(human_logits)
    human_odds = torch.stack((human_logits, zeros), dim=-1)
    fit = pull = torch.zeros(())
    for _ in range(rule.passes):
        logits = learning(decks)
        fit = score_loss(logits, wins=wins, games=games)
        pull = kl_to(torch.stack((logits, zeros), dim=-1), human_odds).mean()
        apply(fit + rule.pull * pull, [optimizer], max_grad_norm=max_grad_norm)
    return {"deck_score/loss": float(fit.detach()), "deck_score/pull": float(pull.detach())}

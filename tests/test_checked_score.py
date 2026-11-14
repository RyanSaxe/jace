"""Check when the deck search switches to the learning deck score, through `check_scores`.

The two scores here are plain functions: the human-trained one predicts an
even chance for every deck, and the learning one predicts the chances it is
given. When those are the decks' true chances, the learning one's test loss
is lower by an amount the test controls through how far the chances are
from even. Each iteration's records are drawn at the true chances with a
fixed seed.
"""

import torch
from torch import Tensor

from jace.loops import FIRST_CHECK, PlayedDecks, ScoreCheck, ScoreRule, check_scores

__all__ = ()

_DECKS = 64
_GAMES = 200.0
_RULE = ScoreRule(window=2, passes=1, pull=1.0, switch_iterations=3, switch_margin=0.01)


def _iterations(chances: Tensor, *, believed: Tensor, count: int, check: ScoreCheck) -> list[ScoreCheck]:
    # Runs `count` iterations of the check on decks that win at `chances`, scored by an even guess and by a
    # learning score that predicts `believed`.
    generator = torch.Generator().manual_seed(0)
    decks = torch.eye(_DECKS)
    checks: list[ScoreCheck] = []
    for _ in range(count):
        wins = torch.binomial(torch.full((_DECKS,), _GAMES), chances, generator=generator)
        played = PlayedDecks(decks=decks, wins=wins, games=torch.full((_DECKS,), _GAMES))
        check, _ = check_scores(
            check,
            played=played,
            human=lambda decks: torch.zeros(len(decks)),
            learning=lambda _decks: believed.logit(),
            rule=_RULE,
        )
        checks.append(check)
    return checks


def test_the_search_switches_only_once_the_learning_score_is_better_by_the_margin_over_enough_iterations() -> None:
    # Chances 0.5 +- 0.25 make the truth beat an even guess by about 0.13 nats a game; +- 0.05 by about 0.005.
    spread = torch.arange(_DECKS) % 2 * 2 - 1.0
    clear, slight = 0.5 + 0.25 * spread, 0.5 + 0.05 * spread
    clearly_better = _iterations(clear, believed=clear, count=4, check=FIRST_CHECK)
    barely_better = _iterations(slight, believed=slight, count=4, check=FIRST_CHECK)

    assert [check.reads_learning for check in clearly_better] == [False, False, True, True]
    assert not any(check.reads_learning for check in barely_better)
    assert len(clearly_better[-1].recent) == _RULE.window


def test_the_switch_is_one_way() -> None:
    clear = 0.5 + 0.25 * (torch.arange(_DECKS) % 2 * 2 - 1.0)
    switched = _iterations(clear, believed=clear, count=3, check=FIRST_CHECK)[-1]
    # Every deck now wins half its games, so the learning score's confident guesses cost it.
    after = _iterations(torch.full((_DECKS,), 0.5), believed=clear, count=4, check=switched)

    assert switched.reads_learning
    assert all(check.reads_learning for check in after)
    assert after[-1].human_losses[-1] < after[-1].learning_losses[-1]

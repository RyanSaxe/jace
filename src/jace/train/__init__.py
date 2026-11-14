"""The pieces a training loop is written from.

`play_games` plays a list of games and returns their decisions, the loss functions score
a model's output, and `apply` takes the gradient step. A loop is a short
function that calls these in the order it wants, so nothing here decides
what is trained when, where decks come from, or how a dial changes over
training.
"""

from jace.train.collect import (
    Collected,
    GameResult,
    NoGamesError,
    PlayGames,
    Rollout,
    StartBatch,
    join,
    play_games,
    select,
)
from jace.train.losses import (
    PlayerRule,
    choice_loss,
    entropy,
    guess_loss,
    kl_to,
    player_loss,
    score_loss,
    win_chance_loss,
)
from jace.train.sequences import Sequences, advantages, earlier_guesses, sequences
from jace.train.update import Metrics, apply, update_guess, update_player

__all__ = (
    "Collected",
    "GameResult",
    "Metrics",
    "NoGamesError",
    "PlayGames",
    "PlayerRule",
    "Rollout",
    "Sequences",
    "StartBatch",
    "advantages",
    "apply",
    "choice_loss",
    "earlier_guesses",
    "entropy",
    "guess_loss",
    "join",
    "kl_to",
    "play_games",
    "player_loss",
    "score_loss",
    "select",
    "sequences",
    "update_guess",
    "update_player",
    "win_chance_loss",
)

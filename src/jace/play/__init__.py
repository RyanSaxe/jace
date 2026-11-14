"""Models that play a game, and the types they are called through.

The guess model predicts the opponent's deck and hand, and the player model
chooses actions with that guess as input. Code that runs games depends only
on the `Policy` and `Guesser` function types, never on a model class.
"""

from jace.play.device import guesser_on, policy_on
from jace.play.guess import FieldNotSetError, GuessModel, TooManyCopiesError
from jace.play.player import PlayerModel, as_boards, random_policy
from jace.play.tokens import BOARD_ARRAYS
from jace.play.types import Boards, Earlier, Guess, Guesser, Policy

__all__ = (
    "BOARD_ARRAYS",
    "Boards",
    "Earlier",
    "FieldNotSetError",
    "Guess",
    "GuessModel",
    "Guesser",
    "PlayerModel",
    "Policy",
    "TooManyCopiesError",
    "as_boards",
    "guesser_on",
    "policy_on",
    "random_policy",
)

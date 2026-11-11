"""Models that play a game, and the types they are called through.

The guess model predicts the opponent's deck and hand, and the player model
chooses actions with that guess as input. Code that runs games depends only
on the `Policy` and `Guesser` function types, never on a model class.
"""

from jace.play.guess import GuessModel
from jace.play.player import PlayerModel, as_boards, random_policy
from jace.play.types import Boards, Earlier, Guess, Guesser, Policy

__all__ = ("Boards", "Earlier", "Guess", "GuessModel", "Guesser", "PlayerModel", "Policy", "as_boards", "random_policy")

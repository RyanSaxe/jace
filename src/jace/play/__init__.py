"""Models that play a game, the model that judges a position for training, and their types.

The guess model predicts the opponent's deck and hand, and the player model
chooses actions with that guess as input. The win-chance model estimates
the acting seat's chance to win from the board and the opponent's true
hidden cards; only training calls it. Code that runs games depends only on
the `Policy` and `Guesser` function types, and training also on the
`WinChance` type, never on a model class.
"""

from jace.play.guess import FieldNotSetError, GuessModel, TooManyCopiesError
from jace.play.player import PlayerModel, as_boards, random_policy
from jace.play.tokens import BOARD_ARRAYS
from jace.play.types import Boards, Earlier, Guess, Guesser, Policy, WinChance
from jace.play.win_chance import WinChanceModel

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
    "WinChance",
    "WinChanceModel",
    "as_boards",
    "random_policy",
)

"""Models that play a game, the model that judges a position for training, and their types.

The guess model predicts the opponent's deck and hand. It reads the board,
the cards the opponent has shown or revealed, and the meta, which describes
what decks of the format look like. The player model chooses actions, and
the guess is one of its inputs. The win-chance model estimates the acting
seat's chance to win from the board and the opponent's true hidden cards.
Only training calls the win-chance model.

Code that runs games depends only on the `Policy` and `Guesser` function
types, and training also depends on the `WinChance` type. None of that code
depends on a model class.
"""

from jace.play.guess import GuessModel, MetaNotSetError
from jace.play.meta import Meta, blend_meta, day_zero_meta, fit_meta
from jace.play.player import PlayerModel, as_boards, random_policy
from jace.play.tokens import BOARD_ARRAYS
from jace.play.types import Boards, Earlier, Guess, Guesser, Policy, WinChance
from jace.play.win_chance import WinChanceModel

__all__ = (
    "BOARD_ARRAYS",
    "Boards",
    "Earlier",
    "Guess",
    "GuessModel",
    "Guesser",
    "Meta",
    "MetaNotSetError",
    "PlayerModel",
    "Policy",
    "WinChance",
    "WinChanceModel",
    "as_boards",
    "blend_meta",
    "day_zero_meta",
    "fit_meta",
    "random_policy",
)

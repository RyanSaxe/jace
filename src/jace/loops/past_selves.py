"""The player's past selves, kept for the workers that play against them.

A player that plays only its current self can chase cycles. It learns to
beat its own latest habits, which can drift away from good play, and it can
forget how to beat a style it no longer plays. In a run that keeps past
selves, a few workers each round play the other seat with a past copy of
the player, so older styles stay among its opponents.
"""

import random
from dataclasses import dataclass, field

from torch import nn

from jace.train.workers import Weights

__all__ = ("PastSelves",)


@dataclass(slots=True)
class PastSelves:
    """Copies of the player from earlier in a run, and the past players each round plays.

    `keep` adds a copy of the player at every `every`-th iteration, and only
    when `count` is above zero. `draw` returns `count` past players, each
    drawn uniformly from the copies kept so far with `rng`, or none before
    the first copy. Each copy's weights are on the CPU under the name
    "past_player", the name `jace.loops.entry_points.play_actor` gives the
    workers' past player.
    """

    rng: random.Random
    every: int
    count: int
    kept: list[Weights] = field(default_factory=list)

    def keep(self, player: nn.Module, *, iteration: int) -> None:
        if self.count > 0 and iteration % self.every == 0:
            state = player.state_dict()
            self.kept.append({"past_player": {key: value.detach().cpu().clone() for key, value in state.items()}})

    def draw(self) -> list[Weights]:
        return [self.rng.choice(self.kept) for _ in range(self.count)] if self.kept else []

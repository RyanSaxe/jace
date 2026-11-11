"""Train a small player on the toy game and print how often it picks its best card.

High Card needs no game engine: each seat holds three cards, picks one, and
the higher pick wins. The best play is always the highest card, so the share
of decisions where the player picks it shows whether training works. The
script uses the same pieces a real loop does: `collect` plays the games and
`update_player` learns from them. Run it with `uv run python -m jace.loops.toy`.
"""

import sys

import torch
from torch import Tensor, nn

from jace.engine.toy import HAND_SIZE, TOY_CARDS, ToyBatch
from jace.modules import run
from jace.play import Boards, Earlier, Guess
from jace.train import PlayerRule, collect, update_player

__all__ = ("main",)

GAMES_AT_ONCE = 64
ITERATIONS = 40
DECISIONS_PER_ITERATION = 512
REPORT_EVERY = 5
HIDDEN_WIDTH = 32
LEARNING_RATE = 3e-3
RULE = PlayerRule(clip=0.2, stay=0.1, spread=0.01, keep_quantile=0.0, keep_floor=0.0, value_weight=0.5)


def main() -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = nn.Sequential(nn.Linear(HAND_SIZE + 1, HIDDEN_WIDTH), nn.Tanh(), nn.Linear(HIDDEN_WIDTH, HAND_SIZE + 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    def policy(boards: Boards, _guess: Guess) -> tuple[Tensor, Tensor]:
        # Returns a score for each of the three cards and the player's chance of winning.
        output = run(model, torch.cat([boards["hand"], boards["seen"]], dim=-1))
        return output[:, :HAND_SIZE], output[:, HAND_SIZE].sigmoid()

    def even_policy(boards: Boards, _guess: Guess) -> tuple[Tensor, Tensor]:
        # The opponent picks each card equally often.
        rows = len(boards["hand"])
        return torch.zeros(rows, HAND_SIZE), torch.full((rows,), 0.5)

    def blank_guesser(boards: Boards, _earlier: Earlier) -> Guess:
        # The toy game has nothing useful to guess, so every card gets the same chance.
        rows = len(boards["hand"])
        even = torch.full((rows, TOY_CARDS), 1 / TOY_CARDS)
        return Guess(deck=even, hand=even)

    batch = ToyBatch(lambda: ((0,), (1,)), games=GAMES_AT_ONCE, seed=0)
    pending = None
    for iteration in range(1, ITERATIONS + 1):
        collected = collect(
            batch,
            pending,
            learner=policy,
            opponent=even_policy,
            guesser=blank_guesser,
            cards=TOY_CARDS,
            history=1,
            decisions=DECISIONS_PER_ITERATION,
            generator=generator,
        )
        pending = collected.pending
        rollout = collected.rollout
        mine = rollout.learner
        best_card_rate = (rollout.boards["hand"][mine].argmax(dim=-1) == rollout.actions[mine]).float().mean()
        win_rate = rollout.results[mine].mean()
        update_player(
            policy,
            rollout,
            rule=RULE,
            lam_advantage=0.5,
            lam_value=0.8,
            slice_rows=256,
            max_grad_norm=1.0,
            optimizers=[optimizer],
        )
        if iteration == 1 or iteration % REPORT_EVERY == 0:
            sys.stdout.write(f"iteration {iteration:2d}  best card {best_card_rate:.2f}  win rate {win_rate:.2f}\n")


if __name__ == "__main__":
    main()

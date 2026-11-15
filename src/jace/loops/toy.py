"""Train a small player on the toy game and print how often it picks its best card.

High Card runs without the game engine. Each seat holds three cards and picks one,
and the higher pick wins. The best play is always the highest card, so if
training works, the share of decisions in which the player picks its
highest card rises. The script uses the same pieces a real loop does.
`play_games` plays a list of games to the end, and `update_player` updates
the player from them, with a win chance that reads the opponent's hand.
Run it with `uv run python -m jace.loops.toy`.
"""

import sys

import torch
from torch import Tensor, nn

from jace.engine.toy import HAND_SIZE, TOY_CARDS, ToyBatch
from jace.play import Boards, Earlier, Guess
from jace.train import PlayerRule, play_games, update_player

__all__ = ("main",)

GAMES_AT_ONCE = 64
ITERATIONS = 40
GAMES_PER_ITERATION = 512
REPORT_EVERY = 5
HIDDEN_WIDTH = 32
LEARNING_RATE = 3e-3
RULE = PlayerRule(clip=0.2, stay=0.1, spread=0.01, keep_quantile=0.0, keep_floor=0.0, normalize=False)


def _even_policy(boards: Boards, _guess: Guess) -> Tensor:
    # The opponent picks each card equally often.
    rows = len(boards["hand"])
    return torch.zeros(rows, HAND_SIZE)


def main() -> None:
    torch.manual_seed(0)
    generator = torch.Generator().manual_seed(0)
    model = nn.Sequential(nn.Linear(HAND_SIZE + 1, HIDDEN_WIDTH), nn.Tanh(), nn.Linear(HIDDEN_WIDTH, HAND_SIZE))
    win_chance_model = nn.Sequential(
        nn.Linear(HAND_SIZE + 1 + TOY_CARDS, HIDDEN_WIDTH), nn.Tanh(), nn.Linear(HIDDEN_WIDTH, 1)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    win_chance_optimizer = torch.optim.Adam(win_chance_model.parameters(), lr=LEARNING_RATE)

    def policy(boards: Boards, _guess: Guess) -> Tensor:
        # Returns a score for each of the three cards.
        view = torch.cat([boards["hand"], boards["seen"]], dim=-1)
        return model(view)

    def win_chance(boards: Boards, truth: Guess) -> Tensor:
        # Reads what the acting seat sees and the opponent's true hand.
        view_and_truth = torch.cat([boards["hand"], boards["seen"], truth.hand], dim=-1)
        return win_chance_model(view_and_truth).squeeze(-1).sigmoid()

    def blank_guesser(boards: Boards, _earlier: Earlier) -> Guess:
        # The toy game has nothing useful to guess, so every card gets the same chance.
        rows = len(boards["hand"])
        even = torch.full((rows, TOY_CARDS), 1 / TOY_CARDS)
        return Guess(deck=even, hand=even)

    matchups = [((0,), (1,))] * GAMES_PER_ITERATION
    for iteration in range(1, ITERATIONS + 1):
        collected = play_games(
            lambda games, seed=iteration: ToyBatch(games, games=GAMES_AT_ONCE, seed=seed),
            matchups,
            learner=policy,
            opponent=_even_policy,
            guesser=blank_guesser,
            cards=TOY_CARDS,
            history=1,
            generator=generator,
        )
        rollout = collected.rollout
        mine = rollout.learner
        best_card_rate = (rollout.boards["hand"][mine].argmax(dim=-1) == rollout.actions[mine]).float().mean()
        win_rate = rollout.results[mine].mean()
        update_player(
            policy,
            rollout,
            win_chance=win_chance,
            rule=RULE,
            lam_advantage=0.5,
            lam_value=0.8,
            slice_rows=256,
            max_grad_norm=1.0,
            policy_optimizers=[optimizer],
            win_chance_optimizers=[win_chance_optimizer],
            passes=1,
            generator=generator,
        )
        if iteration == 1 or iteration % REPORT_EVERY == 0:
            sys.stdout.write(f"iteration {iteration:2d}  best card {best_card_rate:.2f}  win rate {win_rate:.2f}\n")


if __name__ == "__main__":
    main()

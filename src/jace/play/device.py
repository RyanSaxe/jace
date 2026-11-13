"""Run a policy or a guesser on another device without changing its callers.

Games are collected and rollouts are stored on the CPU, where the engine
writes its boards. A model can still live on a GPU: these wrappers move a
call's inputs to the model's device and its outputs back, so `play_games` and
the update functions keep handing over and receiving CPU tensors. Gradients
flow through the move back, so a wrapped policy trains like a bare one.
"""

import torch

from jace.play.types import Boards, Earlier, Guess, Guesser, Policy

__all__ = ("guesser_on", "policy_on")


def policy_on(policy: Policy, device: torch.device) -> Policy:
    def moved(boards: Boards, guess: Guess) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = policy(
            {name: board.to(device) for name, board in boards.items()},
            Guess(deck=guess.deck.to(device), hand=guess.hand.to(device)),
        )
        return logits.cpu(), values.cpu()

    return moved


def guesser_on(guesser: Guesser, device: torch.device) -> Guesser:
    def moved(boards: Boards, earlier: Earlier) -> Guess:
        guesses = Guess(deck=earlier.guesses.deck.to(device), hand=earlier.guesses.hand.to(device))
        guess = guesser(
            {name: board.to(device) for name, board in boards.items()},
            Earlier(guesses=guesses, mask=earlier.mask.to(device)),
        )
        return Guess(deck=guess.deck.cpu(), hand=guess.hand.cpu())

    return moved

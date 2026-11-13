"""The card encoder's inputs: how a card's engine program becomes token ids.

The engine defines each card as a run of (kind, value) pairs. The encoder
reads them as tokens, and the ids below are the ones a pair cannot take: a
row is padded with `PADDING_TOKEN`, a card the engine cannot play has the
single token `NO_PROGRAM_TOKEN`, and a pair the vocabulary lacks becomes
`UNKNOWN_TOKEN`. The vocabulary's pairs take the ids from
`FIRST_PROGRAM_TOKEN` on, in the order the engine's vocabulary numbers them.
The encoder learns one position for each of the first `MAX_PROGRAM_TOKENS`
pairs, and the engine's longest program has 149.
"""

__all__ = ("FIRST_PROGRAM_TOKEN", "MAX_PROGRAM_TOKENS", "NO_PROGRAM_TOKEN", "PADDING_TOKEN", "UNKNOWN_TOKEN")

PADDING_TOKEN = 0
NO_PROGRAM_TOKEN = 1
UNKNOWN_TOKEN = 2
FIRST_PROGRAM_TOKEN = 3
MAX_PROGRAM_TOKENS = 256

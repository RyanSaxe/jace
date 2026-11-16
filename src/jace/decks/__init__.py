"""Build decks from a pool of cards and score them.

A pool is a plain tuple of card ids, so this package never imports the
drafter. The same code can build a limited deck from a draft or a
constructed deck from a collection.

The builder, `DeckBuilder`, samples whole decks in batches and gives each
its log-probability. The deck score, `DeckScore`, predicts a deck's chance
of winning a game. `build_decks` samples many decks for a pool and returns
the distinct ones that the deck score ranks best. The caller chooses which
of them to play, because deciding how to explore decks is the training
loop's job.
"""

from jace.decks.builder import (
    BuildRules,
    Builds,
    BuildScores,
    DeckBuilder,
    InvalidBuildRulesError,
    NoBasicLandsError,
    build_log_probs,
    builds_from_decks,
    join_builds,
    sample_builds,
)
from jace.decks.score import DeckScore, Scorer
from jace.decks.search import build_decks

__all__ = (
    "BuildRules",
    "BuildScores",
    "Builds",
    "DeckBuilder",
    "DeckScore",
    "InvalidBuildRulesError",
    "NoBasicLandsError",
    "Scorer",
    "build_decks",
    "build_log_probs",
    "builds_from_decks",
    "join_builds",
    "sample_builds",
)

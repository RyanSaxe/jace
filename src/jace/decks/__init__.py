"""Build decks from a pool of cards and score them.

A pool arrives as a plain tuple of card ids, so this package builds limited
decks from a draft and constructed decks from a collection with the same
code and never imports the drafter. The builder samples whole decks in
batches and gives each its log-probability, the deck score rates a deck's
chance of winning, and `build_decks` samples many decks for a pool and
returns the distinct ones the score ranks best. The caller picks which ones
to play, because how decks are explored is the training loop's job.
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

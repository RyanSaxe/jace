"""Build decks from a pool of cards and score them.

A pool arrives as a plain tuple of card ids, so this package builds limited
decks from a draft and constructed decks from a collection with the same
code and never imports the drafter. A proposer writes many whole decks, a
swap search improves each one against the deck score, and `build_decks`
returns every distinct result with its score. The caller picks which one to
play, because exploring is the training loop's job.
"""

from jace.decks.builder import (
    BuildRules,
    Candidate,
    DeckBuilderModel,
    InvalidBuildRulesError,
    NoLegalDeckError,
    Proposer,
    build_decks,
    improve,
    model_proposer,
    random_proposer,
)
from jace.decks.score import DeckScore, Scorer

__all__ = (
    "BuildRules",
    "Candidate",
    "DeckBuilderModel",
    "DeckScore",
    "InvalidBuildRulesError",
    "NoLegalDeckError",
    "Proposer",
    "Scorer",
    "build_decks",
    "improve",
    "model_proposer",
    "random_proposer",
)

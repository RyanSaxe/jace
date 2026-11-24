"""Learn to draft, build decks, and play Magic: The Gathering Limited by self-play.

The package separates pure RL algorithms (`jace.rl`), rollout collection (`jace.collect`),
evaluation and decision forensics (`jace.eval`), counterfactual forks (`jace.forks`),
run storage (`jace.storage`), supervised pretraining (`jace.pretrain`), gameplay training
CLIs (`jace.train`), and Limited domain modules (`jace.data`, `jace.cards`, `jace.draft`,
`jace.decks`, `jace.modules`) behind import boundaries enforced by pre-commit and CI.
Nothing is exported from the package root so callers import the subpackage they need directly.
"""

__all__ = ()

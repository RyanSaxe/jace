"""Learn to draft, build decks, and play Magic: The Gathering Limited by self-play.

The package separates pure RL algorithms (`jace.rl`), evaluation and decision
forensics (`jace.eval`), Academy-ready run storage (`jace.runs`), and Limited
domain modules (`jace.data`, `jace.cards`, `jace.draft`, `jace.decks`) behind
import boundaries enforced by pre-commit and CI. Nothing is exported from the
package root so callers import the subpackage they need directly.
"""

__all__ = ()

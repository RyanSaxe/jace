"""Learn to draft, build decks and play Magic limited by self-play.

The package is split into modules that each own one part of the loop, and only
the engine module may import the game engine. Nothing is exported from the
package root, so callers import the module they need by name.
"""

__all__ = ()

"""Learn to play Magic: The Gathering by self-play, with drafting and deck building.

The package is split into modules that each implement one part of the training
loop, and only the `jace.engine` package may import the game engine. Nothing is
exported from the package root, so callers import the module they need by name.
"""

__all__ = ()

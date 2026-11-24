"""Top-level self-play gameplay training and experiment screening CLIs.

Wires `jace.rl`, `jace.collect`, `jace.eval`, `jace.forks`, and `jace.storage.gameplay`
into runnable entry points (`full_data`, `ten_decks`, and `harness`). Nothing is
exported from the package root so `python -m jace.train.<module>` executes without
importing sibling CLIs prematurely.
"""

__all__ = ()

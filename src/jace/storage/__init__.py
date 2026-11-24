"""Run storage subpackage for gameplay and pretraining artifacts.

Separates Academy-native gameplay run folders (`jace.storage.gameplay`) from
file-locked, SHA-256 verified pretraining run records (`jace.storage.pretrain`).
Nothing is imported at package load time so `jace.storage.pretrain` remains a
zero-dependency leaf with no transitive `manabot` or `managym` imports.
"""

__all__ = ()

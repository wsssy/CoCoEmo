"""Backbone adapters.

Import the specific adapter you need (each requires its own environment + the
corresponding TTS repo/checkpoints):

    from cocoemo.backbones import cosyvoice2
    from cocoemo.backbones import indextts2

The adapter modules are intentionally NOT imported here, so that importing
``cocoemo.backbones`` never pulls in a TTS model. ``BACKBONES`` lists the
available adapter module names; ``get_spec`` returns a backbone's constants
without importing its heavy dependencies.
"""

from .base import Backbone, BackboneSpec

BACKBONES = ("cosyvoice2", "indextts2")


def get_spec(name: str) -> BackboneSpec:
    """Return the BackboneSpec for ``name`` by importing only that adapter module."""
    import importlib
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone {name!r}; choose from {BACKBONES}")
    module = importlib.import_module(f"cocoemo.backbones.{name}")
    return module.SPEC


__all__ = ["Backbone", "BackboneSpec", "BACKBONES", "get_spec"]

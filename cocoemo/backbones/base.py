"""Backbone adapter interface.

CosyVoice2 and IndexTTS2 cannot share a Python environment, so each backbone is
isolated behind this small interface. The shared steering / discriminability /
evaluation code only ever touches a ``BackboneSpec`` + the functions an adapter
exposes; it never imports a TTS model directly.

Each adapter module (``cocoemo.backbones.cosyvoice2`` / ``.indextts2``) provides:

  * ``SPEC``            - a BackboneSpec with the paper's per-model constants
  * ``load_model(...)`` - load (and, for CosyVoice2, monkey-patch) the model
  * the backbone-specific entry points re-exported from ``cocoemo.steering._core``:
      get_op_dict, get_head_geometry, extract_audio, inference

Run each adapter in its own environment (see ``env/``).
"""

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable


@dataclass(frozen=True)
class BackboneSpec:
    """Per-model constants used for steering site selection (from the paper)."""

    name: str
    num_layers: int               # transformer layers in the SLM/GPT (24 for both)
    hidden_dim: int               # 896 (CosyVoice2) / 1024 (IndexTTS2)
    steer_layers: List[int]       # selected top-K steering layers
    steer_operator: str           # operator/hook site, e.g. "attn_output"
    stable_alpha_range: tuple = (0.0, 4.5)   # documented intelligibility-safe range


@runtime_checkable
class Backbone(Protocol):
    """Structural interface every backbone adapter satisfies."""

    SPEC: BackboneSpec

    def load_model(self, model_dir: str, **kwargs):
        """Load the TTS model (and patch it, for CosyVoice2). Returns the handle(s)
        the adapter's extraction/inference functions expect."""
        ...

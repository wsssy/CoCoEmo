"""Steering vector construction and inference-time injection (backbone-agnostic).

This package exposes the parts of the steering machinery that do NOT depend on a
specific TTS backbone:

  * steering operators (translation / norm-preserving) and forward hooks
  * mean-difference extraction from cached representations
  * mixed-emotion vector composition (Eq. 7) and injection config (Eq. 8)
  * save / load of steering vectors

Backbone-specific entry points (model loading, audio->activation extraction, and
steered synthesis) live in ``cocoemo.backbones.cosyvoice2`` / ``.indextts2``,
which import the shared helpers from here.

The proven implementation is kept verbatim in ``_core`` (only absolute paths were
stripped); this module curates the public, backbone-agnostic surface.
"""

# --- Backward-compat for shipped steering vectors -------------------------- #
# The released *.pt steering vectors were pickled when the discriminability code
# lived as a top-level module named ``discriminability`` (the HeadGeometry /
# LayerDiscriminabilityMetrics objects in their metadata reference it). Register an
# alias so ``torch.load`` resolves those classes to the packaged module and the
# vectors load without ``ModuleNotFoundError: No module named 'discriminability'``.
import sys as _sys
from cocoemo.discriminability import _probe as _discriminability_probe
_sys.modules.setdefault("discriminability", _discriminability_probe)

# The backbone-agnostic helpers are byte-identical across both backbone cores
# (only the CosyVoice/IndexTTS-specific functions diverged between the forks), so
# the shared public API is re-exported from the CosyVoice2 core as canonical.
from ._core_cosyvoice import (
    # operators
    translation_op_,
    norm_preserve_steer_op_,
    norm_preserve_steer,
    # hooks
    PreHookSave,
    ForwardHookSave,
    PreHookInject,
    ForwardHookInject,
    hook_representations_for_saving,
    hook_model_inject,
    unhook,
    # extraction -> mean difference
    extract_with_hooks,
    extract_pos_neg,
    create_mean_difference,
    # mixed-emotion composition + injection
    combine_steering_vectors,
    create_mixed_steering_vectors,
    create_sample_specific_mixed_vectors,
    prepare_steering_injection_config,
    update_operations_to_hook_info,
    generate_with_hooks,
    logit_extract_with_hooks,
    # persistence
    save_steering_vectors,
    load_steering_vectors,
)

__all__ = [
    "translation_op_",
    "norm_preserve_steer_op_",
    "norm_preserve_steer",
    "PreHookSave",
    "ForwardHookSave",
    "PreHookInject",
    "ForwardHookInject",
    "hook_representations_for_saving",
    "hook_model_inject",
    "unhook",
    "extract_with_hooks",
    "extract_pos_neg",
    "create_mean_difference",
    "combine_steering_vectors",
    "create_mixed_steering_vectors",
    "create_sample_specific_mixed_vectors",
    "prepare_steering_injection_config",
    "update_operations_to_hook_info",
    "generate_with_hooks",
    "logit_extract_with_hooks",
    "save_steering_vectors",
    "load_steering_vectors",
]

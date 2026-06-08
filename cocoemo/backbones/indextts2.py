"""IndexTTS2 backbone adapter.

Thin wrapper over the proven steering code: model loading, the paper's steering
constants, and the IndexTTS2-specific entry points re-exported from the shared
steering core.

Environment: the IndexTTS2 environment (follow official setup, then ``pip install -e .`` CoCoEmo).
Requirements:
  * a local clone of IndexTTS2 (https://github.com/index-tts/index-tts) with the
    ``INDEXTTS_ROOT`` environment variable pointing at it
  * the IndexTTS2 checkpoints directory + config.yaml (pass as ``model_dir`` / ``cfg_path``)

Paper config (Table 4 / Appendix B,C,I): GPT2-style SLM, 24 layers, hidden 1024,
steering at attn_output of layers 6, 8 and 1.
"""

import os
import sys

from .base import BackboneSpec

# IndexTTS2-specific entry points (proven IndexTTS2 steering core, verbatim).
from cocoemo.steering._core_indextts import (
    get_indextts2_op_dict as get_op_dict,
    get_indextts2_head_geometry as get_head_geometry,
    extract_with_hooks_audio_indextts2 as extract_audio,
    hook_representations_for_saving_indextts2,
    inference_indextts2_with_steering as inference,
)


SPEC = BackboneSpec(
    name="indextts2",
    num_layers=24,
    hidden_dim=1024,
    steer_layers=[6, 8, 1],
    steer_operator="attn_output",
    stable_alpha_range=(0.0, 6.0),
)


def load_model(model_dir: str = "checkpoints", cfg_path: str | None = None, **kwargs):
    """Load IndexTTS2.

    Args:
        model_dir: IndexTTS2 checkpoints directory.
        cfg_path: path to ``config.yaml`` (defaults to ``<model_dir>/config.yaml``).
        **kwargs: forwarded to ``IndexTTS2(...)`` (e.g. ``use_fp16``, ``device``).

    Returns:
        The ``IndexTTS2`` instance expected by the extraction/inference functions.
    """
    # Add the IndexTTS2 repo to sys.path if provided, so ``indextts`` is importable.
    indextts_root = os.environ.get("INDEXTTS_ROOT")
    if indextts_root and indextts_root not in sys.path:
        sys.path.insert(0, indextts_root)

    if cfg_path is None:
        cfg_path = os.path.join(model_dir, "config.yaml")

    from indextts.infer_v2 import IndexTTS2

    tts = IndexTTS2(cfg_path=cfg_path, model_dir=model_dir, **kwargs)
    return tts


def generate_steered_speech(model, text, reference_audio_path, steering_vectors,
                            layers, alpha, *, operations=None, output_path=None,
                            emo_vector=None, verbose=False, **infer_kwargs):
    """Uniform steered-synthesis call for the merged pipeline (IndexTTS2).

    Mirrors the proven ``generate_steered_speech_indextts2`` wrapper with a
    uniform signature that matches the CosyVoice2 adapter.

    Args:
        model: IndexTTS2 instance from ``load_model``.
        text: text to synthesize.
        reference_audio_path: neutral reference audio path (used for speaker embedding).
        steering_vectors: dict from ``load_steering_vectors`` (or already-extracted).
        layers: steering layers (e.g. ``SPEC.steer_layers``).
        alpha: scalar steering strength or ``{operation: float}``.
        operations: operators to steer (default ``[SPEC.steer_operator]``).
        output_path: optional wav path to save.
        emo_vector: optional IndexTTS2 native emotion conditioning vector.

    Returns:
        ``{"audio": Tensor or None, "sample_rate": int or None, "output_path": str}``.
    """
    import torchaudio
    from pathlib import Path

    operations = operations or [SPEC.steer_operator]

    # Accept raw checkpoint dict (with 'steering_vectors' key) or pre-extracted.
    if isinstance(steering_vectors, dict) and "steering_vectors" in steering_vectors:
        vector_dict = steering_vectors["steering_vectors"]
        steering_vectors = {op: vector_dict[op] for op in operations if op in vector_dict}

    if isinstance(alpha, (int, float)):
        alpha = {op: alpha for op in operations}

    result = inference(
        tts=model,
        text=text,
        emo_vector=emo_vector,
        spk_audio_prompt=reference_audio_path,
        steering_vectors=steering_vectors,
        operations=operations,
        layers=layers,
        alpha=alpha,
        output_path=output_path,
        verbose=verbose,
        **infer_kwargs,
    )

    audio = None
    sample_rate = None
    if output_path and Path(output_path).exists():
        audio, sample_rate = torchaudio.load(output_path)

    return {
        "audio": audio,
        "sample_rate": sample_rate,
        "output_path": result if result is not None else output_path,
    }

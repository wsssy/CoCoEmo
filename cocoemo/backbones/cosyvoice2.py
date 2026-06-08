"""CosyVoice2 backbone adapter.

Thin wrapper over the proven steering code: model loading + monkey-patch, the
paper's steering constants, and the CosyVoice2-specific entry points re-exported
from the shared steering core.

Environment: the CosyVoice2 conda env (see ``env/cosyvoice2.txt``).
Requirements:
  * a local clone of CosyVoice (https://github.com/FunAudioLLM/CosyVoice) with the
    ``COSYVOICE_ROOT`` environment variable pointing at it
  * the CosyVoice2-0.5B pretrained model directory (pass as ``model_dir``)

Paper config (Table 4 / Sec 4.5): Qwen2-based SLM, 24 layers, hidden 896,
steering at attn_output of layers 17 and 14.
"""

from .base import BackboneSpec

# CosyVoice2-specific entry points (proven CosyVoice2 steering core, verbatim).
from cocoemo.steering._core_cosyvoice import (
    get_cosyvoice_op_dict as get_op_dict,
    get_cosyvoice_head_geometry as get_head_geometry,
    extract_with_hooks_audio as extract_audio,
    extract_pos_neg_audio,
    create_steering_vectors_from_audio_data,
    inference_zero_shot_with_steering,
    inference_sft_with_steering,
    inference_cross_lingual_with_steering,
)
# The monkey-patch that adds token/synthesis helpers onto a CosyVoice2 instance.
from ._cosyvoice_ext import patch_cosyvoice_model


SPEC = BackboneSpec(
    name="cosyvoice2",
    num_layers=24,
    hidden_dim=896,
    steer_layers=[17, 14],
    steer_operator="attn_output",
    stable_alpha_range=(0.0, 6.0),
)


def load_model(model_dir: str, patch: bool = True):
    """Load CosyVoice2 and (by default) apply the steering monkey-patch.

    Args:
        model_dir: path to the CosyVoice2-0.5B pretrained model directory.
        patch: whether to attach the extraction/synthesis helper methods.

    Returns:
        The patched ``CosyVoice2`` instance (its ``.frontend`` is the frontend used
        by the extraction functions).
    """
    # Import is deferred so this module can be imported for SPEC/constants without
    # the CosyVoice repo present. ``COSYVOICE_ROOT`` is added to sys.path by
    # ``_cosyvoice_ext`` on import above.
    from cosyvoice.cli.cosyvoice import CosyVoice2

    cosyvoice = CosyVoice2(model_dir)
    if patch:
        patch_cosyvoice_model(cosyvoice)
    return cosyvoice


# Default steered-synthesis entry point for this backbone (zero-shot from a neutral
# reference, as used throughout the paper).
inference = inference_zero_shot_with_steering


def generate_steered_speech(model, text, reference_audio_path, steering_vectors,
                            layers, alpha, *, operations=None, prompt_text="",
                            output_path=None, speed=1.0, inference_type="zero_shot",
                            verbose=False):
    """Uniform steered-synthesis call for the merged pipeline (CosyVoice2).

    Mirrors the proven ``generate_steered_speech`` wrapper but takes a single
    ``reference_audio_path`` so the pipeline can call every backbone identically.

    Args:
        model: patched CosyVoice2 instance from ``load_model``.
        text: text to synthesize.
        reference_audio_path: neutral reference audio (loaded at 16 kHz).
        steering_vectors: the dict from ``load_steering_vectors`` (or a mixed vector).
        layers: steering layers (e.g. ``SPEC.steer_layers``).
        alpha: scalar steering strength or ``{operation: float}``.
        operations: operators to steer (default ``[SPEC.steer_operator]``).
        prompt_text: transcript of the reference audio (required for zero-shot
            voice cloning alignment in CosyVoice2).
        output_path: optional wav path to save.

    Returns:
        ``{"audio": Tensor, "sample_rate": int}``.
    """
    operations = operations or [SPEC.steer_operator]
    if isinstance(alpha, (int, float)):
        alpha = {op: alpha for op in operations}

    # Accept either a raw checkpoint from load_steering_vectors (which nests the
    # vectors under a "steering_vectors" key alongside metadata) or an already
    # extracted {operation: {layer: tensor}} dict. This mirrors the extraction in
    # apply_steering_tts.py.
    if isinstance(steering_vectors, dict) and "steering_vectors" in steering_vectors:
        vector_dict = steering_vectors["steering_vectors"]
        steering_vectors = {op: vector_dict[op] for op in operations if op in vector_dict}

    from cosyvoice.utils.file_utils import load_wav
    prompt_speech_16k = load_wav(reference_audio_path, 16000)

    for output in inference_zero_shot_with_steering(
        cosyvoice=model,
        tts_text=text,
        prompt_text=prompt_text,
        prompt_speech_16k=prompt_speech_16k,
        steering_vectors=steering_vectors,
        operations=operations,
        layers=layers,
        alpha=alpha,
        stream=False,
        speed=speed,
        inference_type=inference_type,
    ):
        audio = output["tts_speech"]
        if output_path:
            import os
            import torchaudio
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            torchaudio.save(output_path, audio, model.sample_rate)
            if verbose:
                print(f"  saved {output_path}")
        return {"audio": audio, "sample_rate": model.sample_rate}

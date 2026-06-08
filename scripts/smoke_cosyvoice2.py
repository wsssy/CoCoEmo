"""End-to-end smoke test for the CoCoEmo CosyVoice2 path.

Loads CosyVoice2 via the new package adapter, loads a shipped steering vector,
and synthesizes one utterance with no-steer (alpha=0) and steered (alpha=3.0),
saving both wavs. Validates that the refactored package actually runs on GPU,
not just imports.

Env vars (set by the sbatch wrapper):
    COSYVOICE_ROOT  - local CosyVoice repo clone
    COCOEMO_MODEL   - CosyVoice2-0.5B pretrained model dir
    COCOEMO_SV      - a steering vector .pt
    COCOEMO_REF     - reference (neutral) wav
    COCOEMO_OUT     - output dir for wavs
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import cocoemo.backbones.cosyvoice2 as cv2
from cocoemo.steering import load_steering_vectors


def main():
    model_dir = os.environ["COCOEMO_MODEL"]
    sv_path = os.environ["COCOEMO_SV"]
    ref_wav = os.environ["COCOEMO_REF"]
    out_dir = os.environ.get("COCOEMO_OUT", "smoke_out")
    text = os.environ.get("COCOEMO_TEXT", "I can not believe you did that.")
    prompt_text = os.environ.get("COCOEMO_PROMPT_TEXT", "")

    os.makedirs(out_dir, exist_ok=True)

    print(f"[smoke] SPEC: layers={cv2.SPEC.steer_layers} op={cv2.SPEC.steer_operator} "
          f"dim={cv2.SPEC.hidden_dim}", flush=True)
    print(f"[smoke] loading CosyVoice2 from {model_dir}", flush=True)
    model = cv2.load_model(model_dir)

    print(f"[smoke] loading steering vector {sv_path}", flush=True)
    sv = load_steering_vectors(sv_path)

    layers = cv2.SPEC.steer_layers
    ops = [cv2.SPEC.steer_operator]

    for alpha in (0.0, 3.0, 5.0):
        tag = f"alpha_{alpha:.1f}"
        out_path = os.path.join(out_dir, f"smoke_{tag}.wav")
        print(f"[smoke] synthesizing {tag} -> {out_path}", flush=True)
        res = cv2.generate_steered_speech(
            model=model,
            text=text,
            reference_audio_path=ref_wav,
            steering_vectors=sv,
            layers=layers,
            prompt_text=prompt_text,
            alpha=alpha,
            operations=ops,
            output_path=out_path,
            verbose=True,
        )
        audio = res["audio"]
        print(f"[smoke]   ok: shape={tuple(audio.shape)} sr={res['sample_rate']}", flush=True)

    print("[smoke] SUCCESS: CosyVoice2 steering path runs end-to-end.", flush=True)


if __name__ == "__main__":
    main()

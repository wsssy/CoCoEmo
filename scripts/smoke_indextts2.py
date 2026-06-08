"""End-to-end smoke test for the CoCoEmo IndexTTS2 path.

Loads IndexTTS2 via the new package adapter, loads a shipped steering vector,
and synthesizes one utterance with no-steer (alpha=0) and steered (alpha=3.0),
saving both wavs.

Env vars:
    INDEXTTS_ROOT   - local IndexTTS2 repo clone
    COCOEMO_MODEL   - IndexTTS2 checkpoints dir
    COCOEMO_CFG     - IndexTTS2 config.yaml path
    COCOEMO_SV      - a steering vector .pt
    COCOEMO_REF     - reference (neutral) wav for speaker embedding
    COCOEMO_OUT     - output dir for wavs
"""

import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import cocoemo.backbones.indextts2 as idx
from cocoemo.steering import load_steering_vectors


def main():
    model_dir = os.environ["COCOEMO_MODEL"]
    cfg_path = os.environ.get("COCOEMO_CFG", os.path.join(model_dir, "config.yaml"))
    sv_path = os.environ["COCOEMO_SV"]
    ref_wav = os.environ["COCOEMO_REF"]
    out_dir = os.environ.get("COCOEMO_OUT", "smoke_out_idx")
    text = os.environ.get("COCOEMO_TEXT", "I can not believe you did that.")

    os.makedirs(out_dir, exist_ok=True)

    print(f"[smoke-idx] SPEC: layers={idx.SPEC.steer_layers} op={idx.SPEC.steer_operator} "
          f"dim={idx.SPEC.hidden_dim}", flush=True)
    print(f"[smoke-idx] loading IndexTTS2 from {model_dir}", flush=True)
    model = idx.load_model(model_dir, cfg_path=cfg_path,
                           use_fp16=False, use_cuda_kernel=False, use_deepspeed=False)

    print(f"[smoke-idx] loading steering vector {sv_path}", flush=True)
    sv = load_steering_vectors(sv_path)

    layers = idx.SPEC.steer_layers
    ops = [idx.SPEC.steer_operator]

    for alpha in (0.0, 3.0, 5.0):
        tag = f"alpha_{alpha:.1f}"
        out_path = os.path.join(out_dir, f"smoke_idx_{tag}.wav")
        print(f"[smoke-idx] synthesizing {tag} -> {out_path}", flush=True)
        res = idx.generate_steered_speech(
            model=model,
            text=text,
            reference_audio_path=ref_wav,
            steering_vectors=sv,
            layers=layers,
            alpha=alpha,
            operations=ops,
            output_path=out_path,
            verbose=True,
        )
        if res["audio"] is not None:
            print(f"[smoke-idx]   ok: shape={tuple(res['audio'].shape)} sr={res['sample_rate']}", flush=True)
        else:
            print(f"[smoke-idx]   ok: wrote to {out_path}", flush=True)

    print("[smoke-idx] SUCCESS: IndexTTS2 steering path runs end-to-end.", flush=True)


if __name__ == "__main__":
    main()

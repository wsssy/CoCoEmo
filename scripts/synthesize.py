"""Stage 3: Steered speech synthesis.

Runs either CosyVoice2 or IndexTTS2 through the same interface, driven by the
``--backbone`` flag.

Two steering modes:

  Single-emotion: pass ``--steering-vector`` pointing to one .pt file.
  Mixed-emotion:  pass ``--steering-dir`` pointing to a directory of .pt files
                  (e.g. ``steering_vectors/cosyvoice2/``). In manifest mode the
                  soft-label columns (p_happy, p_sad, ...) are used to compose
                  per-sample mixed vectors via Eq. 7.

Usage (single emotion, single utterance):
    python scripts/synthesize.py \
        --backbone cosyvoice2 \
        --model-dir /path/to/CosyVoice2-0.5B \
        --steering-vector steering_vectors/cosyvoice2/angry_neutral_attn_output.pt \
        --text "I can not believe you did that." \
        --reference-audio neutral_ref.wav \
        --alpha 3.0 \
        --output-dir out/single

Usage (mixed emotion, manifest):
    python scripts/synthesize.py \
        --backbone cosyvoice2 \
        --model-dir /path/to/CosyVoice2-0.5B \
        --steering-dir steering_vectors/cosyvoice2 \
        --manifest examples/toy_mixed_manifest.csv \
        --data-root examples/toy_audio \
        --alpha 3.0 \
        --output-dir out/mixed
"""

import argparse
import glob
import importlib
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

EMOTIONS = ["angry", "happy", "sad", "surprise"]


def get_backbone(name):
    mod = importlib.import_module(f"cocoemo.backbones.{name}")
    return mod


def main():
    parser = argparse.ArgumentParser(description="CoCoEmo unified steered synthesis")
    parser.add_argument("--backbone", choices=["cosyvoice2", "indextts2"], required=True)
    parser.add_argument("--model-dir", required=True, help="TTS model/checkpoint directory")
    parser.add_argument("--cfg-path", default=None, help="IndexTTS2 config.yaml (default: <model-dir>/config.yaml)")
    parser.add_argument("--alpha", type=float, default=3.0, help="Steering strength")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Override steering layers (default: backbone SPEC)")
    parser.add_argument("--operation", default=None,
                        help="Override steering operator (default: backbone SPEC)")
    parser.add_argument("--output-dir", required=True, help="Directory for output wavs")

    # Steering source (mutually exclusive)
    sv_group = parser.add_mutually_exclusive_group(required=True)
    sv_group.add_argument("--steering-vector", default=None,
                          help="Path to a single .pt steering vector (single-emotion mode)")
    sv_group.add_argument("--steering-dir", default=None,
                          help="Directory of .pt steering vectors (mixed-emotion mode). "
                               "Composes per-sample mixed vectors from manifest soft labels "
                               "(p_happy, p_sad, p_angry, p_surprise).")

    # Single-utterance mode
    parser.add_argument("--text", default=None, help="Text to synthesize (single-utterance mode)")
    parser.add_argument("--reference-audio", default=None, help="Reference audio path (single-utterance)")
    parser.add_argument("--prompt-text", default="",
                        help="Transcript of the reference audio (CosyVoice2 zero-shot; "
                             "single-utterance mode only)")

    # Manifest mode
    parser.add_argument("--manifest", default=None, help="Manifest CSV path (batch mode)")
    parser.add_argument("--data-root", default=None, help="Root dir for manifest-relative audio paths")
    parser.add_argument("--split", default=None, help="Optional split filter (train/val/test)")
    parser.add_argument("--text-col", default="text", help="Manifest column for synthesis text")
    parser.add_argument("--ref-col", default="reference_filepath",
                        help="Manifest column for reference audio path")
    parser.add_argument("--ref-text-col", default="reference_text",
                        help="Manifest column for the transcript of the reference audio "
                             "(CosyVoice2 prompt_text)")
    parser.add_argument("--id-col", default="wav_filename",
                        help="Manifest column for output filename (no .wav needed)")
    parser.add_argument("--max-samples", type=int, default=None, help="Cap on samples to synthesize")

    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # --- load backbone ---
    bb = get_backbone(args.backbone)
    spec = bb.SPEC
    layers = args.layers or spec.steer_layers
    operation = args.operation or spec.steer_operator
    operations = [operation]

    mixed_mode = args.steering_dir is not None

    print(f"[synthesize] backbone={spec.name} layers={layers} op={operation} alpha={args.alpha} "
          f"mode={'mixed' if mixed_mode else 'single'}", flush=True)

    load_kwargs = {}
    if args.backbone == "indextts2":
        load_kwargs.update(cfg_path=args.cfg_path, use_fp16=False,
                           use_cuda_kernel=False, use_deepspeed=False)
    model = bb.load_model(args.model_dir, **load_kwargs)

    from cocoemo.steering import load_steering_vectors

    # --- load steering vectors ---
    if mixed_mode:
        from cocoemo.steering._core_cosyvoice import create_sample_specific_mixed_vectors

        sv_files = {}
        for emo in EMOTIONS:
            pattern = os.path.join(args.steering_dir, f"{emo}_*_attn_output.pt")
            matches = glob.glob(pattern)
            if matches:
                sv_files[emo] = matches[0]
        if not sv_files:
            parser.error(f"No steering vectors found in {args.steering_dir}")
        print(f"[synthesize] loaded {len(sv_files)} emotion vectors: {list(sv_files.keys())}", flush=True)
    else:
        sv = load_steering_vectors(args.steering_vector)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- build sample list ---
    samples = []
    if args.manifest:
        import pandas as pd
        df = pd.read_csv(args.manifest)
        if args.split and "split" in df.columns:
            df = df[df["split"] == args.split].reset_index(drop=True)
        path_cols = [args.ref_col, "filepath"]
        if args.data_root:
            for col in path_cols:
                if col in df.columns:
                    df[col] = df[col].map(lambda p: os.path.join(args.data_root, p) if isinstance(p, str) else p)
        if args.max_samples:
            df = df.head(args.max_samples)
        for _, row in df.iterrows():
            text = row.get(args.text_col, "")
            ref = row.get(args.ref_col, "")
            ref_text = row.get(args.ref_text_col, "")
            sid = row.get(args.id_col, f"sample_{_}")
            s = {"text": str(text), "reference": str(ref),
                 "reference_text": str(ref_text) if ref_text == ref_text else "",
                 "id": str(sid)}
            if mixed_mode:
                s["emotion_percentages"] = {
                    f"p_{emo}": float(row.get(f"p_{emo}", 0.0)) for emo in EMOTIONS
                }
            samples.append(s)
    elif args.text and args.reference_audio:
        if mixed_mode:
            parser.error("--steering-dir (mixed mode) requires --manifest with soft-label columns.")
        samples.append({"text": args.text, "reference": args.reference_audio,
                        "reference_text": args.prompt_text, "id": "single"})
    else:
        parser.error("Provide either --text + --reference-audio, or --manifest.")

    print(f"[synthesize] {len(samples)} sample(s) to synthesize", flush=True)

    # --- synthesize ---
    for i, s in enumerate(samples):
        out_path = os.path.join(args.output_dir, f"{s['id']}.wav")
        extra = {}
        if args.backbone == "cosyvoice2":
            extra["prompt_text"] = s.get("reference_text", "")

        if mixed_mode:
            sample_sv = create_sample_specific_mixed_vectors(
                steering_files_dict=sv_files,
                emotion_percentages=s["emotion_percentages"],
            )
        else:
            sample_sv = sv

        try:
            res = bb.generate_steered_speech(
                model=model,
                text=s["text"],
                reference_audio_path=s["reference"],
                steering_vectors=sample_sv,
                layers=layers,
                alpha=args.alpha,
                operations=operations,
                output_path=out_path,
                verbose=args.verbose,
                **extra,
            )
            if args.verbose or (i + 1) % 50 == 0 or i == 0:
                shape = tuple(res["audio"].shape) if res.get("audio") is not None else "?"
                print(f"  [{i+1}/{len(samples)}] {s['id']} -> {out_path} shape={shape}", flush=True)
        except Exception as e:
            print(f"  [{i+1}/{len(samples)}] FAILED {s['id']}: {type(e).__name__}: {e}", flush=True)

    print(f"[synthesize] done: {len(samples)} samples -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

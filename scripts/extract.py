"""Stage 2: Steering-vector extraction — compute mean-difference vectors (Eq. 6).

Extracts per-layer activations from emotional vs neutral audio (train split only),
computes mean-difference steering vectors, and saves them as portable .pt files.

Uses single-emotion labeled data. Pass --split train to use only the training set
(as per Section D.1: training set is used for computing steering vector centroids).

Inputs:
    --backbone, --model-dir, --manifest (single-emotion CSV), --data-root,
    --pos-emotion, --neg-emotion, --split

Outputs:
    <output-dir>/<pos>_<neg>_<operation>.pt   — steering vector per emotion pair

Usage:
    python scripts/extract.py \
        --backbone cosyvoice2 \
        --model-dir /path/to/CosyVoice2-0.5B \
        --manifest examples/toy_single_manifest.csv \
        --data-root examples/toy_audio \
        --pos-emotion angry --neg-emotion neutral \
        --split train \
        --output-dir steering_vectors/cosyvoice2
"""

import argparse
import importlib
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    parser = argparse.ArgumentParser(description="CoCoEmo steering-vector extraction")
    parser.add_argument("--backbone", choices=["cosyvoice2", "indextts2"], required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cfg-path", default=None, help="IndexTTS2 config.yaml")
    parser.add_argument("--manifest", required=True, nargs="+",
                        help="One or more manifest CSVs")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--pos-emotion", required=True, help="Target emotion (e.g. angry)")
    parser.add_argument("--neg-emotion", default="neutral", help="Baseline emotion (default: neutral)")
    parser.add_argument("--operations", nargs="+", default=["attn_output"])
    parser.add_argument("--split", default=None, help="Filter by split column if present")
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import pandas as pd

    bb = importlib.import_module(f"cocoemo.backbones.{args.backbone}")
    spec = bb.SPEC

    print(f"[extract] backbone={spec.name} pos={args.pos_emotion} neg={args.neg_emotion} "
          f"ops={args.operations}", flush=True)

    load_kwargs = {}
    if args.backbone == "indextts2":
        load_kwargs.update(cfg_path=args.cfg_path, use_fp16=False,
                           use_cuda_kernel=False, use_deepspeed=False)
    model_obj = bb.load_model(args.model_dir, **load_kwargs)

    if args.backbone == "cosyvoice2":
        from cocoemo.steering._core_cosyvoice import (
            extract_with_hooks_audio, get_cosyvoice_op_dict, create_mean_difference,
        )
        op_dict = get_cosyvoice_op_dict(args.operations)
        cosyvoice = model_obj
        model = cosyvoice.model
        frontend = cosyvoice.frontend
    else:
        from cocoemo.steering._core_indextts import (
            extract_with_hooks_audio, get_indextts2_op_dict, create_mean_difference,
        )
        op_dict = get_indextts2_op_dict(args.operations)
        model = model_obj
        frontend = None

    frames = []
    for m in args.manifest:
        df = pd.read_csv(m)
        if args.split and "split" in df.columns:
            df = df[df["split"] == args.split].reset_index(drop=True)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    emo_col = "dominant_emotion" if "dominant_emotion" in data.columns else "emotion"
    pos_df = data[data[emo_col] == args.pos_emotion].reset_index(drop=True)

    if len(pos_df) == 0:
        print(f"[extract] ERROR: no samples with {emo_col}={args.pos_emotion}", flush=True)
        sys.exit(1)

    emo_paths, emo_texts = [], []
    neu_paths, neu_texts = [], []
    for _, row in pos_df.iterrows():
        fp = str(row["filepath"])
        rp = str(row.get("reference_filepath", ""))
        if args.data_root:
            fp = os.path.join(args.data_root, fp)
            if rp:
                rp = os.path.join(args.data_root, rp)
        emo_paths.append(fp)
        emo_texts.append(str(row.get("text", "")))
        if rp:
            neu_paths.append(rp)
            neu_texts.append(str(row.get("reference_text", "")))

    if not neu_paths:
        print("[extract] ERROR: no neutral reference paths found in manifest", flush=True)
        sys.exit(1)

    print(f"[extract] {len(emo_paths)} emotional + {len(neu_paths)} neutral samples", flush=True)

    extract_kwargs = dict(operations=args.operations, layers=spec.num_layers, op_dict=op_dict)
    if args.backbone == "cosyvoice2":
        extract_kwargs["model"] = model
        extract_kwargs["frontend"] = frontend
    else:
        extract_kwargs["model"] = model

    print("[extract] extracting emotional activations...", flush=True)
    pos_reps = extract_with_hooks_audio(audio_paths=emo_paths, texts=emo_texts, **extract_kwargs)

    print("[extract] extracting neutral activations...", flush=True)
    neg_reps = extract_with_hooks_audio(audio_paths=neu_paths, texts=neu_texts, **extract_kwargs)

    print("[extract] computing mean-difference vectors...", flush=True)
    steering_vectors = create_mean_difference(args.operations, pos_reps, neg_reps)

    for op in args.operations:
        out_name = f"{args.pos_emotion}_{args.neg_emotion}_{op}.pt"
        out_path = os.path.join(args.output_dir, out_name)

        torch.save({
            "steering_vectors": {op: steering_vectors[op]},
            "metadata": {
                "pos_emotion": args.pos_emotion,
                "neg_emotion": args.neg_emotion,
                "operation": op,
                "backbone": spec.name,
                "n_pos": len(emo_paths),
                "n_neg": len(neu_paths),
                "split": args.split or "all",
            },
        }, out_path)

        print(f"[extract] saved {out_name}", flush=True)

    print(f"[extract] done -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()

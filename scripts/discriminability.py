"""Stage 1: Discriminability analysis — identify best steering layers.

Extracts activations from emotional vs neutral audio, trains linear probes
per layer, and saves per-layer accuracies + recommended layers (Section 2).

Uses single-emotion labeled data with train/val splits:
  - train split: trains linear probes
  - val split:   evaluates probes (site selection)

Inputs:
    --backbone, --model-dir, --manifest (single-emotion CSV with split column), --data-root

Outputs:
    <output-dir>/discriminability.json   — per-layer accuracies, best layers

Usage:
    python scripts/discriminability.py \
        --backbone cosyvoice2 \
        --model-dir /path/to/CosyVoice2-0.5B \
        --manifest examples/toy_single_manifest.csv \
        --data-root examples/toy_audio \
        --output-dir results/discriminability
"""

import argparse
import importlib
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def main():
    parser = argparse.ArgumentParser(description="CoCoEmo discriminability analysis")
    parser.add_argument("--backbone", choices=["cosyvoice2", "indextts2"], required=True)
    parser.add_argument("--model-dir", required=True, help="TTS model/checkpoint directory")
    parser.add_argument("--cfg-path", default=None, help="IndexTTS2 config.yaml")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with emotional + reference audio")
    parser.add_argument("--data-root", default=None, help="Root dir for manifest-relative audio paths")
    parser.add_argument("--operations", nargs="+", default=None,
                        help="Operations to probe (default: all for the backbone)")
    parser.add_argument("--classifier", default="linear", choices=["linear", "centroid"])
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    COSYVOICE2_ALL_OPS = [
        "emb_pre_attn_post_ln", "attn_output", "q_proj", "k_proj", "v_proj",
        "W0_x_attn_output", "emb_post_attn_pre_ln", "emb_post_attn_post_ln",
        "emb_post_mlp_residual", "layer_output",
    ]
    INDEXTTS2_ALL_OPS = [
        "emb_pre_attn_post_ln", "attn_output", "qkv_proj",
        "W0_x_attn_output", "emb_post_attn_pre_ln", "emb_post_attn_post_ln",
        "emb_post_mlp_residual", "layer_output",
    ]
    if args.operations is None:
        args.operations = COSYVOICE2_ALL_OPS if args.backbone == "cosyvoice2" else INDEXTTS2_ALL_OPS
    os.makedirs(args.output_dir, exist_ok=True)

    import pandas as pd

    bb = importlib.import_module(f"cocoemo.backbones.{args.backbone}")
    spec = bb.SPEC

    print(f"[discriminability] backbone={spec.name} ops={args.operations}", flush=True)

    load_kwargs = {}
    if args.backbone == "indextts2":
        load_kwargs.update(cfg_path=args.cfg_path, use_fp16=False,
                           use_cuda_kernel=False, use_deepspeed=False)
    model_obj = bb.load_model(args.model_dir, **load_kwargs)

    if args.backbone == "cosyvoice2":
        from cocoemo.steering._core_cosyvoice import (
            extract_with_hooks_audio, get_cosyvoice_op_dict,
        )
        op_dict = get_cosyvoice_op_dict(args.operations)
        cosyvoice = model_obj
        model = cosyvoice.model
        frontend = cosyvoice.frontend
    else:
        from cocoemo.steering._core_indextts import (
            extract_with_hooks_audio, get_indextts2_op_dict,
        )
        op_dict = get_indextts2_op_dict(args.operations)
        model = model_obj
        frontend = None

    from cocoemo.discriminability._probe import compute_discriminability_for_steering

    df = pd.read_csv(args.manifest)

    if "split" not in df.columns:
        print("[discriminability] ERROR: manifest must have a 'split' column (train/val/test)", flush=True)
        sys.exit(1)

    emo_col = "emotion" if "emotion" in df.columns else "dominant_emotion"
    emotional = df[df[emo_col] != "neutral"]

    def collect_paths(subset):
        emo_p, emo_t, neu_p, neu_t = [], [], [], []
        for _, row in subset.iterrows():
            fp = str(row["filepath"])
            rp = str(row["reference_filepath"])
            if args.data_root:
                fp = os.path.join(args.data_root, fp)
                rp = os.path.join(args.data_root, rp)
            emo_p.append(fp)
            emo_t.append(str(row.get("text", "")))
            neu_p.append(rp)
            neu_t.append(str(row.get("reference_text", "")))
        return emo_p, emo_t, neu_p, neu_t

    train_df = emotional[emotional["split"] == "train"]
    val_df = emotional[emotional["split"] == "val"]

    train_emo, train_emo_t, train_neu, train_neu_t = collect_paths(train_df)
    eval_emo, eval_emo_t, eval_neu, eval_neu_t = collect_paths(val_df)

    print(f"[discriminability] {len(train_emo)} train + {len(eval_emo)} val samples "
          f"(emotions: {sorted(train_df[emo_col].unique())})", flush=True)

    extract_kwargs = dict(operations=args.operations, layers=spec.num_layers, op_dict=op_dict)
    if args.backbone == "cosyvoice2":
        extract_kwargs["model"] = model
        extract_kwargs["frontend"] = frontend
    else:
        extract_kwargs["model"] = model

    pos_train = extract_with_hooks_audio(audio_paths=train_emo, texts=train_emo_t, **extract_kwargs)
    neg_train = extract_with_hooks_audio(audio_paths=train_neu, texts=train_neu_t, **extract_kwargs)
    pos_eval = extract_with_hooks_audio(audio_paths=eval_emo, texts=eval_emo_t, **extract_kwargs)
    neg_eval = extract_with_hooks_audio(audio_paths=eval_neu, texts=eval_neu_t, **extract_kwargs)

    op_to_layer_to_acc, _ = compute_discriminability_for_steering(
        pos_reps=pos_train, neg_reps=neg_train,
        pos_reps_eval=pos_eval, neg_reps_eval=neg_eval,
        operations=args.operations,
        classifier_type=args.classifier,
        return_metrics=True,
    )

    results = {}
    for op in args.operations:
        layer_to_acc = op_to_layer_to_acc[op]
        best = sorted(layer_to_acc, key=layer_to_acc.get, reverse=True)[:5]
        results[op] = {
            "layer_accuracies": {str(k): v for k, v in layer_to_acc.items()},
            "recommended_layers": best,
        }
        print(f"\n  [{op}] layer-wise discriminability:")
        for layer in sorted(layer_to_acc.keys()):
            bar = "#" * int(layer_to_acc[layer] * 40)
            print(f"    layer {layer:2d}: {layer_to_acc[layer]:.3f}  {bar}")
        print(f"  recommended layers: {best}")

    out_path = os.path.join(args.output_dir, "discriminability.json")
    with open(out_path, "w") as f:
        json.dump({"backbone": spec.name, "operations": args.operations,
                   "classifier": args.classifier, "results": results}, f, indent=2)
    print(f"\n[discriminability] saved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()

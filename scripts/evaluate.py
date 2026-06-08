"""Stage 4: Evaluation — per-sample and aggregate metrics.

Computes per-utterance metrics on synthesized wavs and, when a baseline
(alpha=0) wav directory is provided, also computes the mixed-emotion metrics
(Spearman ρ, H-Rate) that measure whether steering follows the intended
emotion proportions.

Per-sample metrics (no baseline needed):
    E-SIM  — Emotion2Vec embedding cosine similarity with ground-truth audio
    TEP    — target emotion probability (mean over active emotions)
    S-SIM  — WavLM speaker similarity with reference audio
    WER    — Whisper word error rate

Mixed-emotion metrics (require --baseline-wav-dir):
    ρ      — Spearman rank correlation between ground-truth emotion proportions
             and the steering-induced probability *increase* over baseline
    H-Rate — fraction of samples where the dominant emotion shows the largest
             probability increase

Outputs:
    <output-dir>/per_sample_results.json  — per-utterance metrics
    <output-dir>/summary.json             — aggregate averages

Usage:
    python scripts/evaluate.py \
        --wav-dir out/mixed_alpha3 \
        --baseline-wav-dir out/mixed_alpha0 \
        --manifest examples/toy_mixed_manifest.csv \
        --data-root examples/toy_audio \
        --output-dir results/evaluation
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

EMOTIONS = ["angry", "happy", "sad", "surprise"]


def main():
    parser = argparse.ArgumentParser(description="CoCoEmo evaluation")
    parser.add_argument("--wav-dir", required=True, help="Directory of synthesized wavs (steered)")
    parser.add_argument("--baseline-wav-dir", default=None,
                        help="Directory of baseline (alpha=0) wavs for mixed metrics (ρ, H-Rate)")
    parser.add_argument("--manifest", required=True, help="Manifest CSV with ground-truth info")
    parser.add_argument("--data-root", default=None, help="Root for manifest-relative audio paths")
    parser.add_argument("--split", default=None)
    parser.add_argument("--ref-col", default="reference_filepath",
                        help="Manifest column for reference audio (S-SIM)")
    parser.add_argument("--text-col", default="text", help="Manifest column for WER reference text")
    parser.add_argument("--id-col", default="wav_filename", help="Column to match wav filenames")
    parser.add_argument("--language", default="en", help="Whisper language hint")
    parser.add_argument("--device", default=None, help="torch device (default: auto)")
    parser.add_argument("--output-dir", required=True, help="Where to write results")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    import pandas as pd
    from cocoemo.evaluation.metrics import (
        load_models, get_emotion_probabilities, cosine_similarity_wavs,
        whisper_wer_for_wav, speaker_similarity,
    )
    from cocoemo.evaluation.mixed_metrics import spearman_rank_corr

    print("[evaluate] loading manifest...", flush=True)
    df = pd.read_csv(args.manifest)
    if args.split and "split" in df.columns:
        df = df[df["split"] == args.split].reset_index(drop=True)
    if args.data_root:
        for col in [args.ref_col, "filepath"]:
            if col in df.columns:
                df[col] = df[col].map(
                    lambda p: os.path.join(args.data_root, p) if isinstance(p, str) else p)

    print("[evaluate] loading metric models (Emotion2Vec, Whisper, WavLM)...", flush=True)
    models = load_models(device=args.device)

    wav_dir = Path(args.wav_dir)
    baseline_dir = Path(args.baseline_wav_dir) if args.baseline_wav_dir else None

    # ── Per-sample evaluation ──
    results = []
    for i, row in df.iterrows():
        sid = str(row.get(args.id_col, f"sample_{i}"))
        wav_path = wav_dir / f"{sid}.wav"
        if not wav_path.exists():
            continue

        ref_text = str(row.get(args.text_col, ""))
        ref_audio = str(row.get(args.ref_col, ""))
        gt_audio = str(row.get("filepath", ""))

        # Emotion percentages from manifest
        emotion_pcts = {f"p_{emo}": float(row.get(f"p_{emo}", 0.0)) for emo in EMOTIONS}
        target_emotions = [emo for emo in EMOTIONS if emotion_pcts.get(f"p_{emo}", 0) > 0]
        dominant = str(row.get("dominant_emotion", ""))

        entry = {
            "sentence_id": sid,
            "dominant_emotion": dominant,
            "target_emotions": target_emotions,
            "emotion_percentages": emotion_pcts,
        }

        # --- Emotion2Vec: probabilities (TEP) + E-SIM ---
        try:
            probs, pred_emo = get_emotion_probabilities(models["emotion_model"], str(wav_path))
            entry["emotion_probs"] = probs
            entry["predicted_emotion"] = pred_emo
            entry["target_emotion_scores"] = {emo: probs.get(emo, 0.0) for emo in target_emotions}
            entry["tep"] = float(np.mean([probs.get(emo, 0.0) for emo in target_emotions])) if target_emotions else None
        except Exception as e:
            entry["tep_error"] = str(e)

        if gt_audio and os.path.exists(gt_audio):
            try:
                entry["e_sim"] = cosine_similarity_wavs(models["emotion_model"], str(wav_path), gt_audio)
            except Exception:
                pass

        # --- S-SIM ---
        if ref_audio and os.path.exists(ref_audio):
            try:
                entry["s_sim"] = speaker_similarity(
                    models["speaker_feature_extractor"], models["speaker_model"],
                    str(wav_path), ref_audio,
                )
            except Exception:
                pass

        # --- WER ---
        if ref_text:
            try:
                entry["wer"] = whisper_wer_for_wav(
                    models["asr_model"], str(wav_path), ref_text, language=args.language,
                )
            except Exception as e:
                if i == 0:
                    print(f"[evaluate] WARNING: WER failed ({type(e).__name__}: {e}). "
                          "Is ffmpeg installed?", flush=True)

        # --- Baseline comparison for mixed metrics ---
        if baseline_dir is not None:
            base_wav = baseline_dir / f"{sid}.wav"
            if base_wav.exists() and target_emotions:
                try:
                    base_probs, _ = get_emotion_probabilities(models["emotion_model"], str(base_wav))
                    entry["baseline_emotion_probs"] = base_probs
                    entry["baseline_target_scores"] = {emo: base_probs.get(emo, 0.0) for emo in target_emotions}

                    p_vals = [emotion_pcts.get(f"p_{emo}", 0) for emo in target_emotions]
                    increases = [
                        entry["target_emotion_scores"].get(emo, 0) - base_probs.get(emo, 0)
                        for emo in target_emotions
                    ]
                    entry["emotion_increases"] = {emo: inc for emo, inc in zip(target_emotions, increases)}

                    if len(target_emotions) >= 2:
                        rho = spearman_rank_corr(p_vals, increases)
                        entry["spearman_rho"] = rho

                        # H-Rate: dominant emotion has largest increase?
                        if dominant in target_emotions:
                            dom_idx = target_emotions.index(dominant)
                            entry["dominant_hit"] = 1.0 if increases[dom_idx] == max(increases) else 0.0
                except Exception:
                    pass

        results.append(entry)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(df)}] {sid} evaluated", flush=True)

    # ── Save per-sample results ──
    per_sample_path = os.path.join(args.output_dir, "per_sample_results.json")
    with open(per_sample_path, "w") as f:
        json.dump({"evaluation": {"per_sample_results": results}}, f, indent=2)

    # ── Aggregate summary ──
    def safe_mean(key):
        vals = [r[key] for r in results if key in r and r[key] is not None]
        return float(np.mean(vals)) if vals else None

    def safe_count(key):
        return sum(1 for r in results if key in r and r[key] is not None)

    summary = {
        "n_samples": len(results),
        "per_sample_metrics": {
            "e_sim":  {"mean": safe_mean("e_sim"),  "n": safe_count("e_sim")},
            "tep":    {"mean": safe_mean("tep"),    "n": safe_count("tep")},
            "s_sim":  {"mean": safe_mean("s_sim"),  "n": safe_count("s_sim")},
            "wer":    {"mean": safe_mean("wer"),    "n": safe_count("wer")},
        },
    }

    if baseline_dir is not None:
        summary["mixed_metrics"] = {
            "spearman_rho": {"mean": safe_mean("spearman_rho"), "n": safe_count("spearman_rho")},
            "h_rate":       {"mean": safe_mean("dominant_hit"), "n": safe_count("dominant_hit")},
        }

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print summary ──
    print("\n  ┌──────────────────────────────────────────────────┐")
    print("  │             Evaluation Results Summary            │")
    print("  ├──────────────────────────────────────────────────┤")
    pm = summary["per_sample_metrics"]
    for name, label in [("e_sim", "E-SIM"), ("tep", "TEP"), ("s_sim", "S-SIM"), ("wer", "WER")]:
        m = pm[name]
        if m["mean"] is not None:
            print(f"  │  {label:<6s} : {m['mean']:.4f}  (n={m['n']})                │")
    if "mixed_metrics" in summary:
        mm = summary["mixed_metrics"]
        print("  ├──────────────────────────────────────────────────┤")
        print("  │  Mixed-emotion metrics (vs baseline)             │")
        print("  ├──────────────────────────────────────────────────┤")
        for name, label in [("spearman_rho", "ρ"), ("h_rate", "H-Rate")]:
            m = mm[name]
            if m["mean"] is not None:
                print(f"  │  {label:<6s} : {m['mean']:.4f}  (n={m['n']})                │")
    print("  └──────────────────────────────────────────────────┘")

    print(f"\n[evaluate] done: {len(results)} samples evaluated")
    print(f"  per-sample -> {per_sample_path}")
    print(f"  summary    -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()

"""Aggregate mixed-emotion metrics: Spearman rank correlation (rho) and
Dominant-Hit Rate (H-Rate).

These were previously tangled inside the plotting script; they are extracted here
as pure functions with NO matplotlib dependency so they can be unit-tested and
reused. They operate purely on the per-sample ``results.json`` written by the
synthesis/evaluation stage, computing each steered run's metrics *relative to the
no-steer (alpha=0) baseline*.

per_sample_results schema (one dict per utterance), as produced by the synthesis
pipeline:
    {
        "sentence_id": str,
        "target_emotions": [str, ...],
        "emotion_percentages": {"p_happy": float, "p_sad": float, ...},
        "target_emotion_scores": {"happy": float, "sad": float, ...},  # Emotion2Vec
    }
"""

import json
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
#  Rank-correlation primitives (no SciPy dependency)
# --------------------------------------------------------------------------- #
def _rankdata(values, atol=1e-8, rtol=1e-6):
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and np.isclose(values[order[j + 1]], values[order[i]], atol=atol, rtol=rtol):
            j += 1
        avg_rank = (i + j) / 2 + 1
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    return ranks


def _pearson_corr(x_vals, y_vals):
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    if x.size < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def spearman_rank_corr(x_vals, y_vals):
    x_ranks = _rankdata(np.asarray(x_vals, dtype=float))
    y_ranks = _rankdata(np.asarray(y_vals, dtype=float))
    return _pearson_corr(x_ranks, y_ranks)


# --------------------------------------------------------------------------- #
#  Loading per-sample results + baseline map
# --------------------------------------------------------------------------- #
def load_per_sample_results(results_json_path):
    with open(results_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("evaluation", {}).get("per_sample_results", [])


def build_baseline_map(root_dir, baseline_folder):
    """Map sentence_id -> baseline (alpha=0) per-sample record, for relative scoring."""
    baseline_path = Path(root_dir) / baseline_folder / "evaluation" / "results.json"
    if not baseline_path.exists():
        return {}
    baseline_samples = load_per_sample_results(baseline_path)
    return {item.get("sentence_id"): item for item in baseline_samples if item.get("sentence_id")}


# --------------------------------------------------------------------------- #
#  Spearman rho and Dominant-Hit Rate (relative to no-steer baseline)
# --------------------------------------------------------------------------- #
def compute_rank_correlation(results_json_path, baseline_map):
    """Mean per-sample Spearman correlation between ground-truth emotion proportions
    and the steering-induced probability *increase* over the baseline."""
    samples = load_per_sample_results(results_json_path)
    spearman_vals = []

    for sample in samples:
        sentence_id = sample.get("sentence_id")
        if not sentence_id or sentence_id not in baseline_map:
            continue
        baseline_sample = baseline_map[sentence_id]
        emotion_percentages = sample.get("emotion_percentages", {})
        target_emotions = sample.get("target_emotions", [])
        if not target_emotions:
            continue

        emotions = [
            emotion
            for emotion in target_emotions
            if emotion_percentages.get(f"p_{emotion}", 0.0) > 0
        ]
        if len(emotions) < 2:
            continue

        p_vals = [emotion_percentages.get(f"p_{emotion}", 0.0) for emotion in emotions]
        scores = sample.get("target_emotion_scores", {})
        baseline_scores = baseline_sample.get("target_emotion_scores", {})
        increases = []
        missing = False
        for emotion in emotions:
            if emotion not in scores or emotion not in baseline_scores:
                missing = True
                break
            increases.append(scores[emotion] - baseline_scores[emotion])
        if missing:
            continue

        spearman = spearman_rank_corr(p_vals, increases)
        if spearman is None:
            continue
        spearman_vals.append(spearman)

    if not spearman_vals:
        return None
    return float(np.mean(spearman_vals))


def compute_dominant_hit_rate(results_json_path, baseline_map, atol=1e-8, rtol=1e-6):
    """Fraction of samples where the ground-truth dominant emotion also shows the
    largest steering-induced probability increase over the baseline."""
    samples = load_per_sample_results(results_json_path)
    hit_vals = []

    for sample in samples:
        sentence_id = sample.get("sentence_id")
        if not sentence_id or sentence_id not in baseline_map:
            continue
        baseline_sample = baseline_map[sentence_id]
        emotion_percentages = sample.get("emotion_percentages", {})
        target_emotions = sample.get("target_emotions", [])
        if not target_emotions:
            continue

        emotions = [
            emotion
            for emotion in target_emotions
            if emotion_percentages.get(f"p_{emotion}", 0.0) > 0
        ]
        if len(emotions) < 2:
            continue

        p_vals = np.array([emotion_percentages.get(f"p_{emotion}", 0.0) for emotion in emotions], dtype=float)
        scores = sample.get("target_emotion_scores", {})
        baseline_scores = baseline_sample.get("target_emotion_scores", {})
        increases = []
        missing = False
        for emotion in emotions:
            if emotion not in scores or emotion not in baseline_scores:
                missing = True
                break
            increases.append(scores[emotion] - baseline_scores[emotion])
        if missing:
            continue

        increases = np.array(increases, dtype=float)
        p_max = np.max(p_vals)
        inc_max = np.max(increases)
        dom_indices = np.where(np.isclose(p_vals, p_max, atol=atol, rtol=rtol))[0]
        inc_indices = np.where(np.isclose(increases, inc_max, atol=atol, rtol=rtol))[0]
        hit_vals.append(1.0 if np.intersect1d(dom_indices, inc_indices).size > 0 else 0.0)

    if not hit_vals:
        return None
    return float(np.mean(hit_vals))


# --------------------------------------------------------------------------- #
#  Parsing the human-readable results.txt summary (E-SIM / S-SIM / WER / TEP)
# --------------------------------------------------------------------------- #
def parse_results_txt(results_path):
    """Parse the per-run ``results.txt`` summary into a metrics dict."""
    metrics = {
        "wer_mean": None,
        "speaker_sim_mean": None,
        "emotion2vec_mean": None,
        "target_emotion_scores": {},
    }
    target_rows = []
    state = None

    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("Word Error Rate"):
                state = "wer"
                continue
            if stripped.startswith("Speaker Similarity"):
                state = "speaker"
                continue
            if stripped.startswith("Emotion2Vec Similarity"):
                state = "emotion2vec"
                continue
            if stripped.startswith("Target Emotion Scores (conditioned on target set)"):
                state = "target_table"
                continue

            if state in {"wer", "speaker", "emotion2vec"} and stripped.startswith("Mean:"):
                value = stripped.split(":", 1)[1].strip()
                try:
                    value = float(value)
                except ValueError:
                    value = None
                if state == "wer":
                    metrics["wer_mean"] = value
                elif state == "speaker":
                    metrics["speaker_sim_mean"] = value
                elif state == "emotion2vec":
                    metrics["emotion2vec_mean"] = value
                state = None
                continue

            if state == "target_table":
                if not stripped:
                    if target_rows:
                        state = None
                    continue
                if stripped.startswith("Emotion") or stripped.startswith("---"):
                    continue
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        emotion = parts[0]
                        count = int(parts[1])
                        mean = float(parts[2])
                        target_rows.append((emotion, count, mean))
                    except ValueError:
                        continue

    if target_rows:
        for emotion, count, mean in target_rows:
            metrics["target_emotion_scores"][emotion] = {"count": count, "mean": mean}

    return metrics

"""CoCoEmo evaluation: per-utterance metrics and aggregate mixed-emotion metrics.

Runs in a standalone environment (no TTS backbone required) on synthesized wavs
and the per-sample ``results.json`` they produce.
"""

from .mixed_metrics import (
    spearman_rank_corr,
    compute_rank_correlation,
    compute_dominant_hit_rate,
    build_baseline_map,
    parse_results_txt,
)

__all__ = [
    "spearman_rank_corr",
    "compute_rank_correlation",
    "compute_dominant_hit_rate",
    "build_baseline_map",
    "parse_results_txt",
]

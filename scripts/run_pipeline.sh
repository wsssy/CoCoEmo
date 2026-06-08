#!/usr/bin/env bash
# CoCoEmo full pipeline: discriminability → extraction → synthesis → evaluation
#
# Edit the variables below to match your setup, then run:
#   bash scripts/run_pipeline.sh
#
set -euo pipefail

# ── Configuration (edit these, or override via environment variables) ───────
BACKBONE="${BACKBONE:-cosyvoice2}"                          # cosyvoice2 | indextts2
MODEL_DIR="${MODEL_DIR:-/path/to/CosyVoice2-0.5B}"         # TTS model checkpoint
CFG_PATH="${CFG_PATH:-}"                                    # IndexTTS2 only: config.yaml path
PYTHON="${PYTHON:-python}"                                  # python executable

SINGLE_MANIFEST="${SINGLE_MANIFEST:-examples/toy_single_manifest.csv}"
MIXED_MANIFEST="${MIXED_MANIFEST:-examples/toy_mixed_manifest.csv}"
DATA_ROOT="${DATA_ROOT:-examples/toy_audio}"
OUT="${OUT:-pipeline_out}"

ALPHA="${ALPHA:-3.0}"                                       # steering strength
# ────────────────────────────────────────────────────────────────────────────

CFG_FLAG=""
if [ -n "$CFG_PATH" ]; then
    CFG_FLAG="--cfg-path $CFG_PATH"
fi

echo "=== CoCoEmo Pipeline ($BACKBONE) ==="
echo ""

# ── Stage 1: Discriminability (single-emotion, train+val splits) ──
echo "=== Stage 1: Discriminability ==="
$PYTHON scripts/discriminability.py \
    --backbone "$BACKBONE" \
    --model-dir "$MODEL_DIR" $CFG_FLAG \
    --manifest "$SINGLE_MANIFEST" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT/discriminability"
echo ""

# ── Stage 2: Steering vector extraction (single-emotion, train split only) ──
echo "=== Stage 2: Steering vector extraction ==="
for EMO in angry happy sad surprise; do
    echo "  extracting: $EMO vs neutral"
    $PYTHON scripts/extract.py \
        --backbone "$BACKBONE" \
        --model-dir "$MODEL_DIR" $CFG_FLAG \
        --manifest "$SINGLE_MANIFEST" \
        --data-root "$DATA_ROOT" \
        --pos-emotion "$EMO" \
        --split train \
        --output-dir "$OUT/steering_vectors"
done
echo ""

# ── Stage 3: Mixed-emotion synthesis (baseline + steered) ──
echo "=== Stage 3: Synthesis ==="
echo "  3a: baseline (alpha=0)"
$PYTHON scripts/synthesize.py \
    --backbone "$BACKBONE" \
    --model-dir "$MODEL_DIR" $CFG_FLAG \
    --steering-dir "$OUT/steering_vectors" \
    --manifest "$MIXED_MANIFEST" \
    --data-root "$DATA_ROOT" \
    --alpha 0.0 \
    --output-dir "$OUT/synthesis_alpha0"

echo "  3b: steered (alpha=$ALPHA)"
$PYTHON scripts/synthesize.py \
    --backbone "$BACKBONE" \
    --model-dir "$MODEL_DIR" $CFG_FLAG \
    --steering-dir "$OUT/steering_vectors" \
    --manifest "$MIXED_MANIFEST" \
    --data-root "$DATA_ROOT" \
    --alpha "$ALPHA" \
    --output-dir "$OUT/synthesis_alpha${ALPHA}"
echo ""

# ── Stage 4: Evaluation ──
echo "=== Stage 4: Evaluation ==="
$PYTHON scripts/evaluate.py \
    --wav-dir "$OUT/synthesis_alpha${ALPHA}" \
    --baseline-wav-dir "$OUT/synthesis_alpha0" \
    --manifest "$MIXED_MANIFEST" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUT/evaluation"
echo ""

echo "=== Pipeline complete ==="
echo "  discriminability -> $OUT/discriminability/discriminability.json"
echo "  steering vectors -> $OUT/steering_vectors/"
echo "  synthesis (alpha=0)    -> $OUT/synthesis_alpha0/"
echo "  synthesis (alpha=$ALPHA) -> $OUT/synthesis_alpha${ALPHA}/"
echo "  evaluation       -> $OUT/evaluation/summary.json"

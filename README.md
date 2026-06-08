# CoCoEmo: Composable and Controllable Human-Like Emotional TTS via Activation Steering

## Overview
Implementation for *CoCoEmo* (ICML 2026). CoCoEmo steers the
**speech language model (SLM)** of hybrid TTS systems with emotion direction vectors to
produce **quantitative mixed-emotion** speech and **text-emotion mismatch** speech,
without retraining.



Two backbones are supported: **CosyVoice2** and **IndexTTS2**.

<p align="center">
  <img src="cocoemo/images/pipeline.png" width="100%" alt="CoCoEmo pipeline overview">
</p>

## Setup

### 1. Clone CoCoEmo

```bash
git clone https://github.com/wsssy/CoCoEmo.git
cd CoCoEmo
```

### 2. Set up a TTS backbone

Follow the official installation for your chosen backbone, then install CoCoEmo
into the same environment.

**CosyVoice2** ([official repo](https://github.com/FunAudioLLM/CosyVoice)):
```bash
# Follow CosyVoice2 official setup, then:
pip install -e /path/to/CoCoEmo
export COSYVOICE_ROOT=/path/to/CosyVoice
```

**IndexTTS2** ([official repo](https://github.com/index-tts/index-tts)):
```bash
# Follow IndexTTS2 official setup (uses uv, creates .venv), then:
/path/to/index-tts/.venv/bin/pip install -e /path/to/CoCoEmo
export INDEXTTS_ROOT=/path/to/index-tts
```

### 3. Evaluation dependencies (optional)

Evaluation uses Emotion2Vec, Whisper, and WavLM -- no TTS backbone needed.
Install into either backbone env or a separate one. Requires `ffmpeg`.

```bash
pip install -r env/eval.txt
```

## Quick start

Precomputed steering vectors are shipped in `steering_vectors/`, so you can
synthesize steered speech immediately, no dataset or training needed.

### CosyVoice2

**Single-emotion:**
```bash
python scripts/synthesize.py \
    --backbone cosyvoice2 \
    --model-dir /path/to/CosyVoice2-0.5B \
    --steering-vector steering_vectors/cosyvoice2/angry_neutral_attn_output.pt \
    --text "I can not believe you did that." \
    --reference-audio examples/toy_audio/neutral.wav \
    --prompt-text "The octopus has eight legs." \
    --alpha 3.0 \
    --output-dir out/cosyvoice2/single
```

**Mixed-emotion** (composes per-sample vectors from soft labels via Eq. 7):
```bash
python scripts/synthesize.py \
    --backbone cosyvoice2 \
    --model-dir /path/to/CosyVoice2-0.5B \
    --steering-dir steering_vectors/cosyvoice2 \
    --manifest examples/toy_mixed_manifest.csv \
    --data-root examples/toy_audio \
    --alpha 3.0 \
    --output-dir out/cosyvoice2/mixed
```

### IndexTTS2

**Single-emotion:**
```bash
python scripts/synthesize.py \
    --backbone indextts2 \
    --model-dir /path/to/index-tts/checkpoints \
    --cfg-path /path/to/index-tts/checkpoints/config.yaml \
    --steering-vector steering_vectors/indextts2/angry_neutral_attn_output.pt \
    --text "I can not believe you did that." \
    --reference-audio examples/toy_audio/neutral.wav \
    --alpha 3.0 \
    --output-dir out/indextts2/single
```

**Mixed-emotion:**
```bash
python scripts/synthesize.py \
    --backbone indextts2 \
    --model-dir /path/to/index-tts/checkpoints \
    --cfg-path /path/to/index-tts/checkpoints/config.yaml \
    --steering-dir steering_vectors/indextts2 \
    --manifest examples/toy_mixed_manifest.csv \
    --data-root examples/toy_audio \
    --alpha 3.0 \
    --output-dir out/indextts2/mixed
```

**Evaluate** (per-sample: E-SIM, TEP, S-SIM, WER; aggregate: Spearman rho, H-Rate):
```bash
python scripts/evaluate.py \
    --wav-dir out/mixed \
    --baseline-wav-dir out/mixed_alpha0 \
    --manifest examples/toy_mixed_manifest.csv \
    --data-root examples/toy_audio \
    --output-dir out/evaluation
```

## Full pipeline

To reproduce everything from scratch (discriminability analysis, steering vector
extraction, synthesis, and evaluation), edit the config variables in
[`scripts/run_pipeline.sh`](scripts/run_pipeline.sh) and run:

```bash
bash scripts/run_pipeline.sh
```

The pipeline has four stages:

| Stage | Script | Input data | Output |
|-------|--------|------------|--------|
| 1. Discriminability | `discriminability.py` | Single-emotion (train/val) | `discriminability.json` |
| 2. Extraction | `extract.py` | Single-emotion (train only) | `*.pt` steering vectors |
| 3. Synthesis | `synthesize.py` | Mixed-emotion manifest | `*.wav` |
| 4. Evaluation | `evaluate.py` | Synthesized wavs | `summary.json` |

See [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) for manifest formats.

## Steering sites

| Backbone | Layers | Operator | Hidden dim | 
|----------|--------|----------|-----------|
| CosyVoice2 | 17, 14 | `attn_output` | 896 | 
| IndexTTS2 | 6, 8, 1 | `attn_output` | 1024 | 


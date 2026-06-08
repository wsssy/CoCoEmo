# Data Format

You need to prepare your own
data and organize it in the format described below. See
[`examples/toy_single_manifest.csv`](../examples/toy_single_manifest.csv) and
[`examples/toy_mixed_manifest.csv`](../examples/toy_mixed_manifest.csv) for
concrete examples.



## Directory layout

Place all datasets under a single root directory and pass it as `--data-root`.
Audio paths in your manifest should be relative to this root:

```
<DATA_ROOT>/
  CREMA-D/AudioWAV/...
  ESD/0001/Angry/...
  RAVDESS/Audio_Speech_Actors_01-24/...
  IEMOCAP_full_release/Session1/sentences/wav/...
```

## Manifest format (steering-vector extraction + single-emotion synthesis)

A CSV with the following columns:

| Column | Required | Description |
|--------|----------|-------------|
| `wav_filename` | yes | unique sample ID (used as output filename) |
| `filepath` | yes | path to the audio file (relative to `--data-root`) |
| `text` | yes | transcript of the audio |
| `emotion` | yes | emotion label: `happy`, `sad`, `angry`, `surprise`, `neutral` |
| `speaker` | yes | speaker ID (for speaker-disjoint splits) |
| `split` | yes | `train`, `val`, or `test` |
| `reference_filepath` | yes | path to a **neutral** reference audio from the **same speaker** |
| `reference_text` | yes | transcript of the reference audio (used as CosyVoice2 `prompt_text`) |

The steering-vector extraction computes mean activations over matched
emotion–neutral pairs (same speaker, same transcript when available), so each
emotional sample should have a corresponding neutral reference.

## Manifest format (mixed-emotion synthesis + evaluation)

For mixed-emotion experiments, the manifest additionally needs soft emotion
distributions derived from multi-rater annotations:

| Column | Required | Description |
|--------|----------|-------------|
| `dominant_emotion` | yes | the majority-vote emotion label |
| `p_happy`, `p_sad`, `p_angry`, `p_surprise`, `p_neutral` | yes | soft emotion proportions (sum to 1.0) |

These proportions are used as mixing weights for the steering vectors (Eq. 7)
and as ground truth for Spearman correlation and Dominant-Hit Rate evaluation.



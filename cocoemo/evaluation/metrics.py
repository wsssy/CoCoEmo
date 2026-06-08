"""Per-utterance evaluation metrics for CoCoEmo.

Backbone-agnostic: every function takes an already-loaded model and operates on
audio files, so this module is independent of CosyVoice2 / IndexTTS2 and runs in
its own (lightweight) evaluation environment.

Metrics:
    * Emotion2Vec emotion probabilities / Target Emotion Probability (TEP)
    * E-SIM  : cosine similarity of Emotion2Vec embeddings
    * WER    : Whisper-Large-V3 transcription word error rate
    * S-SIM  : WavLM speaker-embedding cosine similarity

Model loading is the caller's responsibility (see ``load_models`` helper at the
bottom) so that models are loaded once and reused across a run.
"""

import re
import unicodedata
from typing import Dict

import numpy as np


LABEL_MAP = {
    '生气/angry': 'angry',
    '厌恶/disgusted': 'disgust',
    '恐惧/fearful': 'fear',
    '开心/happy': 'happy',
    '中立/neutral': 'neutral',
    '其他/other': 'other',
    '难过/sad': 'sad',
    '吃惊/surprised': 'surprise',
    '<unk>': 'unk',
}


# --------------------------------------------------------------------------- #
#  Emotion2Vec: emotion probabilities (TEP) and embedding similarity (E-SIM)
# --------------------------------------------------------------------------- #
def get_emotion_probabilities(emotion_model, wav_path: str) -> Dict[str, float]:
    """Returns emotion probabilities with clean English labels."""
    rec_result = emotion_model.generate(
        wav_path,
        output_dir=None,
        granularity="utterance",
        extract_embedding=False,
    )

    result = rec_result[0]
    labels = result["labels"]
    scores = result["scores"]

    probs = {}
    for label, score in zip(labels, scores):
        if label in LABEL_MAP:
            probs[LABEL_MAP[label]] = float(score)
    max_emotion = max(probs.items(), key=lambda x: x[1])[0]

    return probs, max_emotion


def extract_emotion2vec_feats(emotion_model, wav_path: str) -> np.ndarray:
    """Returns utterance-level emotion2vec embedding ("feats") as a 1D numpy array."""
    rec_result = emotion_model.generate(
        wav_path,
        output_dir=None,
        granularity="utterance",
        extract_embedding=True,
    )

    feats = rec_result[0]["feats"]
    feats = np.asarray(feats, dtype=np.float32).reshape(-1)  # ensure 1D

    return feats


def cosine_similarity_wavs(emotion_model, wav1_path: str, wav2_path: str) -> float:
    """Cosine similarity between emotion2vec utterance embeddings of two wav files (E-SIM)."""
    emb1 = extract_emotion2vec_feats(emotion_model, wav1_path)
    emb2 = extract_emotion2vec_feats(emotion_model, wav2_path)

    emb1 = emb1 / (np.linalg.norm(emb1) + 1e-12)
    emb2 = emb2 / (np.linalg.norm(emb2) + 1e-12)

    return float(np.dot(emb1, emb2))


# --------------------------------------------------------------------------- #
#  Whisper ASR + WER
# --------------------------------------------------------------------------- #
import jiwer
from jiwer import Compose
from jiwer.transforms import (
    ToLowerCase,
    RemoveMultipleSpaces,
    Strip,
    ReduceToListOfListOfWords,
)
from word2num import word2num


def _map_if_list(x, fn):
    if isinstance(x, list):
        return [fn(t) for t in x]
    return fn(x)


class UnicodeNormalize:
    def __init__(self, form="NFKC"):
        self.form = form

    def __call__(self, x):
        return _map_if_list(x, lambda s: unicodedata.normalize(self.form, s))


class RemoveUnicodePunctuation:
    def __call__(self, x):
        def f(s: str) -> str:
            return "".join(
                " " if unicodedata.category(ch).startswith(("P", "S")) else ch
                for ch in s
            )
        return _map_if_list(x, f)


class NormalizeDigitNumbers:
    """Normalize digit-based numbers by removing thousand separators (commas, periods, spaces)."""

    def __call__(self, x):
        return _map_if_list(x, self._normalize_digit_numbers)

    def _normalize_digit_numbers(self, s: str) -> str:
        pattern = r'(\$|€|£|¥)?\s*(\d{1,3}(?:[,.\s]\d{3})+)(?:\.\d+)?'

        def clean_number(match):
            number_str = match.group(0)
            if re.search(r'\.\d{1,2}$', number_str):
                parts = number_str.rsplit('.', 1)
                integer_part = re.sub(r'[,.\s]', '', parts[0])
                return integer_part + '.' + parts[1]
            else:
                return re.sub(r'[,.\s]', '', number_str)

        return re.sub(pattern, clean_number, s)


class NormalizeNumbers:
    def __call__(self, x):
        return _map_if_list(x, self._convert_words_to_numbers)

    def _convert_words_to_numbers(self, s: str) -> str:
        number_pattern = r'\b(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|hundreds|thousand|thousands|million|millions|billion|billions)\b'

        single_digits = {'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'}
        scale_words = {'hundred', 'hundreds', 'thousand', 'thousands', 'million', 'millions', 'billion', 'billions'}
        compound_words = {'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
                          'seventeen', 'eighteen', 'nineteen', 'twenty', 'thirty', 'forty', 'fifty',
                          'sixty', 'seventy', 'eighty', 'ninety'}

        def try_convert_chunk(text):
            try:
                clean_text = text.strip().lower()
                clean_text = re.sub(r'\ba\s+hundred', 'one hundred', clean_text)
                clean_text = re.sub(r'\ba\s+thousand', 'one thousand', clean_text)
                clean_text = re.sub(r'\ba\s+million', 'one million', clean_text)
                num = word2num(clean_text)
                if isinstance(num, float) and num.is_integer():
                    return str(int(num))
                return str(num)
            except Exception:
                return text

        result = []
        last_end = 0
        matches = list(re.finditer(number_pattern, s, re.IGNORECASE))
        if not matches:
            return s

        i = 0
        while i < len(matches):
            match = matches[i]
            start = match.start()
            current_word = match.group(0).lower()

            if start > last_end:
                result.append(s[last_end:start])

            if current_word in single_digits:
                should_group = False
                if i + 1 < len(matches):
                    next_match = matches[i + 1]
                    gap = s[match.end():next_match.start()]
                    next_word = next_match.group(0).lower()
                    if next_word in scale_words and re.match(r'^[\s\-]*(and|a)?[\s\-]*$', gap, re.IGNORECASE):
                        should_group = True

                if should_group:
                    end = match.end()
                    j = i + 1
                    while j < len(matches):
                        next_match = matches[j]
                        gap = s[end:next_match.start()]
                        if re.match(r'^[\s\-]*(and|a)?[\s\-]*$', gap, re.IGNORECASE):
                            end = next_match.end()
                            j += 1
                        else:
                            break
                    chunk = s[start:end]
                    converted = try_convert_chunk(chunk)
                    result.append(converted)
                    last_end = end
                    i = j
                else:
                    converted = try_convert_chunk(current_word)
                    result.append(converted)
                    last_end = match.end()
                    i += 1
            else:
                end = match.end()
                j = i + 1
                while j < len(matches):
                    next_match = matches[j]
                    gap = s[end:next_match.start()]
                    if re.match(r'^[\s\-]*(and|a)?[\s\-]*$', gap, re.IGNORECASE):
                        end = next_match.end()
                        j += 1
                    else:
                        break
                chunk = s[start:end]
                converted = try_convert_chunk(chunk)
                result.append(converted)
                last_end = end
                i = j

        if last_end < len(s):
            result.append(s[last_end:])

        return ''.join(result)


# Build the normalization pipeline ONCE (do not recreate inside compute_wer).
NORM = Compose([
    UnicodeNormalize("NFKC"),
    ToLowerCase(),
    NormalizeDigitNumbers(),      # Normalize digit-based numbers first (10,000 -> 10000)
    NormalizeNumbers(),           # Then convert word-based numbers (ten thousand -> 10000)
    RemoveUnicodePunctuation(),
    RemoveMultipleSpaces(),
    Strip(),
    ReduceToListOfListOfWords(),
])


def whisper_asr(asr_model, wav_path: str, language: str | None = None) -> str:
    result = asr_model.transcribe(
        wav_path,
        language=language,
        task="transcribe",
        temperature=0.0,
    )
    return result["text"].strip()


def compute_wer(reference: str, hypothesis: str) -> float:
    return float(jiwer.wer(reference, hypothesis, NORM, NORM))


def whisper_wer_for_wav(asr_model, wav_path: str, reference_text: str, language: str | None = None) -> float:
    hyp = whisper_asr(asr_model, wav_path, language=language)
    return compute_wer(reference_text, hyp)


# --------------------------------------------------------------------------- #
#  WavLM speaker similarity (S-SIM)
# --------------------------------------------------------------------------- #
import torch
import torchaudio


def load_audio(path, target_sr=16000):
    wav, sr = torchaudio.load(path)      # [C, T]
    wav = wav.float()

    if wav.size(0) > 1:                  # mono
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:                  # resample
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav.squeeze(0).numpy()        # [T], float32


@torch.no_grad()
def speaker_similarity(feature_extractor, model, wav1_path, wav2_path):
    device = next(model.parameters()).device

    audio1 = load_audio(wav1_path, feature_extractor.sampling_rate)
    audio2 = load_audio(wav2_path, feature_extractor.sampling_rate)

    inputs = feature_extractor(
        [audio1, audio2],
        sampling_rate=feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs)
    embeddings = torch.nn.functional.normalize(outputs.embeddings, dim=-1)

    sim = torch.nn.functional.cosine_similarity(
        embeddings[0], embeddings[1], dim=0
    ).item()

    return sim


# --------------------------------------------------------------------------- #
#  Convenience loader (load each metric model once per run)
# --------------------------------------------------------------------------- #
def load_models(device=None, emotion_model_id="iic/emotion2vec_plus_large",
                whisper_size="large-v3", speaker_model_id="microsoft/wavlm-base-sv",
                hub="hf"):
    """Load the three metric models. Returns a dict of handles.

    Import-on-demand so that callers who only need a subset (e.g. WER) do not pay
    for the others.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    from funasr import AutoModel
    import whisper
    from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

    emotion_model = AutoModel(device=device, model=emotion_model_id, hub=hub)

    asr_model = whisper.load_model(whisper_size)
    asr_model.to(device)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(speaker_model_id)
    speaker_model = WavLMForXVector.from_pretrained(speaker_model_id).to(device).eval()

    return {
        "emotion_model": emotion_model,
        "asr_model": asr_model,
        "speaker_feature_extractor": feature_extractor,
        "speaker_model": speaker_model,
    }

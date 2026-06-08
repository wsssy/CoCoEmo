"""
Model extensions for CosyVoice2 to enable fine-grained emotion steering experiments.

This module provides helper functions and monkey-patching utilities to:
1. Extract speech tokens from the LLM stage without full synthesis
2. Extract speech tokens directly from audio with the tokenizer
3. Synthesize audio from pre-computed tokens with custom reference audio
4. Access intermediate representations for steering vector extraction

Usage:
    from cosyvoice_model_extensions import patch_cosyvoice_model
    
    cosyvoice = CosyVoice2("path/to/model")
    patch_cosyvoice_model(cosyvoice)
    
    # Now you can use extended methods:
    tokens = cosyvoice.extract_tokens_only(text, prompt_text, prompt_audio)
    audio = cosyvoice.synthesize_with_custom_reference(tokens, reference_audio)
    
    tokens_from_audio, _ = cosyvoice.extract_tokens_from_audio(prompt_audio)
    crossmodal_audio = cosyvoice.transfer_emotion_with_audio_tokens(
        prompt_audio, reference_audio
    )
"""

import os
import sys
import uuid
import torch
import torchaudio
from typing import Dict, Any

# The CosyVoice repo (https://github.com/FunAudioLLM/CosyVoice) must be importable.
# Point COSYVOICE_ROOT at your local clone; we add it and its bundled Matcha-TTS to
# sys.path. See README "Backbones" for setup.
_COSYVOICE_ROOT = os.environ.get("COSYVOICE_ROOT")
if _COSYVOICE_ROOT:
    sys.path.insert(0, _COSYVOICE_ROOT)
    sys.path.insert(0, os.path.join(_COSYVOICE_ROOT, "third_party", "Matcha-TTS"))


###############################################################################
##                    Extension Methods for CosyVoice2Model                  ##
###############################################################################


def synthesize_from_tokens(model, speech_tokens, flow_prompt_token, prompt_feat, 
                          flow_embedding, speed=1.0):
    """
    Synthesize audio from pre-computed speech tokens using flow-matching.
    
    This method takes pre-extracted speech tokens and runs only the flow-matching
    and vocoder stages, using potentially different reference audio features.
    
    Args:
        model: CosyVoice2Model instance
        speech_tokens: Pre-computed speech tokens from LLM
        flow_prompt_token: Speech tokens for flow reference
        prompt_feat: Mel-spectrogram features for reference
        flow_embedding: Speaker embedding for flow
        speed: Speech speed multiplier
        
    Returns:
        tts_speech: Generated waveform tensor
    """
    device = model.device
    
    with torch.no_grad():
        # Prepare inputs for flow-matching
        speech_tokens = speech_tokens.to(device)
        flow_prompt_token = flow_prompt_token.to(device)
        prompt_feat = prompt_feat.to(device)
        flow_embedding = flow_embedding.to(device)
        
        # Flow expects batch_size=1, ensure tokens have shape [1, seq_len]
        if speech_tokens.dim() == 1:
            speech_tokens = speech_tokens.unsqueeze(0)
        elif speech_tokens.shape[0] != 1:
            # If batch size > 1, take first sample only
            speech_tokens = speech_tokens[0:1]
            
        if flow_prompt_token.dim() == 1:
            flow_prompt_token = flow_prompt_token.unsqueeze(0)
        elif flow_prompt_token.shape[0] != 1:
            flow_prompt_token = flow_prompt_token[0:1]
        
        # Compute lengths after ensuring proper shape [1, seq_len]
        token_len = torch.tensor([speech_tokens.shape[1]], dtype=torch.int32).to(device)
        prompt_token_len = torch.tensor([flow_prompt_token.shape[1]], dtype=torch.int32).to(device)
        prompt_feat_len = torch.tensor([prompt_feat.shape[2]], dtype=torch.int32).to(device)
        
        # Run flow-matching to get mel-spectrogram
        # Note: streaming=False and finalize=True for batch inference
            
        tts_mel, _ = model.flow.inference(
            token=speech_tokens,  # Keep batch dimension [1, seq_len]
            token_len=token_len,
            prompt_token=flow_prompt_token,  # Keep batch dimension [1, seq_len]
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=flow_embedding,
            streaming=False,
            finalize=True
        )
        
        # Run vocoder to get waveform
        tts_speech, _ = model.hift.inference(
            speech_feat=tts_mel,
            cache_source=torch.zeros(1, 1, 0).to(device)
        )
    
    return tts_speech


###############################################################################
##                    Extension Methods for CosyVoice2                       ##
###############################################################################


def extract_tokens_from_audio(cosyvoice, speech_audio_16k):
    """
    Extract speech tokens directly from an audio clip using the tokenizer.
    
    This bypasses the text → token LLM stage entirely and instead uses the
    speech tokenizer on an existing recording. The resulting tokens can then
    be fed into the flow-matching stage with a different reference audio.
    
    Args:
        cosyvoice: CosyVoice2 instance
        speech_audio_16k: Source audio (mono, 16 kHz)
        
    Returns:
        speech_token: Extracted speech tokens
        speech_token_len: Corresponding length tensor
    """
    speech_token, speech_token_len = cosyvoice.frontend._extract_speech_token(speech_audio_16k)
    return speech_token, speech_token_len


def synthesize_with_custom_reference(cosyvoice, tokens, reference_audio_16k, 
                                     reference_text='', speed=1.0):
    """
    Synthesize from tokens with different reference audio.
    
    This enables cross-modal emotion transfer testing:
    - Extract tokens from one emotion
    - Synthesize with reference audio from different emotion
    
    Args:
        cosyvoice: CosyVoice2 instance
        tokens: Pre-computed speech tokens
        reference_audio_16k: Reference audio at 16kHz
        reference_text: Text of reference (optional, can be empty for cross-lingual)
        speed: Speech speed multiplier
        
    Returns:
        tts_speech: Generated waveform
    """
    # Extract features from reference audio
    prompt_speech_resample = torchaudio.transforms.Resample(
        orig_freq=16000, 
        new_freq=cosyvoice.sample_rate
    )(reference_audio_16k)
    
    speech_feat, speech_feat_len = cosyvoice.frontend._extract_speech_feat(prompt_speech_resample)
    speech_token, speech_token_len = cosyvoice.frontend._extract_speech_token(reference_audio_16k)
    embedding = cosyvoice.frontend._extract_spk_embedding(reference_audio_16k)
    
    # Synthesize using flow-matching with new reference
    tts_speech = synthesize_from_tokens(
        cosyvoice.model,
        speech_tokens=tokens,
        flow_prompt_token=speech_token,
        prompt_feat=speech_feat,
        flow_embedding=embedding,
        speed=speed
    )
    
    return tts_speech


def transfer_emotion_with_audio_tokens(cosyvoice, source_audio_16k, target_reference_audio_16k,
                                       reference_text='', speed=1.0):
    """
    Cross-modal transfer that starts from audio → tokens instead of text → tokens.
    
    Args:
        cosyvoice: CosyVoice2 instance
        source_audio_16k: Audio containing the emotion to analyze
        target_reference_audio_16k: Reference audio that provides prosody/timbre
        reference_text: Optional text metadata for the reference clip
        speed: Speech speed multiplier for synthesis
        
    Returns:
        tts_speech: Generated waveform using source tokens + target reference
    """
    speech_token, _ = cosyvoice.extract_tokens_from_audio(source_audio_16k)
    return cosyvoice.synthesize_with_custom_reference(
        speech_token, target_reference_audio_16k, reference_text=reference_text, speed=speed
    )


###############################################################################
##                          Monkey-Patching Utilities                        ##
###############################################################################

def patch_cosyvoice_model(cosyvoice):
    """
    Monkey-patch a CosyVoice2 instance with extension methods.
    
    This adds new methods to the model without modifying the source code.
    
    Args:
        cosyvoice: CosyVoice2 instance to patch
        
    Returns:
        cosyvoice: The same instance, now with additional methods
    """
    # Patch CosyVoice2 instance methods
    from functools import partial
    
    cosyvoice.extract_tokens_from_audio = partial(extract_tokens_from_audio, cosyvoice)
    cosyvoice.synthesize_with_custom_reference = partial(
        synthesize_with_custom_reference, cosyvoice
    )
    cosyvoice.transfer_emotion_with_audio_tokens = partial(
        transfer_emotion_with_audio_tokens, cosyvoice
    )
    
    # Patch CosyVoice2Model instance methods
    cosyvoice.model.synthesize_from_tokens = partial(
        synthesize_from_tokens, cosyvoice.model
    )
    
    print("✓ CosyVoice2 model patched with extension methods")
    print("  - cosyvoice.extract_tokens_only()")
    print("  - cosyvoice.extract_tokens_from_audio()")
    print("  - cosyvoice.synthesize_with_custom_reference()")
    print("  - cosyvoice.transfer_emotion_with_audio_tokens()")
    print("  - cosyvoice.model.extract_speech_tokens_from_text()")
    print("  - cosyvoice.model.synthesize_from_tokens()")
    
    return cosyvoice


###############################################################################
##                              Testing Utilities                             ##
###############################################################################

def test_cross_modal_synthesis(cosyvoice, text, prompt_text, 
                               base_audio_path, transfer_audio_path,
                               output_base_path, output_transfer_path,
                               output_crossmodal_path, transfer_text=None):
    """
    Test cross-modal emotion transfer.
    
    Args:
        cosyvoice: Patched CosyVoice2 instance
        text: Text to synthesize for base (used for base tokens extraction context)
        prompt_text: Prompt text for base synthesis
        base_audio_path: Path to base emotion audio (e.g., neutral)
        transfer_audio_path: Path to transfer emotion audio (e.g., happy)
        output_base_path: Where to save base synthesis
        output_transfer_path: Where to save normal transfer synthesis
        output_crossmodal_path: Where to save cross-modal synthesis
        transfer_text: Optional text for transfer audio (if different from base text).
                      If None, uses `text` for transfer synthesis.
        
    Returns:
        results: Dict with synthesis results
    """
    from cosyvoice.utils.file_utils import load_wav
    
    print("\n" + "="*80)
    print("CROSS-MODAL SYNTHESIS TEST")
    print("="*80)
    
    # Load audio files
    base_audio = load_wav(base_audio_path, 16000)
    transfer_audio = load_wav(transfer_audio_path, 16000)
    
    # 1. Extract tokens from BASE emotion
    print(f"\n1. Extracting tokens from base emotion...")
    print(f"   Text: {text}")
    print(f"   Base audio: {base_audio_path}")

    tokens, _ = cosyvoice.extract_tokens_from_audio(base_audio)

    print(f"   ✓ Extracted {tokens.shape[1]} tokens")
    
    # 2. Synthesize with BASE reference (baseline)
    print(f"\n2. Synthesizing with BASE reference (baseline)...")
    base_speech = cosyvoice.synthesize_with_custom_reference(
        tokens, base_audio, reference_text=prompt_text
    )
    torchaudio.save(output_base_path, base_speech.cpu(), cosyvoice.sample_rate)
    print(f"   ✓ Saved to {output_base_path}")
    
    # 3. Synthesize normally with TRANSFER reference (control)
    print(f"\n3. Synthesizing normally with TRANSFER reference (control)...")
    # Use transfer_text if provided, otherwise use base text
    transfer_synthesis_text = transfer_text if transfer_text is not None else text
    transfer_prompt_text = transfer_text if transfer_text is not None else prompt_text
    for i, output in enumerate(cosyvoice.inference_zero_shot(
        transfer_synthesis_text, transfer_prompt_text, transfer_audio, stream=False
    )):
        transfer_speech = output['tts_speech']
        torchaudio.save(output_transfer_path, transfer_speech.cpu(), cosyvoice.sample_rate)
        print(f"   ✓ Saved to {output_transfer_path}")
        break  # Only need first output
    
    # 4. CROSS-MODAL: Use BASE tokens with TRANSFER reference
    print(f"\n4. CROSS-MODAL: Using BASE tokens with TRANSFER reference...")
    print(f"   Transfer audio: {transfer_audio_path}")
    crossmodal_speech = cosyvoice.synthesize_with_custom_reference(
        tokens, transfer_audio, reference_text=prompt_text
    )
    torchaudio.save(output_crossmodal_path, crossmodal_speech.cpu(), cosyvoice.sample_rate)
    print(f"   ✓ Saved to {output_crossmodal_path}")
    
    print("\n" + "="*80)
    print("COMPARISON:")
    print("="*80)
    print(f"1. Base synthesis: {output_base_path}")
    print(f"   → Should sound like BASE emotion")
    print(f"2. Normal transfer: {output_transfer_path}")
    print(f"   → Should sound like TRANSFER emotion (gold standard)")
    print(f"3. Cross-modal: {output_crossmodal_path}")
    print(f"   → If sounds like TRANSFER emotion: Flow-matching controls emotion!")
    print(f"   → If sounds like BASE emotion: Emotion is in tokens!")
    print(f"   → If sounds mixed: Both stages contribute to emotion")
    print("="*80)
    
    return {
        'tokens': tokens,
        'base_speech': base_speech,
        'transfer_speech': transfer_speech,
        'crossmodal_speech': crossmodal_speech,
    }


###############################################################################
##                                  Example                                   ##
###############################################################################

if __name__ == "__main__":
    pass
''' 
steering.py : functions and classes used for creating and using steering vectors 

ADAPTED FOR INDEXTTS2:
- IndexTTS2 uses GPT2-style transformer for mel-code generation
- Model structure: IndexTTS2.gpt (UnifiedVoice) -> gpt (GPT2InferenceModel) -> gpt.h (transformer layers)
- Extraction uses pre-computed mel codes from audio + single forward pass (not generation)
- Hook paths adapted for GPT2 structure: gpt.h.{layer}.*

EMOTION EXTRACTION APPROACH:
- reference_audio_paths: Used for both speaker embedding + emotion embedding
- audio_paths: Converted to mel codes (speech tokens) via semantic codec
- Forward pass: text + speaker_emb + emotion_emb + mel_codes → GPT → embeddings
- This allows studying how different mel codes influence next token prediction
'''

import torch 
import re 
import tqdm
from functools import reduce
import sys
import os
import torchaudio

from cocoemo.discriminability import (
    compute_discriminability_for_steering,
    print_discriminability_report,
    find_best_layers_for_steering,
    HeadGeometry,
)

# Import helper functions (for backwards compatibility with original DISCO functions)
try:
    from utils import logit_extract, generate
except ImportError:
    # Define stubs if utils not available
    def generate(model, tokenizer, dataloader, max_new_tokens=512):
        raise NotImplementedError("generate() requires utils.py")
    
    def logit_extract(dataloader, model, tokenizer, position=-1):
        raise NotImplementedError("logit_extract() requires utils.py")

###########################################################################
##                         Steering Operations                           ##
###########################################################################

def translation_op_(x, t, alpha):
    ''' translates representation inplace '''

    t = t.to(device=x.device, dtype=x.dtype)
    x.add_(alpha * t)
    return x

def norm_preserve_steer_op_(x, t, alpha, eps=1e-8):
    ''' applies norm-preserving steering operation inplace '''
    # x: (..., d) hidden state tensor to modify in-place
    # t: (d,) or (..., d) steering vector broadcastable to x
    # alpha: steering strength
    # eps: small epsilon for numerical stability

    t = t.to(device=x.device, dtype=x.dtype)
    h_prime = x + alpha * t
    h_norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
    hp_norm = torch.linalg.norm(h_prime, ord=2, dim=-1, keepdim=True)
    h_tilde = h_prime * (h_norm / (hp_norm + eps))
    
    # Modify x in-place to match translation_op_ behavior
    x.copy_(h_tilde)
    return x

###########################################################################
##                       Steering Vector Creation                        ##
###########################################################################

class PreHookSave:
    ''' saves representations input to function''' 

    def __init__(self):
        self.saved_values        = None  

    def __call__(self, module, input):
        self.saved_values = input[0][:,-1,:].cpu()  # Agglomeration step takes the final token representation for each piece of text i.e. [B,T,D] --> [B,d] (batch tokenization left padding important for this!)
        return input  # returns unmodified input to downstream computation in model fwd pass
        
class ForwardHookSave:
    ''' saves representations output of functions ''' 

    def __init__(self):
        self.saved_values        = None  

    def __call__(self, module, input, output):
        if isinstance(output, tuple): # in some models (e.g., llama 3.1, gemma 2) the output of the layer is a tuple with the first element equal to the representations
            self.saved_values = output[0][:,-1,:].cpu() 
        else:
            self.saved_values = output[:,-1,:].cpu()

        return output  # returns unmodified output to downstream computation in model fwd pass

def norm_preserve_steer(h, s, alpha, eps=1e-8):
    # h: (..., d) hidden state
    # s: (d,) or (..., d) steering vector broadcastable to h
    h_prime = h + alpha * s
    h_norm  = torch.linalg.norm(h, ord=2, dim=-1, keepdim=True)
    hp_norm = torch.linalg.norm(h_prime, ord=2, dim=-1, keepdim=True)
    h_tilde = h_prime * (h_norm / (hp_norm + eps))
    return h_tilde

    
def hook_representations_for_saving(model, op_dict):
    ''' hooks all representation spaces of interest so that forward passes will populate their saved representations for steering vector creation ''' 

    hooks, handles = {}, {} 

    for layer_id, _ in enumerate(model.llm.llm.model.model.layers): # NOTE: CosyVoice2/Qwen2LM uses "model.llm.llm.model.model.layers" format (model -> Qwen2LM -> Qwen2Encoder -> Qwen2ForCausalLM -> Qwen2Model -> layers) 
        hooks[layer_id], handles[layer_id] = {}, {}

        for op in op_dict.keys():
            assert op_dict[op]['hook type'] in ['forward', 'forward_pre']; "error hook type must be forward or forward_pre"

            op_exact_name = op_dict[op]['module'].format(layer = str(layer_id))
            module        = reduce(getattr, op_exact_name.split("."), model)
            if op_dict[op]['hook type'] == 'forward':
                hook                  = ForwardHookSave()
                handle                = module.register_forward_hook(hook)
            elif op_dict[op]['hook type'] == 'forward_pre':
                hook                  = PreHookSave()
                handle                = module.register_forward_pre_hook(hook)
            hooks[layer_id][op]   = hook
            handles[layer_id][op] = handle

    return hooks, handles 

def extract_with_hooks(model, operations, layers, dataloader, tokenizer, op_dict):
    ''' extracts last token representations of desired operations of all texts under the model '''

    # extraction
    hooks, handles = hook_representations_for_saving(model, op_dict) # hook
    op_to_layer_to_rep = {op : {layer : [] for layer in range(layers)} for op in operations}

    for batch in tqdm.tqdm(dataloader, desc = "Representation Extraction"):
        inputs = tokenizer(batch, return_tensors = 'pt', padding = True) 
        with torch.no_grad():
            model(inputs['input_ids'].cuda(), attention_mask=inputs["attention_mask"].cuda()) # populate 
            for layer in range(layers):  # Iterate and save
                for op in operations:
                    op_to_layer_to_rep[op][layer].append(hooks[layer][op].saved_values)
    unhook(handles)  # unhook

    # for each op + layer combo, list of tensors --> tensor
    for op in op_to_layer_to_rep:
        for layer in range(layers):
            op_to_layer_to_rep[op][layer] = torch.vstack(op_to_layer_to_rep[op][layer])

    return op_to_layer_to_rep

def extract_pos_neg(data, model, operations, tokenizer, op_dict, layers = 24, split = "train"):
    ''' extracts positive and negative representation datasets for all operations to create steering vectors for (default layers=24 for Qwen2LM) '''

    op_dict = {op : op_dict[op] for op in operations} # avoid saving representations we don't need
    op_to_layer_to_rep_pos = extract_with_hooks(model, operations, layers, data[split]['dataloader_pos'], tokenizer, op_dict)
    op_to_layer_to_rep_neg = extract_with_hooks(model, operations, layers, data[split]['dataloader_neg'], tokenizer, op_dict)

    return op_to_layer_to_rep_pos, op_to_layer_to_rep_neg


def create_mean_difference(operations, pos_representations, neg_representations):
    ''' given positive and negative representations, creates mean difference vectors for all representation spaces of interest ''' 

    op_to_layer_to_meandiff = {op : {} for op in operations}
    for op in operations:
        temp_pos, temp_neg = pos_representations[op], neg_representations[op]

        for layer in temp_pos.keys():
            pos_repr, neg_repr = temp_pos[layer], temp_neg[layer]
            meandiff = (pos_repr.mean(dim = 0, keepdim = True) - neg_repr.mean(dim = 0, keepdim = True))
            op_to_layer_to_meandiff[op][layer] = meandiff

    return op_to_layer_to_meandiff

###########################################################################
##              IndexTTS2-Specific Extraction Functions                  ##
###########################################################################

def hook_representations_for_saving_indextts2(model, op_dict):
    ''' 
    Hooks representation spaces for IndexTTS2's GPT (UnifiedVoice.gpt).
    
    Args:
        model: UnifiedVoice model (tts.gpt)
        op_dict: Dictionary defining hook locations for GPT2-style layers
        
    Returns:
        hooks: Dict {layer: {op: hook}}
        handles: Dict {layer: {op: handle}}
    '''
    hooks, handles = {}, {}

    for layer_id, _ in enumerate(model.gpt.h):  # GPT2 uses .h for transformer layers
        hooks[layer_id], handles[layer_id] = {}, {}

        for op in op_dict.keys():
            assert op_dict[op]['hook type'] in ['forward', 'forward_pre'], "error: hook type must be forward or forward_pre"

            op_exact_name = op_dict[op]['module'].format(layer=str(layer_id))
            module = reduce(getattr, op_exact_name.split("."), model)
            
            if op_dict[op]['hook type'] == 'forward':
                hook = ForwardHookSave()
                handle = module.register_forward_hook(hook)
            elif op_dict[op]['hook type'] == 'forward_pre':
                hook = PreHookSave()
                handle = module.register_forward_pre_hook(hook)
                
            hooks[layer_id][op] = hook
            handles[layer_id][op] = handle

    return hooks, handles


def extract_with_hooks_audio_indextts2(tts, operations, layers, audio_paths, reference_audio_paths, texts, op_dict,
                                       sample_rate=16000, max_mel_tokens=None):
    '''
    Extract representations from IndexTTS2 GPT during SINGLE FORWARD PASS with pre-computed mel codes.
    
    IMPORTANT: This does NOT do generation! It:
    1. Extracts speaker + emotion embeddings from reference_audio_paths
    2. Converts audio_paths to mel codes (speech tokens) using semantic codec
    3. Does ONE batched forward pass through GPT with all inputs
    4. Extracts embeddings during this forward pass
    
    This approach allows studying how different mel codes (from different emotions) 
    influence the GPT's internal representations when predicting the next token.
    
    Args:
        tts: IndexTTS2 instance
        operations: List of operation names to extract (e.g., ['attn_output', 'layer_output'])
        layers: Number of GPT layers
        audio_paths: List of audio file paths containing EMOTIONAL SPEECH (converted to mel codes)
        reference_audio_paths: List of reference audio for speaker + emotion conditioning
        texts: List of text strings (TTS text)
        op_dict: Dictionary defining hook locations (IndexTTS2 GPT paths)
        sample_rate: Audio sample rate (default 16000)
        max_mel_tokens: Maximum number of mel tokens to extract from audio
        batch_size: Number of samples to process in parallel (default 8)
        
    Returns:
        op_to_layer_to_rep: Nested dict {operation: {layer: tensor}}
        
    Example:
        >>> op_dict = get_indextts2_op_dict(['attn_output', 'layer_output'])
        >>> embeddings = extract_with_hooks_audio_indextts2(
        ...     tts=tts,
        ...     operations=['attn_output', 'layer_output'],
        ...     layers=24,
        ...     audio_paths=happy_audio_paths,  # Emotional speech
        ...     reference_audio_paths=neutral_refs,  # Speaker reference
        ...     texts=texts,
        ...     op_dict=op_dict,
        ...     batch_size=8
        ... )
    '''
    num_layers = min(layers, len(tts.gpt.gpt.h))
    if layers != num_layers:
        print(f"Warning: Requested {layers} layers, using {num_layers} available GPT layers")

    # Hook model (UnifiedVoice)
    hooks, handles = hook_representations_for_saving_indextts2(tts.gpt, op_dict)
    op_to_layer_to_rep = {op: {layer: [] for layer in range(num_layers)} for op in operations}

    num_samples = len(audio_paths)
    num_mel_codes = getattr(tts.gpt, 'number_mel_codes', 8194)
    model_max_mel = getattr(tts.gpt, 'max_mel_tokens', 1815)
    
    # Process one sample at a time (no batching)
    for i in tqdm.tqdm(range(num_samples), desc="Extracting representations (IndexTTS2)"):
        try:
            audio_path = audio_paths[i]
            ref_audio_path = reference_audio_paths[i]
            text = texts[i]

            # STEP 1: Extract speaker embedding from reference audio
            ref_audio, sr = tts._load_and_cut_audio(ref_audio_path, 15)
            ref_audio_16k = torchaudio.transforms.Resample(sr, 16000)(ref_audio)
            ref_inputs = tts.extract_features(ref_audio_16k, sampling_rate=sample_rate, return_tensors="pt")
            ref_input_features = ref_inputs["input_features"].to(tts.device)
            ref_attention_mask = ref_inputs["attention_mask"].to(tts.device)
            spk_cond_emb = tts.get_emb(ref_input_features, ref_attention_mask)  # (1, T, 1024)

            # STEP 2: Convert emotional audio to mel codes
            emo_audio, sr = tts._load_and_cut_audio(audio_path, 15)
            emo_audio_16k = torchaudio.transforms.Resample(sr, 16000)(emo_audio)
            emo_inputs = tts.extract_features(emo_audio_16k, sampling_rate=sample_rate, return_tensors="pt")
            emo_input_features = emo_inputs["input_features"].to(tts.device)
            emo_attention_mask = emo_inputs["attention_mask"].to(tts.device)
            emo_embeddings = tts.get_emb(emo_input_features, emo_attention_mask)
            mel_codes, _ = tts.semantic_codec.quantize(emo_embeddings)

            if mel_codes.ndim == 1:
                mel_codes = mel_codes.unsqueeze(0)
            mel_codes = mel_codes.long().clamp(0, num_mel_codes - 1)

            if mel_codes.shape[1] > model_max_mel:
                print(f"Warning: mel_codes ({mel_codes.shape[1]}) exceeds max_mel_tokens ({model_max_mel})")

            # STEP 3: Prepare text inputs
            text_tokens_list = tts.tokenizer.tokenize(text)
            text_token_ids = tts.tokenizer.convert_tokens_to_ids(text_tokens_list)
            text_tokens = torch.tensor(text_token_ids, dtype=torch.long, device=tts.device).unsqueeze(0)

            # STEP 4: Prepare conditioning
            cond_lengths = torch.tensor([spk_cond_emb.shape[1]], device=tts.device)
            speech_conditioning_latent = tts.gpt.get_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
            emo_vec = tts.gpt.get_emo_conditioning(spk_cond_emb.transpose(1, 2), cond_lengths)
            emo_vec = tts.gpt.emovec_layer(emo_vec)
            emo_vec = tts.gpt.emo_layer(emo_vec)

            # Build conditioning latent
            tmp = torch.zeros(1).to(tts.device)
            duration_emb = tts.gpt.speed_emb(torch.zeros_like(tmp).long())
            duration_emb_half = tts.gpt.speed_emb(torch.ones_like(tmp).long())
            conds_latent = torch.cat((speech_conditioning_latent + emo_vec.unsqueeze(1),
                                      duration_emb_half.unsqueeze(1),
                                      duration_emb.unsqueeze(1)), 1)

            # Prepare GPT inputs
            input_ids, inputs_embeds, _ = tts.gpt.prepare_gpt_inputs(conds_latent, text_tokens)

            # Concatenate mel codes
            inputs = torch.cat([input_ids, mel_codes], dim=1)
            attention_mask = torch.ones_like(inputs, dtype=torch.long, device=tts.device)

            # STEP 5: Forward pass
            tts.gpt.inference_model.store_mel_emb(inputs_embeds)
            with torch.no_grad():
                _ = tts.gpt.inference_model(
                    input_ids=inputs,
                    attention_mask=attention_mask,
                    return_dict=True
                )

            # STEP 6: Collect saved representations from hooks
            for layer in range(num_layers):
                for op in operations:
                    saved = hooks[layer][op].saved_values
                    if saved is not None:
                        op_to_layer_to_rep[op][layer].append(saved)

        except Exception as e:
            import traceback
            print(f"Error processing sample {i}: {e}")
            traceback.print_exc()
            continue

    # Remove hooks
    unhook(handles)

    # Convert lists to tensors
    for op in op_to_layer_to_rep:
        for layer in range(num_layers):
            if len(op_to_layer_to_rep[op][layer]) > 0:
                op_to_layer_to_rep[op][layer] = torch.vstack(op_to_layer_to_rep[op][layer])
            else:
                print(f"Warning: No data collected for {op} at layer {layer}")
                op_to_layer_to_rep[op][layer] = torch.empty(0)

    return op_to_layer_to_rep


###########################################################################
##              CosyVoice-Specific Extraction Functions                  ##
###########################################################################

def extract_with_hooks_audio(model, operations, layers, audio_paths, texts, frontend, op_dict, 
                              sample_rate=16000, use_zero_shot=True, pool_method='mean',
                              max_generation_steps=None):
    ''' 
    Extracts representations from CosyVoice LLM during audio synthesis.
    
    This function is CosyVoice-specific and handles audio+text input processing.
    
    Args:
        model: CosyVoice2Model instance (not CosyVoice2, but the actual model)
        operations: List of operation names to extract (e.g., ['residual'])
        layers: Number of LLM layers (24 for Qwen2)
        audio_paths: List of audio file paths (prompt speech)
        texts: List of text strings (TTS text)
        frontend: CosyVoice frontend for preprocessing
        op_dict: Dictionary defining hook locations
        sample_rate: Audio sample rate (default 16000)
        use_zero_shot: Whether to use zero-shot mode (True) or cross-lingual (False)
        pool_method: How to pool sequence representations ('mean', 'last', or 'first')
        max_generation_steps: Optional cap on how many tokens to decode per sample
            when forcing the LLM generator to run. None = exhaust the generator.
    
    Returns:
        op_to_layer_to_rep: Nested dict {operation: {layer: tensor}}
    '''
    if load_wav is None:
        raise ImportError("load_wav not available. Make sure CosyVoice is installed.")
    
    # Hook model
    hooks, handles = hook_representations_for_saving(model, op_dict)
    op_to_layer_to_rep = {op : {layer : [] for layer in range(layers)} for op in operations}
    
    # Process each audio-text pair
    for audio_path, text in tqdm.tqdm(zip(audio_paths, texts), total=len(audio_paths), 
                                      desc="Extracting representations"):
        try:
            # Load audio
            prompt_speech_16k = load_wav(audio_path, sample_rate)
            
            # Get model inputs using frontend
            # CosyVoice2 uses resample_rate=24000 and zero_shot_spk_id='' for zero-shot
            resample_rate = 24000  # Default for CosyVoice2
            zero_shot_spk_id = ''  # Empty string for true zero-shot
            
            # if use_zero_shot:
                # Use zero-shot mode preprocessing
            # prompt_text = text[:min(len(text), 50)]  # Use part of text as prompt
            prompt_text = text
            model_input = frontend.frontend_zero_shot(
                tts_text=text,
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_speech_16k,
                resample_rate=resample_rate,
                zero_shot_spk_id=zero_shot_spk_id
            )
            # model_input['prompt_text']=torch.zeros(1, 0, dtype=torch.int32)
            # model_input['prompt_text_len']=torch.zeros(1, dtype=torch.int32)
            # else:
            #     # Use cross-lingual mode preprocessing
            #     model_input = frontend.frontend_cross_lingual(
            #         tts_text=text,
            #         prompt_speech_16k=prompt_speech_16k,
            #         resample_rate=resample_rate,
            #         zero_shot_spk_id=zero_shot_spk_id
            #     )
            #     model_input['prompt_text']=torch.zeros(1, 0, dtype=torch.int32)
            #     model_input['prompt_text_len']=torch.zeros(1, dtype=torch.int32)
            
            # Forward pass through LLM only (not the full pipeline)
            # The hooks will capture representations during this forward pass
            with torch.no_grad():
                token_generator = model.llm.inference(
                    text=model_input['text'].cuda(),
                    text_len=model_input['text_len'].cuda(),
                    prompt_text=model_input['prompt_text'].cuda(),
                    prompt_text_len=model_input['prompt_text_len'].cuda(),
                    prompt_speech_token=model_input['llm_prompt_speech_token'].cuda(),
                    prompt_speech_token_len=model_input['llm_prompt_speech_token_len'].cuda(),
                    embedding=model_input['llm_embedding'].cuda()
                )
                tokens_emitted = 0
                try:
                    for _ in token_generator:
                        tokens_emitted += 1
                        if max_generation_steps is not None and tokens_emitted >= max_generation_steps:
                            break
                finally:
                    # Ensure generator resources are released if we break early.
                    try:
                        token_generator.close()
                    except Exception:
                        pass
            
            # Collect saved representations from hooks
            for layer in range(layers):
                for op in operations:
                    saved = hooks[layer][op].saved_values
                    
                    # Apply pooling if needed (hooks save last token by default)
                    # If you want different pooling, modify PreHookSave/ForwardHookSave
                    if saved is not None:
                        op_to_layer_to_rep[op][layer].append(saved)
                    else:
                        print(f"Warning: No data collected for {op} at layer {layer}")
        
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            # Append None or skip - for now skip
            continue
    
    # Remove hooks
    unhook(handles)
    
    # Convert lists to tensors
    for op in op_to_layer_to_rep:
        for layer in range(layers):
            if len(op_to_layer_to_rep[op][layer]) > 0:
                op_to_layer_to_rep[op][layer] = torch.vstack(op_to_layer_to_rep[op][layer])
            else:
                # No data collected for this layer/op
                print(f"Warning: No data collected for {op} at layer {layer}")
                op_to_layer_to_rep[op][layer] = torch.empty(0)
    
    return op_to_layer_to_rep


def extract_pos_neg_audio(data, model, operations, frontend, op_dict, layers=24, 
                          split='train', sample_rate=16000, use_zero_shot=True):
    ''' 
    Extracts positive and negative representation datasets from audio for emotion steering.
    
    This is the CosyVoice equivalent of extract_pos_neg() from DISCO.
    
    Args:
        data: Data dict from prepare_esd_emotion_steering() with structure:
              {split: {'pos_audio_paths': [...], 'neg_audio_paths': [...], 'texts': [...]}}
        model: CosyVoice2Model instance
        operations: List of operations to extract
        frontend: CosyVoice frontend
        op_dict: Operation dictionary defining hook locations
        layers: Number of layers (default 24 for Qwen2)
        split: Data split to use ('train', 'val', or 'test')
        sample_rate: Audio sample rate
        use_zero_shot: Whether to use zero-shot mode
    
    Returns:
        op_to_layer_to_rep_pos: Positive emotion representations
        op_to_layer_to_rep_neg: Negative emotion representations
    '''
    # Filter op_dict to only requested operations
    op_dict = {op : op_dict[op] for op in operations}
    
    # Extract positive representations
    print(f"\nExtracting POSITIVE emotion representations from {split} set...")
    pos_audio_paths = data[split]['pos_audio_paths']
    texts = data[split]['texts']
    
    op_to_layer_to_rep_pos = extract_with_hooks_audio(
        model=model,
        operations=operations,
        layers=layers,
        audio_paths=pos_audio_paths,
        texts=texts,
        frontend=frontend,
        op_dict=op_dict,
        sample_rate=sample_rate,
        use_zero_shot=use_zero_shot
    )
    
    # Extract negative representations
    print(f"\nExtracting NEGATIVE emotion representations from {split} set...")
    neg_audio_paths = data[split]['neg_audio_paths']
    
    op_to_layer_to_rep_neg = extract_with_hooks_audio(
        model=model,
        operations=operations,
        layers=layers,
        audio_paths=neg_audio_paths,
        texts=texts,  # Same texts, different emotion audio
        frontend=frontend,
        op_dict=op_dict,
        sample_rate=sample_rate,
        use_zero_shot=use_zero_shot
    )
    
    return op_to_layer_to_rep_pos, op_to_layer_to_rep_neg


def create_steering_vectors_from_audio_data(data, model, frontend, operations, op_dict, 
                                             layers=24, split='train'):
    '''
    Complete pipeline to create steering vectors from audio data.
    
    This is a convenience function that combines extraction + mean difference calculation.
    
    Args:
        data: Data dict from prepare_esd_emotion_steering()
        model: CosyVoice2Model instance
        frontend: CosyVoice frontend
        operations: List of operations to extract
        op_dict: Operation dictionary
        layers: Number of layers
        split: Which split to use for creating vectors
    
    Returns:
        steering_vectors: Dict {operation: {layer: mean_difference_vector}}
        pos_reps: Positive representations (for analysis)
        neg_reps: Negative representations (for analysis)
    '''
    # Extract representations
    pos_reps, neg_reps = extract_pos_neg_audio(
        data=data,
        model=model,
        operations=operations,
        frontend=frontend,
        op_dict=op_dict,
        layers=layers,
        split=split
    )
    
    # Create steering vectors
    steering_vectors = create_mean_difference(
        operations=operations,
        pos_representations=pos_reps,
        neg_representations=neg_reps
    )
    
    return steering_vectors, pos_reps, neg_reps

###########################################################################
##                       Steering Vector Injection                       ##
###########################################################################

class PreHookInject:
    ''' injects steering vector into input of a function ''' 

    def __init__(self, inject_op = None, inject_dict = {}):
        self.inject_op           = inject_op 
        self.inject_dict         = inject_dict  

    def __call__(self, module, input):
        self.inject_op(input[0], **self.inject_dict)  # input is tuple with first element being representations 
        return input  # returns modified input 

class ForwardHookInject:
    ''' injects steering vector into output of a function ''' 

    def __init__(self, inject_op = None, inject_dict = {}):
        self.inject_op           = inject_op 
        self.inject_dict         = inject_dict  

    def __call__(self, module, input, output):
        if isinstance(output, tuple): # in some models (e.g., llama 3.1, gemma 2) the output of the layer is a tuple with the first element equal to the representations
            self.inject_op(output[0], **self.inject_dict)  
        else:
            self.inject_op(output, **self.inject_dict)

        return output  # returns modified output


def hook_model_inject(model, operations_to_hook_info):
    ''' registers injections hooks for the model for desired operations/layers '''

    hooks   = {}
    handles = {}

    for op in operations_to_hook_info:
        temp = operations_to_hook_info[op]
        if temp['inject'] == True:
            hooks[op], handles[op] = {},{}

            for layer in temp['layer_to_inject'].keys(): # for every layer we are modifying
                assert temp['hook type'] in ['forward_pre', 'forward']; "error in hook type provided, must be forward_pre or forward"
                temp_layer    = temp['layer_to_inject'][layer]

                op_exact_name = temp['module'].format(layer = str(layer))
                module        = reduce(getattr, op_exact_name.split("."), model)

                if temp['hook type'] == "forward_pre":
                    hook          = PreHookInject(inject_op = temp_layer['inject_op'], inject_dict = temp_layer['inject_dict'])
                    handle        = module.register_forward_pre_hook(hook)
                else:
                    hook          = ForwardHookInject(inject_op = temp_layer['inject_op'], inject_dict = temp_layer['inject_dict'])
                    handle        = module.register_forward_hook(hook)
                
                hooks[op][layer]    = hook
                handles[op][layer]  = handle

    return hooks, handles

def generate_with_hooks(model, tokenizer, operations, dataloader, op_to_layer_to_meandiff, inject_op, max_new_tokens = 512, operations_to_hook_info = None, layers = None, alpha = None):
    ''' generates model responses with steering '''
    
    operations_to_hook_info = update_operations_to_hook_info(operations_to_hook_info, operations, layers, alpha, inject_op, op_to_layer_to_meandiff) # populates information about where (operations) and how much (magnitude) we want to steer
    hooks, handles             = hook_model_inject(model, operations_to_hook_info) # hook
    results = generate(model, tokenizer, dataloader, max_new_tokens) # generate steered responses
    unhook(handles) # unhook
    return results 

def logit_extract_with_hooks(op_to_layer_to_meandiff, dataloader, model, tokenizer, operations, position = -1, operations_to_hook_info = None, inject_op = translation_op_, layers = 24, alpha = None):
    ''' extracts the logit vectors from the steered model for next token prediction (default position to grab logits from = -1) for all texts in the dataloader (used for multiple choice grading, default layers=24 for Qwen2LM) '''
    
    operations_to_hook_info = update_operations_to_hook_info(operations_to_hook_info, operations, layers, alpha, inject_op, op_to_layer_to_meandiff) # populates information about where (operations) and how much (magnitude) we want to steer
    hooks, handles          = hook_model_inject(model, operations_to_hook_info)     # hook
    logits                  = logit_extract(dataloader, model, tokenizer, position) # extract
    unhook(handles) # remove hooks

    return logits

###########################################################################
##                                 Helper                                ##
########################################################################### 

def unhook(handles):
    ''' removes hooks at every operation and layer in handles'''

    for op in handles.keys(): # unhook
        for layer in handles[op].keys():
            handles[op][layer].remove()

def update_operations_to_hook_info(operations_to_hook_info, operations, layers, alpha, inject_op, op_to_layer_to_meandiff):
    ''' populates information about where (operations) and how much (magnitude) we want to steer ''' 

    operations_to_hook_info = {op : operations_to_hook_info[op] for op in operations}     # throw away operations we are not steering

    for op in operations:
        operations_to_hook_info[op]['inject']          = True
        operations_to_hook_info[op]['layer_to_inject'] = {layer : {'inject_op' : inject_op, 'inject_dict' : {"t" : op_to_layer_to_meandiff[op][layer].cuda(), "alpha" : alpha[op]} } for layer in layers}
    return operations_to_hook_info

###########################################################################
##                    CosyVoice Operation Dictionaries                   ##
###########################################################################

# Standard operation dictionaries for CosyVoice LLM (Qwen2)
COSYVOICE_OP_DICTS = {
    'emb_pre_attn_post_ln': {
        'module': 'llm.llm.model.model.layers.{layer}.input_layernorm',
        'hook type': 'forward'
    },
    'q_proj': {
        'module': 'llm.llm.model.model.layers.{layer}.self_attn.q_proj',
        'hook type': 'forward'
    },
    'k_proj': {
        'module': 'llm.llm.model.model.layers.{layer}.self_attn.k_proj',
        'hook type': 'forward'
    },
    'v_proj': {
        'module': 'llm.llm.model.model.layers.{layer}.self_attn.v_proj',
        'hook type': 'forward'
    },
    'attn_output': {
        'module': 'llm.llm.model.model.layers.{layer}.self_attn.o_proj',
        'hook type': 'forward_pre'
    },
    'W0_x_attn_output': {
        'module': 'llm.llm.model.model.layers.{layer}.self_attn.o_proj',
        'hook type': 'forward'
    },
    'emb_post_attn_pre_ln': {
        'module': 'llm.llm.model.model.layers.{layer}.post_attention_layernorm',
        'hook type': 'forward_pre'
    },
    'emb_post_attn_post_ln': {
        'module': 'llm.llm.model.model.layers.{layer}.post_attention_layernorm',
        'hook type': 'forward'
    },
    'emb_post_mlp_residual': {
        'module': 'llm.llm.model.model.layers.{layer}.mlp',
        'hook type': 'forward'
    },
    'layer_output': {
        'module': 'llm.llm.model.model.layers.{layer}',
        'hook type': 'forward'
    },
}

def get_cosyvoice_op_dict(operations):
    '''
    Returns operation dictionary for specified operations in CosyVoice.
    
    Args:
        operations: List of operation names (e.g., ['residual', 'attn_output'])
    
    Returns:
        op_dict: Dictionary mapping operations to module paths
    '''
    return {op: COSYVOICE_OP_DICTS[op] for op in operations if op in COSYVOICE_OP_DICTS}


###########################################################################
##                    IndexTTS2 Operation Dictionaries                   ##
###########################################################################

# Standard operation dictionaries for IndexTTS2 GPT (GPT2-style)
# Note: IndexTTS2 uses UnifiedVoice.gpt (GPT2InferenceModel) with layers in gpt.h
INDEXTTS2_OP_DICTS = {
    'emb_pre_attn_post_ln': {
        'module': 'gpt.h.{layer}.ln_1',  # Layer norm before attention
        'hook type': 'forward'
    },
    'qkv_proj': {
        'module': 'gpt.h.{layer}.attn.c_attn',  # Combined Q, K, V projection (GPT2 style)
        'hook type': 'forward'
    },
    'attn_output': {
        'module': 'gpt.h.{layer}.attn.c_proj',  # Attention output projection (before applying)
        'hook type': 'forward_pre'
    },
    'W0_x_attn_output': {
        'module': 'gpt.h.{layer}.attn.c_proj',  # Attention output projection (after applying)
        'hook type': 'forward'
    },
    'emb_post_attn_pre_ln': {
        'module': 'gpt.h.{layer}.ln_2',  # Layer norm after attention (before applying)
        'hook type': 'forward_pre'
    },
    'emb_post_attn_post_ln': {
        'module': 'gpt.h.{layer}.ln_2',  # Layer norm after attention (after applying)
        'hook type': 'forward'
    },
    'emb_post_mlp_residual': {
        'module': 'gpt.h.{layer}.mlp',  # MLP output
        'hook type': 'forward'
    },
    'layer_output': {
        'module': 'gpt.h.{layer}',  # Full transformer layer output
        'hook type': 'forward'
    },
}


def get_indextts2_op_dict(operations):
    '''
    Returns operation dictionary for specified operations in IndexTTS2 (GPT2-style).
    
    Args:
        operations: List of operation names (e.g., ['attn_output', 'layer_output'])
    
    Returns:
        op_dict: Dictionary mapping operations to module paths
        
    Example:
        >>> op_dict = get_indextts2_op_dict(['attn_output', 'layer_output'])
        >>> # Returns hooks for attention output and full layer output
    '''
    return {op: INDEXTTS2_OP_DICTS[op] for op in operations if op in INDEXTTS2_OP_DICTS}


###########################################################################
##                        Utility Functions                              ##
###########################################################################

def save_steering_vectors(steering_vectors, save_path):
    '''
    Saves steering vectors to disk.
    
    Args:
        steering_vectors: Dict {op: {layer: tensor}}
        save_path: Path to save file (will use .pt extension)
    '''
    torch.save(steering_vectors, save_path)
    print(f"Steering vectors saved to {save_path}")


def load_steering_vectors(load_path):
    '''
    Loads steering vectors from disk.
    
    Args:
        load_path: Path to saved file
    
    Returns:
        steering_vectors: Dict {op: {layer: tensor}}
    '''
    steering_vectors = torch.load(load_path, map_location='cpu', weights_only=False)
    print(f"Steering vectors loaded from {load_path}")
    return steering_vectors


def get_cosyvoice_head_geometry(model) -> HeadGeometry:
    """
    Returns attention head geometry (num heads, kv heads, head dim) for CosyVoice's Qwen LLM.
    """
    try:
        config = model.llm.llm.model.config
        num_heads = int(config.num_attention_heads)
        num_kv = int(getattr(config, "num_key_value_heads", num_heads))
        hidden_size = int(config.hidden_size)
    except AttributeError as exc:
        raise ValueError("Unable to infer head geometry from CosyVoice model") from exc

    if num_heads == 0:
        raise ValueError("Invalid head configuration: num_heads=0")
    head_dim = hidden_size // num_heads
    return HeadGeometry(num_heads=num_heads, num_kv_heads=num_kv, head_dim=head_dim)


def get_indextts2_head_geometry(tts) -> HeadGeometry:
    """
    Returns attention head geometry (num heads, head dim) for IndexTTS2's GPT.
    
    Args:
        tts: IndexTTS2 instance (or tts.gpt for UnifiedVoice)
        
    Returns:
        HeadGeometry with num_heads, head_dim, and num_kv_heads (same as num_heads for GPT2)
    """
    try:
        # Handle both IndexTTS2 and UnifiedVoice inputs
        if hasattr(tts, 'gpt'):
            config = tts.gpt.gpt.config  # IndexTTS2.gpt.gpt.config
        else:
            config = tts.gpt.config  # UnifiedVoice.gpt.config
            
        num_heads = int(config.n_head)
        hidden_size = int(config.n_embd)
    except AttributeError as exc:
        raise ValueError("Unable to infer head geometry from IndexTTS2 model") from exc

    if num_heads == 0:
        raise ValueError("Invalid head configuration: num_heads=0")
    head_dim = hidden_size // num_heads
    
    # GPT2 doesn't use grouped query attention, so num_kv_heads == num_heads
    return HeadGeometry(num_heads=num_heads, num_kv_heads=num_heads, head_dim=head_dim)


###########################################################################
##                   Mixed-Emotion Steering Functions                    ##
###########################################################################

def combine_steering_vectors(steering_vectors_list, weights):
    if len(steering_vectors_list) != len(weights):
        raise ValueError(f"Number of vectors ({len(steering_vectors_list)}) must match number of weights ({len(weights)})")
    
    if len(steering_vectors_list) == 0:
        raise ValueError("steering_vectors_list cannot be empty")
    
    # Step 1: Weighted sum of all vectors
    combined = sum(w * v for w, v in zip(weights, steering_vectors_list))
    
    return combined


def create_mixed_steering_vectors(
    steering_files_dict,
    emotion_weights,
):
    # Load all steering files
    loaded_vectors = {}
    for emotion, filepath in steering_files_dict.items():
        data = torch.load(filepath, map_location='cpu', weights_only=False)
        loaded_vectors[emotion] = data.get('steering_vectors', data)
    
    # Verify all emotions are in weights
    for emotion in steering_files_dict.keys():
        if emotion not in emotion_weights:
            raise ValueError(f"Emotion '{emotion}' not found in emotion_weights")
    
    # Get all operations from first emotion file
    first_emotion = next(iter(loaded_vectors.keys()))
    all_operations = list(loaded_vectors[first_emotion].keys())
    
    # Initialize result structure for all operations
    mixed_vectors = {}
    
    # Combine vectors for each operation and layer
    for op in all_operations:
        mixed_vectors[op] = {}
        
        # Get all layers for this operation from first emotion
        all_layers = list(loaded_vectors[first_emotion][op].keys())
        
        for layer in all_layers:
            vectors_to_combine = []
            weights_to_use = []
            
            for emotion in steering_files_dict.keys():
                if op in loaded_vectors[emotion] and layer in loaded_vectors[emotion][op]:
                    vectors_to_combine.append(loaded_vectors[emotion][op][layer])
                    weights_to_use.append(emotion_weights[emotion])
                else:
                    print(f"Warning: {emotion} missing {op} at layer {layer}, skipping this emotion")
            
            if vectors_to_combine:
                mixed_vectors[op][layer] = combine_steering_vectors(
                    vectors_to_combine,
                    weights_to_use,
                )
            else:
                print(f"Warning: No vectors available for {op} at layer {layer}")
    
    return mixed_vectors


def create_sample_specific_mixed_vectors(
    steering_files_dict,
    emotion_percentages,
):
    """
    Create mixed steering vectors for a specific sample based on emotion percentages.
    Combines ALL operations and layers - filtering is done later by generate_steered_speech.
    
    Usage:
        # For a sample with mixed emotions from MSP dataset
        sample_vectors = create_sample_specific_mixed_vectors(
            steering_files_dict={
                'happy': 'path/to/happy.pt',
                'sad': 'path/to/sad.pt'
            },
            emotion_percentages={'p_happy': 0.6, 'p_sad': 0.4},
        )
        
        # Then use with generate_steered_speech:
        generate_steered_speech(
            steering_vectors=sample_vectors,
            operations=['layer_output'],  # Selects operation
            layers=[12, 13, 14, 15, 16],  # Selects layers
            alpha=2.0,  # Controls steering strength
            ...
        )
        
        # Formula: p_happy * v_happy + p_sad * v_sad
        # Then alpha is applied during generation to control steering strength
    
    Args:
        steering_files_dict: Dict mapping emotion names to steering file paths
        emotion_percentages: Dict with emotion percentages (e.g., {'p_happy': 0.6, 'p_sad': 0.4})
    
    Returns:
        mixed_vectors: Dict {operation: {layer: combined_tensor}} with ALL operations and layers
    """
    # Extract emotion weights from percentages (remove 'p_' prefix)
    emotion_weights = {}
    for key, value in emotion_percentages.items():
        if key.startswith('p_'):
            emotion_name = key[2:]  # Remove 'p_' prefix
            if emotion_name in steering_files_dict:
                emotion_weights[emotion_name] = value  # Just use percentage as weight
    
    if not emotion_weights:
        raise ValueError(
            f"No matching emotions found. "
            f"Percentages: {list(emotion_percentages.keys())}, "
            f"Available: {list(steering_files_dict.keys())}"
        )
    
    # Use create_mixed_steering_vectors with percentage weights
    # It will combine ALL operations and layers
    return create_mixed_steering_vectors(
        steering_files_dict=steering_files_dict,
        emotion_weights=emotion_weights,
    )


###########################################################################
##          CosyVoice Inference Wrapper Functions (DISCO-style)         ##
###########################################################################

def prepare_steering_injection_config(steering_vectors, operations, layers, alpha=1.0, op_dict=None):
    '''
    Prepares injection configuration for CosyVoice steering (DISCO-style).
    
    This function creates the operations_to_hook_info dict needed by hook_model_inject().
    Similar to DISCO's update_operations_to_hook_info() but for CosyVoice structure.
    
    Args:
        steering_vectors: Dict {op: {layer: tensor}} from create_mean_difference()
        operations: List of operations to inject into (must be in steering_vectors)
        layers: List of layer indices to inject into
        alpha: Steering strength (float or dict {op: float})
        op_dict: Optional operation dict (will use COSYVOICE_OP_DICTS if None)
    
    Returns:
        operations_to_hook_info: Configuration dict for hook_model_inject()
    '''
    if op_dict is None:
        op_dict = INDEXTTS2_OP_DICTS
    
    # Handle alpha as scalar or dict
    if isinstance(alpha, (int, float)):
        alpha_dict = {op: alpha for op in operations}
    else:
        alpha_dict = alpha
    
    operations_to_hook_info = {}
    
    for op in operations:
        if op not in steering_vectors:
            raise ValueError(f"Operation '{op}' not found in steering_vectors")
        
        if op not in op_dict:
            raise ValueError(f"Operation '{op}' not found in operation dictionary")
        
        op_dict_entry = op_dict[op]
        
        # Build layer_to_inject dict
        layer_to_inject = {}
        for layer in layers:
            if layer not in steering_vectors[op]:
                print(f"Warning: Layer {layer} not in steering_vectors['{op}'], skipping")
                continue
            
            layer_to_inject[layer] = {
                # 'inject_op': norm_preserve_steer_op_,
                'inject_op': translation_op_,
                'inject_dict': {
                    't': steering_vectors[op][layer].cuda(),
                    'alpha': alpha_dict[op]
                }
            }
        
        operations_to_hook_info[op] = {
            'module': op_dict_entry['module'],
            'hook type': op_dict_entry['hook type'],
            'inject': True,
            'layer_to_inject': layer_to_inject
        }
    
    return operations_to_hook_info


def inference_zero_shot_with_steering(
    cosyvoice,
    tts_text,
    prompt_text,
    prompt_speech_16k,
    steering_vectors,
    operations,
    layers,
    alpha=1.0,
    stream=True,
    speed=1.0
):
    '''
    Wraps CosyVoice.inference_zero_shot() with steering vector injection (DISCO-style).
    
    This function follows DISCO's generate_with_hooks() pattern:
    1. Prepare injection configuration
    2. Register hooks on the model
    3. Call NORMAL CosyVoice inference (unchanged!)
    4. Hooks automatically apply steering during generation
    5. Clean up hooks
    
    Args:
        cosyvoice: CosyVoice2 instance
        tts_text: Text to synthesize
        prompt_text: Prompt text for zero-shot
        prompt_speech_16k: Prompt audio (16kHz)
        steering_vectors: Dict {op: {layer: tensor}} from create_mean_difference()
        operations: List of operations to steer (e.g., ['layer_output'])
        layers: List of layer indices to apply steering
        alpha: Steering strength (default 1.0)
        stream: Whether to stream audio chunks (default True)
        speed: Speech speed multiplier (default 1.0)
    
    Yields:
        Audio chunks (if stream=True) or final audio (if stream=False)
    
    Example:
        >>> steering_vecs = load_steering_vectors('happy_vs_neutral.pt')
        >>> prompt_audio = load_wav('neutral_prompt.wav', 16000)
        >>> for audio in inference_zero_shot_with_steering(
        ...     cosyvoice, 
        ...     tts_text="This should sound happy!",
        ...     prompt_text="This is a test",
        ...     prompt_speech_16k=prompt_audio,
        ...     steering_vectors=steering_vecs['steering_vectors'],
        ...     operations=['layer_output'],
        ...     layers=[12, 13, 14, 15, 16],
        ...     alpha=1.0
        ... ):
        ...     # Process audio chunk
        ...     pass
    '''
    # Step 1: Prepare injection configuration (DISCO-style)
    injection_config = prepare_steering_injection_config(
        steering_vectors=steering_vectors,
        operations=operations,
        layers=layers,
        alpha=alpha
    )
    
    # Step 2: Register hooks on the model
    hooks, handles = hook_model_inject(cosyvoice.model, injection_config)
    
    try:
        # Step 3: Call NORMAL CosyVoice inference (unchanged!)
        # Hooks will automatically intercept forward passes and apply steering
        for audio_chunk in cosyvoice.inference_zero_shot(
            tts_text=tts_text,
            prompt_text=prompt_text,
            prompt_speech_16k=prompt_speech_16k,
            stream=stream,
            speed=speed
        ):
            yield audio_chunk
    
    finally:
        # Step 4: Clean up hooks (always executed, even if error occurs)
        unhook(handles)


def inference_sft_with_steering(
    cosyvoice,
    tts_text,
    spk_id,
    steering_vectors,
    operations,
    layers,
    alpha=1.0,
    stream=True,
    speed=1.0
):
    '''
    Wraps CosyVoice.inference_sft() with steering vector injection.
    
    Similar to inference_zero_shot_with_steering() but for SFT (speaker fine-tuning) mode.
    
    Args:
        cosyvoice: CosyVoice2 instance
        tts_text: Text to synthesize
        spk_id: Speaker ID for SFT
        steering_vectors: Dict {op: {layer: tensor}}
        operations: List of operations to steer
        layers: List of layer indices
        alpha: Steering strength
        stream: Whether to stream
        speed: Speech speed
    
    Yields:
        Audio chunks
    '''
    # Prepare and register hooks
    injection_config = prepare_steering_injection_config(
        steering_vectors=steering_vectors,
        operations=operations,
        layers=layers,
        alpha=alpha
    )
    
    hooks, handles = hook_model_inject(cosyvoice.model, injection_config)
    
    try:
        # Call normal inference with steering applied via hooks
        for audio_chunk in cosyvoice.inference_sft(
            tts_text=tts_text,
            spk_id=spk_id,
            stream=stream,
            speed=speed
        ):
            yield audio_chunk
    
    finally:
        unhook(handles)


def inference_cross_lingual_with_steering(
    cosyvoice,
    tts_text,
    prompt_speech_16k,
    steering_vectors,
    operations,
    layers,
    alpha=1.0,
    stream=True,
    speed=1.0
):
    '''
    Wraps CosyVoice.inference_cross_lingual() with steering vector injection.
    
    Similar to inference_zero_shot_with_steering() but for cross-lingual mode.
    
    Args:
        cosyvoice: CosyVoice2 instance
        tts_text: Text to synthesize
        prompt_speech_16k: Prompt audio (16kHz)
        steering_vectors: Dict {op: {layer: tensor}}
        operations: List of operations to steer
        layers: List of layer indices
        alpha: Steering strength
        stream: Whether to stream
        speed: Speech speed
    
    Yields:
        Audio chunks
    '''
    # Prepare and register hooks
    injection_config = prepare_steering_injection_config(
        steering_vectors=steering_vectors,
        operations=operations,
        layers=layers,
        alpha=alpha
    )
    
    hooks, handles = hook_model_inject(cosyvoice.model, injection_config)
    
    try:
        # Call normal inference with steering applied via hooks
        for audio_chunk in cosyvoice.inference_cross_lingual(
            tts_text=tts_text,
            prompt_speech_16k=prompt_speech_16k,
            stream=stream,
            speed=speed
        ):
            yield audio_chunk
    
    finally:
        unhook(handles)

def inference_indextts2_with_steering(
    tts,
    text,
    emo_vector,
    spk_audio_prompt,
    steering_vectors,
    operations,
    layers,
    alpha=0.0,
    output_path=None,
    verbose=True,
    **infer_kwargs
):
    '''
    Apply steering to IndexTTS2 GPT during mel-code generation only.

    This wraps tts.gpt.inference_speech with hooks, then calls tts.infer to
    generate the final audio without modifying the rest of the pipeline.
    '''
    injection_config = prepare_steering_injection_config(
        steering_vectors=steering_vectors,
        operations=operations,
        layers=layers,
        alpha=alpha,
        op_dict=get_indextts2_op_dict(operations)
    )

    hooks, handles = hook_model_inject(tts.gpt, injection_config)
    try:
        return tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            emo_vector=emo_vector,
            output_path=output_path,
            verbose=verbose,
            **infer_kwargs
        )
    finally:
        unhook(handles)

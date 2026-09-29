"""Attention-received and mean-pooled embedding features for turn selection."""
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import List, Optional, Tuple
from dataclasses import dataclass, field

def load_shared_model(model_name: str, device: str=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model_kwargs = {'torch_dtype': torch.float16 if device == 'cuda' else torch.float32}
    model_type = getattr(AutoConfig.from_pretrained(model_name), 'model_type', '')
    if model_type in {'llama', 'mistral', 'qwen2', 'gpt2', 'gpt_neo', 'gpt_neox', 'falcon'}:
        model_kwargs['attn_implementation'] = 'eager'
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    except (TypeError, ValueError):
        if 'attn_implementation' in model_kwargs:
            print(f'[load_shared_model] attn_implementation=eager rejected for {model_type}, retrying without.')
        model_kwargs.pop('attn_implementation', None)
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return (model, tokenizer, device)

@dataclass
class Turn:
    speaker: str
    text: str
    tokens: List[int] = field(default_factory=list)
    perplexity: float = None
    is_boundary: bool = False

class EpisodicBoundaryDetector:

    def __init__(self, model_name: str='gpt2', threshold_multiplier: float=1.5, window_size: int=3, device: str=None, detection_mode: str='topk', top_k_ratio: float=0.2, shared_model: Optional[Tuple[object, object, str]]=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.threshold_multiplier = threshold_multiplier
        self.window_size = window_size
        self.detection_mode = detection_mode
        self.top_k_ratio = top_k_ratio
        self.model_name = model_name
        if shared_model is not None:
            self.model, self.tokenizer, self.device = shared_model
        else:
            print(f'Loading model: {model_name} on {self.device} (detection: {detection_mode})')
            self.model, self.tokenizer, self.device = load_shared_model(model_name, self.device)

    def get_real_attention_scores(self, turns: List[Turn], return_diagnostics: bool=False, verbose: bool=False):
        if not turns:
            if return_diagnostics:
                return ([], {'was_truncated': False, 'truncated_tokens': 0, 'num_truncated_turns': 0, 'num_zero_turns': 0})
            return []
        SEP = ' '
        turn_texts = [t.text for t in turns]
        full_text = SEP.join(turn_texts)
        turn_spans = []
        prefix = ''
        for i, text in enumerate(turn_texts):
            if i > 0:
                prefix += SEP
            start_ids = self.tokenizer(prefix, add_special_tokens=False)['input_ids']
            prefix += text
            end_ids = self.tokenizer(prefix, add_special_tokens=False)['input_ids']
            turn_spans.append((len(start_ids), len(end_ids)))
        untruncated_len = len(self.tokenizer(full_text, add_special_tokens=False)['input_ids'])
        full_inputs = self.tokenizer(full_text, return_tensors='pt', truncation=True, max_length=1024, add_special_tokens=True).to(self.device)
        total_tokens = full_inputs['input_ids'].shape[1]
        bos_offset = 0
        if self.tokenizer.bos_token_id is not None and total_tokens > 0 and (full_inputs['input_ids'][0, 0].item() == self.tokenizer.bos_token_id):
            bos_offset = 1
        content_capacity = total_tokens - bos_offset
        was_truncated = untruncated_len > content_capacity
        truncated_tokens = max(0, untruncated_len - content_capacity)
        if was_truncated and verbose:
            print(f'[get_real_attention_scores] WARNING: dialogue truncated — {untruncated_len} tokens → {content_capacity} tokens ({truncated_tokens} tokens cut).  Truncated turns will receive score 0.0 (conservative / experimentally fair).')
        turn_spans = [(s + bos_offset, min(e + bos_offset, total_tokens)) for s, e in turn_spans]
        with torch.no_grad():
            outputs = self.model(**full_inputs, output_attentions=True)
        all_attns = torch.stack([a[0] for a in outputs.attentions], dim=0)
        mean_attn = all_attns.mean(dim=(0, 1)).cpu().float().numpy()
        attn_received = mean_attn.sum(axis=0)
        raw_scores = []
        num_truncated_turns = 0
        for start, end in turn_spans:
            if 0 <= start < end and end <= len(attn_received):
                raw_scores.append(float(attn_received[start:end].mean()))
            else:
                raw_scores.append(0.0)
                num_truncated_turns += 1
        num_zero_turns = sum((1 for s in raw_scores if s == 0.0))
        arr = np.array(raw_scores, dtype=float)
        lo, hi = (arr.min(), arr.max())
        if hi > lo + 1e-09:
            arr = (arr - lo) / (hi - lo)
        else:
            arr = np.ones(len(arr)) / len(arr)
        scores = arr.tolist()
        if return_diagnostics:
            diagnostics = {'was_truncated': was_truncated, 'truncated_tokens': truncated_tokens, 'num_truncated_turns': num_truncated_turns, 'num_zero_turns': num_zero_turns}
            return (scores, diagnostics)
        return scores

    def get_turn_embeddings(self, turns: List[Turn]) -> 'np.ndarray':
        import torch as _torch
        embeddings = []
        for turn in turns:
            inputs = self.tokenizer(turn.text, return_tensors='pt', truncation=True, max_length=128, padding=False, add_special_tokens=True).to(self.device)
            with _torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[-1][0]
            emb = hidden.mean(dim=0).cpu().float().numpy()
            embeddings.append(emb)
        arr = np.array(embeddings, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-09
        return arr / norms

"""Turn selectors used in the FMTS experiment."""
import random
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from turn_types import Turn


LLMLINGUA_WORD_SEPARATOR = "\t\t|\t\t"
LLMLINGUA_LABEL_SEPARATOR = " "


def mean_llmlingua_word_labels(labeled_text: str) -> float:
    """Average binary retention labels, never digits in the original words.

    LLMLingua returns word + label_sep + {0,1}, joined by word_sep.
    These are hard word-retention decisions, not importance probabilities.
    Malformed/missing output must stop the run rather than silently select
    earliest turns after assigning all scores zero.
    """
    if not isinstance(labeled_text, str) or not labeled_text:
        raise ValueError("LLMLingua returned no labeled original prompt")
    labels = []
    for item in labeled_text.split(LLMLINGUA_WORD_SEPARATOR):
        try:
            word, label = item.rsplit(LLMLINGUA_LABEL_SEPARATOR, 1)
        except ValueError as exc:
            raise ValueError("Malformed LLMLingua word-label entry") from exc
        # XLM-R can emit an empty surface word for a standalone whitespace
        # boundary (e.g. before punctuation). Its binary label is legitimate.
        if label not in {"0", "1"}:
            raise ValueError("Expected a binary LLMLingua word label")
        labels.append(int(label))
    return sum(labels) / len(labels)

@dataclass
class CompressedContext:
    method: str
    kept_turn_indices: List[int]
    compression_ratio: float
    text: str

class ContextCompressor:

    def __init__(self, compression_ratio: float=0.5, cue_search_radius: int=5):
        self.compression_ratio = compression_ratio
        self.cue_search_radius = cue_search_radius

    def _turns_to_text(self, turns: List[Turn], indices: List[int]) -> str:
        return ' '.join((turns[i].text for i in sorted(indices)))

    def _n_keep(self, n_turns: int) -> int:
        return max(1, int(n_turns * self.compression_ratio))

    def full_context(self, turns: List[Turn]) -> CompressedContext:
        indices = list(range(len(turns)))
        return CompressedContext('full_context', indices, 1.0, self._turns_to_text(turns, indices))

    def random_compression(self, turns: List[Turn]) -> CompressedContext:
        k = self._n_keep(len(turns))
        indices = sorted(random.sample(range(len(turns)), k))
        return CompressedContext('random', indices, self.compression_ratio, self._turns_to_text(turns, indices))

    def first_n(self, turns: List[Turn]) -> CompressedContext:
        k = self._n_keep(len(turns))
        indices = list(range(min(k, len(turns))))
        return CompressedContext('first_n', indices, len(indices) / max(1, len(turns)), self._turns_to_text(turns, indices))

    def uniform_stride(self, turns: List[Turn]) -> CompressedContext:
        n = len(turns)
        k = self._n_keep(n)
        if n == 0 or k <= 0:
            return CompressedContext('uniform_stride', [], 0.0, '')
        if k >= n:
            indices = list(range(n))
        else:
            step = n / k
            indices = sorted({min(n - 1, int(round(i * step))) for i in range(k)})
        return CompressedContext('uniform_stride', indices, len(indices) / max(1, n), self._turns_to_text(turns, indices))

    def recency(self, turns: List[Turn]) -> CompressedContext:
        k = self._n_keep(len(turns))
        indices = list(range(max(0, len(turns) - k), len(turns)))
        return CompressedContext('recency', indices, len(indices) / len(turns), self._turns_to_text(turns, indices))

    def attention_based(self, turns: List[Turn], attention_scores: List[float]) -> CompressedContext:
        k = self._n_keep(len(turns))
        scored = sorted(enumerate(attention_scores), key=lambda x: x[1], reverse=True)
        indices = sorted([i for i, _ in scored[:k]])
        return CompressedContext('attention_h2o', indices, self.compression_ratio, self._turns_to_text(turns, indices))

    def embedding_mmr(self, turns: List[Turn], embeddings: Optional[np.ndarray]=None) -> CompressedContext:
        k = self._n_keep(len(turns))
        n = len(turns)
        if embeddings is None:
            vocab = {}
            for t in turns:
                for w in t.text.lower().split():
                    if w not in vocab:
                        vocab[w] = len(vocab)
            emb = np.zeros((n, len(vocab) + 1))
            for i, t in enumerate(turns):
                for w in t.text.lower().split():
                    emb[i, vocab.get(w, 0)] += 1
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-09
            embeddings = emb / norms
        query = embeddings.mean(axis=0)
        lambda_param = 0.5
        selected = []
        remaining = list(range(n))
        relevance_scores = np.array([float(np.dot(embeddings[i], query) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(query) + 1e-09)) for i in range(n)])
        for _ in range(k):
            if not remaining:
                break
            rel = np.array([relevance_scores[i] for i in remaining])
            if selected:
                sim_to_sel = np.array([max((float(np.dot(embeddings[i], embeddings[s]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[s]) + 1e-09)) for s in selected)) for i in remaining])
            else:
                sim_to_sel = np.zeros(len(remaining))
            scores = lambda_param * rel - (1 - lambda_param) * sim_to_sel
            best_local = int(np.argmax(scores))
            selected.append(remaining[best_local])
            remaining.pop(best_local)
        indices = sorted(selected)
        return CompressedContext('embedding_mmr', indices, len(indices) / n, self._turns_to_text(turns, indices))
    _llmlingua2_compressor = None

    @classmethod
    def _get_llmlingua2(cls):
        if cls._llmlingua2_compressor is None:
            try:
                from llmlingua import PromptCompressor
            except ImportError as e:
                raise ImportError('llmlingua package required for llmlingua2 baseline. Install with: pip install llmlingua') from e
            cls._llmlingua2_compressor = PromptCompressor(
                model_name='microsoft/llmlingua-2-xlm-roberta-large-meetingbank',
                use_llmlingua2=True,
                model_config={
                    'revision': 'ebaba9b0e874dadd3003ffcff828e4397e568089',
                    'trust_remote_code': False,
                    'torch_dtype': 'float32',
                },
            )
        return cls._llmlingua2_compressor

    def llmlingua2(self, turns: List[Turn]) -> CompressedContext:
        k = self._n_keep(len(turns))
        n = len(turns)
        if k >= n:
            indices = list(range(n))
            return CompressedContext('llmlingua2', indices, 1.0, self._turns_to_text(turns, indices))
        pc = self._get_llmlingua2()
        scores = []
        for turn_index, t in enumerate(turns):
            text = (t.text or '').strip()
            if not text:
                scores.append(0.0)
                continue
            try:
                result = pc.compress_prompt(
                    [text], rate=0.5, force_tokens=[], return_word_label=True,
                    word_sep=LLMLINGUA_WORD_SEPARATOR,
                    label_sep=LLMLINGUA_LABEL_SEPARATOR,
                )
                scores.append(mean_llmlingua_word_labels(
                    result.get('fn_labeled_original_prompt')
                ))
            except Exception as exc:
                raise RuntimeError(
                    f"LLMLingua-2 scoring failed for turn {turn_index}; "
                    "no fallback selection was produced"
                ) from exc
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        kept = sorted(ranked[:k])
        return CompressedContext('llmlingua2', kept, len(kept) / n, self._turns_to_text(turns, kept))

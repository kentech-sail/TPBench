"""Text-retention diagnostics for FMTS contexts."""
import numpy as np
import re
from typing import List, Optional
from rouge_score import rouge_scorer
from dataclasses import dataclass, field

@dataclass
class EvalResult:
    method: str
    rouge_l: float
    coherence_score: float
    boundary_recall: float
    qa_accuracy: float
    boundary_qa_accuracy: float
    compression_ratio: float
    compressed_tokens: int
    full_tokens: int
    token_ratio: float
    compressed_chars: int
    full_chars: int
    char_ratio: float

class Evaluator:

    def __init__(self, use_bertscore: bool=False):
        self.rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.use_bertscore = use_bertscore
        if use_bertscore:
            try:
                from bert_score import score as bert_score_fn
                self._bert_score_fn = bert_score_fn
                print('BERTScore enabled.')
            except ImportError:
                print('BERTScore not installed. pip install bert-score. Falling back to ROUGE.')
                self.use_bertscore = False

    def rouge_l(self, hypothesis: str, reference: str) -> float:
        score = self.rouge.score(reference, hypothesis)
        return score['rougeL'].fmeasure

    def coherence_score(self, compressed_text: str, full_text: str) -> float:
        if self.use_bertscore and full_text.strip() and compressed_text.strip():
            try:
                P, R, F1 = self._bert_score_fn([compressed_text], [full_text], lang='en', verbose=False)
                return float(F1[0])
            except Exception:
                pass
        return self.rouge_l(compressed_text, full_text)

    def storage_stats(self, compressed_text: str, full_text: str) -> dict:
        compressed_tokens = len(compressed_text.split())
        full_tokens = len(full_text.split())
        compressed_chars = len(compressed_text)
        full_chars = len(full_text)
        return {'compressed_tokens': compressed_tokens, 'full_tokens': full_tokens, 'token_ratio': compressed_tokens / full_tokens if full_tokens else 0.0, 'compressed_chars': compressed_chars, 'full_chars': full_chars, 'char_ratio': compressed_chars / full_chars if full_chars else 0.0}

    def boundary_recall(self, kept_indices: List[int], true_boundary_indices: List[int]) -> float:
        if not true_boundary_indices:
            return 1.0
        kept_set = set(kept_indices)
        recalled = sum((1 for idx in true_boundary_indices if idx in kept_set))
        return recalled / len(true_boundary_indices)

    def qa_accuracy(self, compressed_text: str, qa_pairs: List[dict], model=None, requires_boundary: Optional[bool]=None) -> float:
        if requires_boundary is not None:
            qa_pairs = [qa for qa in qa_pairs if bool(qa.get('requires_boundary', False)) == requires_boundary]
        if not qa_pairs:
            return 1.0

        def normalize_tokens(text: str) -> set:
            tokens = set(re.findall('[a-z0-9]+', text.lower()))
            tokens.update(re.findall('[가-힣]+', text))
            return tokens
        correct = 0
        eligible = 0
        for qa in qa_pairs:
            if qa.get('qa_type') == 'dep':
                continue
            eligible += 1
            question = qa.get('question', '')
            expected = qa.get('answer', '').lower()
            qa_type = qa.get('qa_type', '')
            old_val = qa.get('old_value', '')
            new_val = qa.get('new_value', '')
            if model is not None:
                try:
                    prediction = model(compressed_text, question).lower()
                    if qa_type == 'reversal_combined' and old_val and new_val:
                        correct += int(old_val.lower() in prediction and new_val.lower() in prediction)
                    else:
                        correct += int(expected in prediction or prediction in expected)
                except Exception:
                    pass
            elif qa_type == 'reversal_combined' and old_val and new_val:
                old_in = old_val.lower() in compressed_text.lower()
                new_in = new_val.lower() in compressed_text.lower()
                correct += int(old_in and new_in)
            elif expected and (expected in compressed_text or expected in compressed_text.lower()):
                correct += 1
            else:
                key_words = normalize_tokens(expected)
                context_words = normalize_tokens(compressed_text)
                overlap = len(key_words & context_words) / max(len(key_words), 1)
                correct += int(overlap > 0.5)
        if eligible == 0:
            return float('nan')
        return correct / eligible

    def evaluate_all(self, compressed_results: list, full_text: str, true_boundary_indices: List[int], qa_pairs: Optional[List[dict]]=None, qa_model=None) -> List[EvalResult]:
        results = []
        for comp in compressed_results:
            storage = self.storage_stats(comp.text, full_text)
            results.append(EvalResult(method=comp.method, rouge_l=self.rouge_l(comp.text, full_text), coherence_score=self.coherence_score(comp.text, full_text), boundary_recall=self.boundary_recall(comp.kept_turn_indices, true_boundary_indices), qa_accuracy=self.qa_accuracy(comp.text, qa_pairs or [], qa_model), boundary_qa_accuracy=self.qa_accuracy(comp.text, qa_pairs or [], qa_model, requires_boundary=True), compression_ratio=comp.compression_ratio, compressed_tokens=storage['compressed_tokens'], full_tokens=storage['full_tokens'], token_ratio=storage['token_ratio'], compressed_chars=storage['compressed_chars'], full_chars=storage['full_chars'], char_ratio=storage['char_ratio']))
        return results

    def print_table(self, results: List[EvalResult]):
        header = f"\n{'Method':<35} {'ROUGE-L':>8} {'Coherence':>10} {'QA Acc':>8} {'B-QA':>8} {'B-Recall':>10} {'TurnR':>6} {'TokR':>6}"
        print(header)
        print('-' * 98)
        for r in results:
            print(f'{r.method:<35} {r.rouge_l:>8.3f} {r.coherence_score:>10.3f} {r.qa_accuracy:>8.3f} {r.boundary_qa_accuracy:>8.3f} {r.boundary_recall:>10.3f} {r.compression_ratio:>6.2f} {r.token_ratio:>6.2f}')
        print()
        print('NOTE: Boundary Recall is diagnostic. Boundary QA (B-QA) is the main synthetic metric.')

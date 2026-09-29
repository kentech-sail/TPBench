"""Prepare changed Llama LLMLingua-2 requests, retaining original task protocol.

No inference or API calls. Exact prompt/config fingerprints deduplicate requests.
Unchanged contexts reuse the published per-example predictions explicitly;
available raw responses are reused only for identical prompt/settings hashes.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
REV = '0e9e39f249a16976918f6564b8830bc894c89659'
METHOD = 'llmlingua2_cache'


def read(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def sha(data):
    return hashlib.sha256(data.encode('utf-8')).hexdigest()


def request(probe, paper_probe):
    fields = dict(prompt_system=probe['prompt_system'], prompt_user=probe['prompt_user'],
        max_new_tokens=128 if paper_probe == 'P3' else 96,
        attn_impl='sdpa' if paper_probe == 'P3' else 'eager', max_input_tokens=7168)
    fingerprint = dict(fields, model=MODEL, revision=REV, dtype='bfloat16', do_sample=False)
    return dict(id=sha(json.dumps(fingerprint, sort_keys=True, ensure_ascii=False)), **fields)


def cache_fingerprint(row):
    fingerprint = {k: row[k] for k in ('prompt_system', 'prompt_user', 'max_new_tokens', 'attn_impl', 'max_input_tokens')}
    return sha(json.dumps(dict(fingerprint, model=MODEL, revision=REV, dtype='bfloat16', do_sample=False), sort_keys=True, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--contexts', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    artifact, output = args.artifact.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    contexts = {}
    for row in read(args.contexts):
        if row['family'] != 'main':
            continue
        key = row['dataset'], round(row['ratio'] * 100), row['dialogue_id']
        if key in contexts:
            raise ValueError(f'duplicate corrected context: {key}')
        contexts[key] = row
    if len(contexts) != 3374:
        raise ValueError(f'expected complete 1687-example main pool at both budgets, got {len(contexts)}')
    cached = defaultdict(list)
    inputs = {r['id']: r for r in read(artifact / 'data/reader_evaluation/inputs.jsonl')}
    original_p1_keys = {(r['dataset'], r['sample_seed'], r['dialogue_id']): r['check_id']
        for r in read(artifact / 'data/reader_evaluation/key.jsonl')
        if r['paper_probe'] == 'P1' and r['method'] == METHOD}
    for answer in read(artifact / 'results/reader_evaluation/answers.jsonl'):
        if answer.get('error'):
            continue
        row = inputs[answer['id']]
        cached[cache_fingerprint(row)].append(dict(answer, cache_source='results/reader_evaluation/answers.jsonl', source_id=answer['id']))
    # Original P3 LLMLingua answers are absent from the matched baseline key.
    for ds in ('sgd', 'multiwoz'):
        for seed in (42, 43, 44):
            probes = read(artifact / f'data/probes/probes_{ds}_s{seed}_r30_n200_late_intent.jsonl')
            index = {(r['dialogue_id'], r['method']): r for r in probes}
            path = artifact / f'results/diagnostics/update_evidence/p3_{ds}_s{seed}_reader_llama.jsonl'
            for answer in read(path):
                if answer.get('error'):
                    continue
                probe = index[answer['dialogue_id'], answer['method']]
                req = request(probe, 'P3')
                cached[req['id']].append(dict(answer, cache_source=str(path.relative_to(artifact)), source_id=answer.get('task_uid')))
    requests, expanded, reused_scores, reused_raw = {}, [], [], {}
    counts, conflicts = Counter(), 0
    for ds in ('sgd', 'multiwoz'):
        tag = 'sgd' if ds == 'sgd' else 'mw'
        for seed in (42, 43, 44):
            for paper_probe, ratio, probe_path, scored_path in [
                ('P1', 30, f'data/probes/probes_{ds}_s{seed}_r30_p1_n200.jsonl', f'results/p1_saved/{tag}_r30_s{seed}_scored.jsonl'),
                ('P2', 10, f'data/probes/probes_{ds}_s{seed}_r10_p3_n200.jsonl', f'results/main/{tag}_r10_s{seed}_p3_scored.jsonl'),
                ('P2', 30, f'data/probes/probes_{ds}_s{seed}_r30_p3_n200.jsonl', f'results/main/{tag}_r30_s{seed}_p3_scored.jsonl'),
                ('P3', 30, f'data/probes/probes_{ds}_s{seed}_r30_n200_late_intent.jsonl', f'results/diagnostics/update_evidence/p3_{ds}_s{seed}_scored_llama.jsonl')]:
                scores = {r['dialogue_id']: r for r in read(artifact / scored_path) if r['method'] == METHOD}
                for probe in read(artifact / probe_path):
                    if probe['method'] != METHOD:
                        continue
                    context = contexts[ds, ratio, probe['dialogue_id']]
                    original = probe['compressed_text_used']
                    if original != context['saved_context']:
                        raise ValueError(f'original context differs from regeneration input: {probe_path} {probe["dialogue_id"]}')
                    # Preserve the complete original question/schema suffix byte-for-byte.
                    prefix = 'Context:\n' + original
                    if not probe['prompt_user'].startswith(prefix + '\n\nQuestion:'):
                        raise ValueError('unexpected original prompt format')
                    fixed = dict(probe, compressed_text_used=context['corrected_context'],
                        compressed_text_chars=len(context['corrected_context']),
                        compressed_text_chars_used=len(context['corrected_context']),
                        meta=dict(probe['meta'], k_turns=len(context['corrected_indices'])),
                        original_meta_k_turns=probe['meta'].get('k_turns'),
                        original_compressed_text_chars=probe.get('compressed_text_chars'),
                        prompt_user='Context:\n' + context['corrected_context'] + probe['prompt_user'][len(prefix):])
                    req = request(fixed, paper_probe)
                    row = dict(fixed, check_id=req['id'], paper_probe=paper_probe, sample_seed=seed,
                        original_probe_path=probe_path, original_score_path=scored_path,
                        original_context_sha256=sha(original), corrected_context_sha256=sha(context['corrected_context']),
                        corrected_kept_turn_indices=context['corrected_indices'],
                        correction='LLMLingua2 word-label-only aggregation; original prompt/pool/settings retained')
                    if paper_probe == 'P1':
                        row['original_check_id'] = original_p1_keys[ds, seed, probe['dialogue_id']]
                        row['question_condition'] = 'initial_goal'
                    else:
                        row['question_condition'] = 'original_question'
                    expanded.append(row)
                    counts[f'{paper_probe}_r{ratio}_expanded'] += 1
                    if context['corrected_context'] == original:
                        reused_scores.append(dict(check_id=req['id'], dataset=ds, sample_seed=seed, paper_probe=paper_probe,
                            ratio=ratio / 100, dialogue_id=probe['dialogue_id'], score=scores[probe['dialogue_id']],
                            provenance='reused_original_scored_prediction_exact_unchanged_context', source=scored_path))
                        counts['unchanged_expanded'] += 1
                        continue
                    counts['changed_expanded'] += 1
                    possibilities = cached[req['id']]
                    unique = {r.get('reader_output_text') for r in possibilities}
                    if len(unique) == 1:
                        source = possibilities[0]
                        reused_raw[req['id']] = dict(id=req['id'], reader_output_text=source['reader_output_text'], error=None,
                            model=MODEL, provenance='reused_original_raw_response_exact_prompt_and_settings',
                            source=source['cache_source'], source_id=source['source_id'],
                            input_tokens=source.get('input_tokens'))
                        counts['cached_changed_expanded'] += 1
                    else:
                        if len(unique) > 1:
                            conflicts += 1
                        requests[req['id']] = req
    for name, rows in [('inputs.jsonl', sorted(requests.values(), key=lambda r: r['id'])),
                       ('key.jsonl', expanded), ('reused_original_scores.jsonl', reused_scores),
                       ('reused_raw_answers.jsonl', sorted(reused_raw.values(), key=lambda r: r['id']))]:
        (output / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    summary = dict(model=MODEL, revision=REV, no_paid_api=True, counts=dict(counts),
        unique_new_requests=len(requests), unique_reused_raw_requests=len(reused_raw),
        cached_output_conflicts_requiring_new_inference=conflicts,
        requests_by_attention=dict(Counter(r['attn_impl'] for r in requests.values())),
        p1_semantic_upper_bound_records=counts['P1_r30_expanded'] - sum(r['paper_probe'] == 'P1' for r in reused_scores),
        scope='Llama main P1/P2/P3 LLMLingua-2 only; original pools/seed sizes and prompt wording; no Mistral or paid API rerun',
        contexts_sha256=hashlib.sha256(args.contexts.read_bytes()).hexdigest())
    (output / 'preparation.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()

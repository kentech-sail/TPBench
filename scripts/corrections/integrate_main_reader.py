"""Integrate a completed, scoped LLMLingua rerun into a staged release.

Only the selected reader's LLMLingua rows and their derived aggregates change.
The caller supplies primary answers; diagnostic batched P2/P3 are never read.
Original LLMLingua rows are retained separately. No inference or API is used.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

METHOD = 'llmlingua2_cache'


def read(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def write(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))


def dump(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def moments(values):
    return dict(mean=statistics.mean(values), std=statistics.stdev(values), n_seeds=len(values), values=values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--staged', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--scored', type=Path, required=True)
    p.add_argument('--answers', type=Path, nargs='+', required=True)
    p.add_argument('--reader', choices=['llama', 'mistral'], required=True)
    a = p.parse_args()
    original, staged = a.original, a.staged
    keys = read(a.prepared / 'key.jsonl')
    scored = read(a.scored / 'expanded_scored.jsonl')
    summary = json.loads((a.scored / 'summary.json').read_text())
    assert summary['complete'] and summary['reader'] == a.reader
    assert summary['expanded_scored'] == len(keys) == len(scored)
    assert len(scored) == (4164 if a.reader == 'llama' else 2964)
    index = {(r['dataset'], r['seed'], r['paper_probe'], r['ratio'], r['dialogue_id']): r for r in scored}
    raw = {}
    for fp in [a.prepared / 'reused_raw_answers.jsonl'] + a.answers:
        for r in read(fp):
            assert r['id'] not in raw
            raw[r['id']] = r
    historical = Path(f'results/historical/llmlingua_{a.reader}_legacy_20260929')
    grouped = defaultdict(list)
    for key in keys:
        row = index[key['dataset'], key['sample_seed'], key['paper_probe'], key['meta']['ratio'], key['dialogue_id']]
        assert row['check_id'] == key['check_id']
        assert key['meta']['k_turns'] == len(key['corrected_kept_turn_indices'])
        grouped[key['original_score_path']].append((key, row))
    for rel, pairs in grouped.items():
        changed = {key['dialogue_id']: r['score'] for key, r in pairs}
        old_rows = read(original / rel)
        assert len(changed) == sum(r['method'] == METHOD for r in old_rows)
        write(staged / historical / rel, [r for r in old_rows if r['method'] == METHOD])
        rows = [changed[r['dialogue_id']] if r['method'] == METHOD else r for r in old_rows]
        write(staged / rel, rows)
        ds, probe = pairs[0][0]['dataset'], pairs[0][0]['paper_probe']
        # Diagnostic raw tables are part of the current provenance chain too.
        if probe == 'P3':
            raw_rel = rel.replace('_scored_', '_reader_')
        elif probe == 'P2' and a.reader == 'mistral':
            raw_rel = rel.replace('_scored.jsonl', '_reader_all.jsonl')
        else:
            raw_rel = None
        if raw_rel:
            raw_rows = read(original / raw_rel)
            write(staged / historical / raw_rel, [r for r in raw_rows if r['method'] == METHOD])
            replacements = {k['dialogue_id']:k for k,_ in pairs}
            fresh_rows = []
            for old in raw_rows:
                if old['method'] != METHOD:
                    fresh_rows.append(old)
                    continue
                key = replacements[old['dialogue_id']]
                req_id = key['check_id']
                row = dict(old, meta=key['meta'], correction_request_id=req_id)
                if req_id in raw:
                    answer = raw[req_id]
                    row.update(answer)
                    row.update(original_task_uid=old.get('task_uid'),
                        task_uid=hashlib.sha256((req_id+'|'+key['dialogue_id']).encode()).hexdigest()[:16],
                        reader_provenance=answer.get('provenance', 'fresh_corrected_context_pinned_local_'+a.reader))
                else:
                    assert key['original_context_sha256'] == key['corrected_context_sha256']
                    row['reader_provenance'] = 'reused_original_raw_exact_unchanged_context'
                fresh_rows.append(row)
            write(staged / raw_rel, fresh_rows)
        tag, seed, ratio = ('mw' if ds == 'multiwoz' else ds), pairs[0][0]['sample_seed'], round(pairs[0][0]['meta']['ratio'] * 100)
        if probe == 'P1':
            aggregate_rel = f'results/main/{tag}_r{ratio}_s{seed}_aggregate.json'
            metrics = {'p1_em_strict':'p1_correct', 'p1_em_loose':'p1_correct_loose', 'p1_token_f1':'p1_token_f1'}
        elif probe == 'P2':
            aggregate_rel = str(Path(rel).with_name(Path(rel).name.replace('_scored.jsonl', '_aggregate.json')))
            metrics = {'p3_em_strict':'p3_correct', 'p3_em_loose':'p3_correct_loose',
                       'p3_overlap_strict':'p3_overlap_strict', 'p3_overlap_loose':'p3_overlap_loose',
                       'abstain_rate':'abstain', 'support_in_ctx_rate':'support_in_ctx', 'err_rate':'err'}
        else:
            aggregate_rel = rel.replace('_scored_', '_aggregate_').removesuffix('.jsonl') + '.json'
            metrics = {'p1late_story_label_rate':'p1late_story_label_correct',
                       'p1late_value_rate':'p1late_value_correct', 'p1late_value_rate_strict':'p1late_value_correct_strict',
                       'p1late_combined_rate':'p1late_combined', 'p1late_combined_rate_strict':'p1late_combined_strict',
                       'p1late_token_f1':'p1late_token_f1', 'abstain_rate':'abstain',
                       'support_in_ctx_rate':'support_in_ctx', 'err_rate':'err'}
        agg = json.loads((staged / aggregate_rel).read_text())
        target = agg[ds] if ds in agg else agg
        rs = [r['score'] for _, r in pairs]
        value = dict(n=len(rs))
        for metric, field in metrics.items():
            value[metric] = statistics.mean(r.get(field, int(bool(r.get('error'))) if field == 'err' else 0) for r in rs)
        if probe == 'P3':
            value.update(p1late_embed_sim=None, p1late_embed_n=0)
        target[METHOD] = value
        dump(staged / aggregate_rel, agg)
    # All readers share probes; applying the same verified revised selection is idempotent.
    by_probe = defaultdict(dict)
    for key in keys:
        by_probe[key['original_probe_path']][key['dialogue_id']] = key
    for rel, replacements in by_probe.items():
        rows = read(staged / rel)
        if not (staged / historical / rel).exists():
            write(staged / historical / rel, [r for r in read(original / rel) if r['method'] == METHOD])
        rows = read(original / rel)
        for row in rows:
            if row['method'] != METHOD:
                continue
            fixed = replacements[row['dialogue_id']]
            # Keep the public dataset row schema small. Complete request/run
            # provenance lives in data/llmlingua_correction/<reader>/key.jsonl.
            for field in ['compressed_text_used','compressed_text_chars_used',
                          'compressed_text_chars','prompt_user','meta',
                          'original_meta_k_turns','original_compressed_text_chars']:
                row[field] = fixed[field]
        write(staged / rel, rows)
    for path in (staged / 'data/construction').glob('*_compressed_r*.jsonl'):
        ds = path.name.split('_compressed')[0]
        ratio = int(path.stem.rsplit('r',1)[1]) / 100
        if ds not in ('sgd', 'multiwoz') or ratio not in (.1, .3):
            continue
        comparison = {(r['dataset'], r['ratio'], r['dialogue_id']): r
                      for r in read(staged / 'data/llmlingua_correction/context_comparisons.jsonl') if r['family'] == 'main'}
        rows = read(path)
        for row in rows:
            c = comparison[ds, ratio, row['dialogue_id']]
            # The original r=.10 pool mixes text and cache-wrapper dictionaries.
            # Publish this corrected method as text consistently; selected indices
            # and original source hashes are retained in the comparison evidence.
            row['methods'][METHOD] = c['corrected_context']
        write(path, rows)
    if a.reader == 'llama':
        # Update the complete P1 inference chain, retaining other methods and baselines.
        old_key = read(original / 'data/reader_evaluation/key.jsonl')
        old_raw = {r['id']: r for r in read(original / 'results/reader_evaluation/answers.jsonl')}
        requests = {r['id']: r for r in read(original / 'data/reader_evaluation/inputs.jsonl')}
        answers = dict(old_raw)
        p1 = {(r['dataset'],r['sample_seed'],r['dialogue_id']):r for r in keys if r['paper_probe'] == 'P1'}
        from prepare_main_reader import request
        combined = []
        for row in old_key:
            if row['paper_probe'] != 'P1' or row['method'] != METHOD:
                combined.append(row)
                continue
            new = p1[row['dataset'],row['sample_seed'],row['dialogue_id']]
            req = request(new, 'P1')
            assert req['id'] == new['check_id']
            requests[req['id']] = req
            if req['id'] in raw:
                answer = raw[req['id']]
            else:
                assert new['original_context_sha256'] == new['corrected_context_sha256']
                assert requests[row['check_id']]['prompt_user'] == new['prompt_user']
                answer = dict(old_raw[row['check_id']], id=req['id'],
                              provenance='reused_original_raw_exact_unchanged_context', original_id=row['check_id'])
            answers[req['id']] = answer
            combined.append(new)
        keep = {r['check_id'] for r in combined}
        assert keep <= requests.keys() and keep <= answers.keys()
        write(staged / 'data/reader_evaluation/key.jsonl', combined)
        write(staged / 'data/reader_evaluation/inputs.jsonl', [requests[k] for k in sorted(keep)])
        write(staged / 'results/reader_evaluation/answers.jsonl', [answers[k] for k in sorted(keep)])
        main = json.loads((staged / 'results/main/seed_summary.json').read_text())
        for cell, methods in main['p1'].items():
            vals = [json.loads((staged / f'results/main/{"mw" if cell.startswith("multiwoz") else "sgd"}_r30_s{s}_aggregate.json').read_text())[METHOD] for s in (42,43,44)]
            methods[METHOD] = {k:moments([v[k] for v in vals]) for k in vals[0]}
        for cell, methods in main['p3'].items():
            if not cell.endswith(('r10','r30')):
                continue
            vals = [json.loads((staged / f'results/main/{cell}_s{s}_p3_aggregate.json').read_text())[('multiwoz' if cell.startswith('mw') else 'sgd')][METHOD] for s in (42,43,44)]
            methods[METHOD] = {k:moments([v[k] for v in vals]) for k in vals[0]}
        dump(staged / 'results/main/seed_summary.json', main)
        subprocess.run([sys.executable,str(staged/'scripts/reader_evaluation/score_answers.py'),
                        '--artifact',str(staged),'--key',str(staged/'data/reader_evaluation/key.jsonl'),
                        '--answers',str(staged/'results/reader_evaluation/answers.jsonl'),
                        '--output',str(staged/'results/reader_evaluation/scored')],check=True)
    else:
        fp = staged / 'results/diagnostics/mistral_128/seed_summary_mistral128.json'
        report = json.loads(fp.read_text())
        for cell in report['cells'].values():
            vals = []
            for name in cell['files']:
                agg = json.loads((fp.parent / name).read_text())
                vals.append((agg['sgd'] if 'sgd' in agg else agg['multiwoz'])[METHOD])
            cell['methods'][METHOD] = {k:moments([v[k] for v in vals]) for k in vals[0]}
        dump(fp, report)
    evidence_data = staged / f'data/llmlingua_correction/{a.reader}'
    evidence_results = staged / f'results/llmlingua_correction/{a.reader}'
    for name in ['inputs.jsonl','key.jsonl','reused_original_scores.jsonl','reused_raw_answers.jsonl','preparation.json']:
        evidence_data.mkdir(parents=True,exist_ok=True)
        if name == 'preparation.json':
            preparation = json.loads((a.prepared/name).read_text())
            preparation.pop('model_local_path', None)
            dump(evidence_data/name, preparation)
        else:
            shutil.copy2(a.prepared/name,evidence_data/name)
    if a.reader == 'llama':
        for name in ['inputs_p1_batch8.jsonl','inputs_p2_serial.jsonl']:
            shutil.copy2(a.prepared/name,evidence_data/name)
    for name in ['expanded_scored.jsonl','summary.json']:
        evidence_results.mkdir(parents=True,exist_ok=True)
        shutil.copy2(a.scored/name,evidence_results/name)
    fresh_ids = {r['id'] for fp in a.answers for r in read(fp)}
    # Reused original raw responses already have their own, explicitly marked
    # prepared file; do not duplicate them in the fresh primary answer file.
    write(evidence_results/'answers.jsonl', [raw[k] for k in sorted(fresh_ids)])
    if a.reader == 'llama':
        semantic_data = staged / 'data/llmlingua_correction'
        for src, dest in [('p1_corrected_key.jsonl','semantic_corrected_p1_key.jsonl'),
                          ('reused_judgments.jsonl','semantic_cached_exact_judgments.jsonl'),
                          ('missing_inputs.jsonl','semantic_pending_inputs.jsonl')]:
            shutil.copy2(a.scored/'semantic'/src,semantic_data/dest)
        pending = summary['missing_p1_unique_judge_pairs']
        cached = summary['reused_p1_unique_judge_pairs']
        dump(semantic_data/'semantic_pending_scope.json',dict(
            status='incomplete_not_a_headline_condition', expanded_answers=1200,
            distinct_reference_answer_pairs=pending+cached, cached_exact_pair_judgments=cached,
            pending_distinct_pairs=pending, paid_api_calls_performed=0,
            excluded_condition=METHOD,
            reason='No fresh paid judgments; missing coverage is not imputed or scored as failure. Entire LLMLingua condition omitted from semantic headline.',
            key='semantic_corrected_p1_key.jsonl', cached_labels='semantic_cached_exact_judgments.jsonl',
            pending_inputs='semantic_pending_inputs.jsonl',
            historical_full_cache='results/historical/semantic_p1_legacy_20260929'))
        fp = staged/'data/semantic_evaluation/p1_scope.json'
        scope=json.loads(fp.read_text());scope['new_llmlingua_unique_pairs_without_judgment']=pending;dump(fp,scope)
        subprocess.run([sys.executable,str(staged/'scripts/semantic_evaluation/aggregate_equivalence.py'),
                        '--output',str(staged/'results/semantic_evaluation/p1_summary.json')],check=True)
    print(f'OK: integrated {a.reader} {len(scored)} LLMLingua expanded rows; unaffected method scores preserved; original LLMLingua rows archived')


if __name__ == '__main__':
    main()

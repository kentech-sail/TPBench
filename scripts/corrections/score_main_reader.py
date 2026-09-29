"""Score corrected LLMLingua main requests and identify missing P1 judge pairs.

Writes an isolated correction bundle; never edits published baseline evidence.
Paid judge inference is not performed. Existing judgments are reused only for
the exact unchanged reference/candidate pair, across all published methods.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys


def read(path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def semantic_id(reference, candidate):
    pair = json.dumps([reference, candidate], ensure_ascii=False)
    return hashlib.sha256(('TPBench semantic audit|' + pair).encode()).hexdigest()[:20]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--answers', type=Path, nargs='+', required=True)
    parser.add_argument('--canonical-scorer-dir', type=Path, required=True)
    parser.add_argument('--reader-name', choices=['llama', 'mistral'], default='llama')
    parser.add_argument('--probe', choices=['P1', 'P2', 'P3'], nargs='+',
        help='Produce a clearly scoped completed subset while another attention run is pending')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    artifact, output = args.artifact.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(args.canonical_scorer_dir.resolve()), str(artifact / 'scripts/scorer'), str(artifact / 'scripts')]
    from scorer_p1 import score_p1, parse_reader_text
    from scorer_p3 import score_p3
    from scorer_p1_late import score_late, support_in_ctx
    from scorer_paper_p3 import score_saved_row
    keys = read(args.prepared / 'key.jsonl')
    if args.probe:
        keys = [r for r in keys if r['paper_probe'] in args.probe]
    reused = {(r['dataset'], r['sample_seed'], r['paper_probe'], r['ratio'], r['dialogue_id']): r
        for r in read(args.prepared / 'reused_original_scores.jsonl')}
    raw = {}
    for path in [args.prepared / 'reused_raw_answers.jsonl'] + args.answers:
        for row in read(path):
            if row['id'] in raw:
                raise ValueError(f'duplicate raw answer {row["id"]}')
            raw[row['id']] = dict(row, answer_source=str(path))
    # The corrected release archives the complete original blinded pair cache;
    # the current headline key deliberately excludes LLMLingua. Rescoring the
    # released corrected requests must still use the complete exact-pair cache.
    historic = artifact / 'results/historical/semantic_p1_legacy_20260929'
    semantic_artifact = historic if historic.exists() else artifact
    old_semantic = {r['id']: r for r in read(semantic_artifact / 'data/semantic_evaluation/p1_inputs.jsonl')}
    old_judgments = {r['id']: r for r in read(semantic_artifact / 'results/semantic_evaluation/p1/judgments.jsonl')}
    old_p1_key = {(r['dataset'], r['seed'], r['dialogue_id'], r['method']): r
        for r in read(semantic_artifact / 'data/semantic_evaluation/p1_key.jsonl')}
    assert set(old_semantic) == set(old_judgments)
    scored, missing_pairs, semantic_keys, reused_pairs = [], {}, [], {}
    groups = defaultdict(list)
    for key in keys:
        ds, seed, probe, ratio, did = key['dataset'], key['sample_seed'], key['paper_probe'], key['meta']['ratio'], key['dialogue_id']
        reuse = reused.get((ds, seed, probe, ratio, did))
        if reuse:
            stored = dict(reuse['score'])
            pred = '' if stored.get('abstain') else (stored.get('pred_value') or '')
            parsed = dict(value=pred, abstain=bool(stored.get('abstain')))
            provenance = reuse['provenance']
            raw_text = None  # No invented raw response for a saved parsed prediction.
            input_tokens = old_p1_key[ds, seed, did, key['method']]['input_tokens'] if probe == 'P1' else None
            format_error = old_p1_key[ds, seed, did, key['method']]['format_error'] if probe == 'P1' else None
        else:
            answer = raw.get(key['check_id'])
            if answer is None:
                raise ValueError(f'missing corrected answer: {key["check_id"]}')
            raw_text = answer['reader_output_text']
            parsed = parse_reader_text(raw_text)
            parsed = parsed if isinstance(parsed, dict) else None
            provenance = answer.get('provenance', f'fresh_corrected_context_pinned_local_{args.reader_name}')
            stored = dict(dialogue_id=did, dataset=ds, method=key['method'], probe_type=key['probe_type'],
                meta=key['meta'], error=answer.get('error'))
            stored['format_error'] = int(parsed is None)
            input_tokens = answer.get('input_tokens')
            format_error = int(parsed is None)
        if probe == 'P1':
            value = score_p1(key, parsed)
            stored.update(gold=value['gold'], pred_value=value['pred'], abstain=value['abstain'],
                p1_correct=value['p1_correct'], p1_correct_loose=value['p1_correct_loose'], p1_token_f1=value['p1_token_f1'])
            strict, loose = value['p1_correct'], value['p1_correct_loose']
            semid = semantic_id(value['gold'], value['pred'])
            pair = dict(id=semid, question='What did the user initially want?', reference=value['gold'], candidate=value['pred'])
            if semid in old_semantic:
                assert old_semantic[semid] == pair
                reused_pairs[semid] = old_judgments[semid]
            else:
                missing_pairs[semid] = pair
            semantic_keys.append(dict(id=key['check_id'], original_check_id=key['original_check_id'],
                dataset=ds, seed=seed, dialogue_id=did, method=key['method'], semantic_id=semid,
                strict=strict, loose=loose, token_f1=value['p1_token_f1'], pred=value['pred'], gold=value['gold'],
                abstain=int(value['abstain']), source=did.split('__qa_')[0], probe='P1', condition='initial_goal',
                input_tokens=input_tokens, format_error=format_error))
        elif probe == 'P2':
            if not reuse:
                stored.update(score_p3(raw_text, key['gold'], key['compressed_text_used'], ds))
                stored['err'] = int(bool(stored.get('error')) or parsed is None)
            strict, loose = stored['p3_correct'], stored['p3_correct_loose']
        else:
            if not reuse:
                value = score_late(key, parsed)
                stored.update(value)
                stored.update(gold_story_label=key['gold_story_label'], gold_value=key['gold_value'], pred_value=value['pred'])
                stored['support_in_ctx'] = support_in_ctx(parsed, key['compressed_text_used'])
            separated = score_saved_row(stored)
            stored.update(p3_joint_strict=separated['p3_joint_strict'], p3_joint_loose=separated['p3_joint_loose'],
                p3_segment_parse=separated['p3_segment_parse'])
            strict, loose = separated['p3_joint_strict'], separated['p3_joint_loose']
        stored.update(meta=key['meta'], correction_provenance=provenance, correction_check_id=key['check_id'])
        row = dict(dataset=ds, seed=seed, paper_probe=probe, ratio=ratio, dialogue_id=did,
            check_id=key['check_id'], strict=strict, loose=loose, score=stored, provenance=provenance)
        scored.append(row)
        groups[ds, seed, probe, ratio].append(row)
    cells = [dict(dataset=ds, seed=seed, paper_probe=probe, ratio=ratio, method='llmlingua2_cache', n=len(rows),
        strict=statistics.mean(r['strict'] for r in rows), loose=statistics.mean(r['loose'] for r in rows))
        for (ds, seed, probe, ratio), rows in sorted(groups.items())]
    for cell in cells:
        rows = groups[cell['dataset'], cell['seed'], cell['paper_probe'], cell['ratio']]
        if cell['paper_probe'] == 'P1':
            cell['token_f1'] = statistics.mean(r['score']['p1_token_f1'] for r in rows)
        elif cell['paper_probe'] == 'P3':
            cell['parse_rate'] = statistics.mean(r['score']['p3_segment_parse'] for r in rows)
            cell['whole_field_strict'] = statistics.mean(r['score']['p1late_combined_strict'] for r in rows)
            cell['whole_field_loose'] = statistics.mean(r['score']['p1late_combined'] for r in rows)
    across_seed = defaultdict(list)
    for cell in cells:
        across_seed[cell['dataset'], cell['paper_probe'], cell['ratio']].append(cell)
    means = []
    for (ds, probe, ratio), rows in sorted(across_seed.items()):
        row = dict(dataset=ds, paper_probe=probe, ratio=ratio, method='llmlingua2_cache',
            seeds=[r['seed'] for r in rows], n_by_seed=[r['n'] for r in rows])
        for metric in ['strict', 'loose'] + (['token_f1'] if probe == 'P1' else
                ['parse_rate', 'whole_field_strict', 'whole_field_loose'] if probe == 'P3' else []):
            values = [r[metric] for r in rows]
            row[metric] = dict(mean=statistics.mean(values), std=statistics.stdev(values))
        means.append(row)
    write(output / 'expanded_scored.jsonl', scored)
    write(output / 'semantic/p1_corrected_key.jsonl', semantic_keys)
    write(output / 'semantic/missing_inputs.jsonl', [missing_pairs[k] for k in sorted(missing_pairs)])
    write(output / 'semantic/reused_judgments.jsonl', [reused_pairs[k] for k in sorted(reused_pairs)])
    summary = dict(complete=True, reader=args.reader_name, scope_probes=sorted({r['paper_probe'] for r in keys}),
        expanded_scored=len(scored), cells=cells, means=means,
        missing_p1_unique_judge_pairs=len(missing_pairs), reused_p1_unique_judge_pairs=len(reused_pairs),
        semantic_status=('not_applicable_no_P1' if not semantic_keys else
            'partial_pending_missing_paid_judgments' if missing_pairs else 'all_corrected_pairs_cached'),
        paid_api_calls_performed=0)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'cells'}, indent=2))


if __name__ == '__main__':
    main()

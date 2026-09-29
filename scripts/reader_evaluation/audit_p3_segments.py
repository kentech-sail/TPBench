"""CPU-only primary segment-local P3 scores and whole-field sensitivity.

Usage: python3 audit_p3_segments.py --artifact PATH --output PATH
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import runpy
import statistics
import sys


def split_answer(pred):
    # A conservative syntax diagnostic, not semantic correctness annotation.
    if pred.count(';') != 1:
        return None
    goal, right = (p.strip() for p in pred.split(';', 1))
    if ':' not in right:
        return None
    label, value = (p.strip() for p in right.split(':', 1))
    return (goal, label, value) if goal and label and value else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifact', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    root, out = args.artifact.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root / 'scripts/scorer'))
    sys.path.insert(0, str(root / 'scripts'))
    source = root / 'scripts/scorer/scorer_p1_late.py'
    mod = runpy.run_path(str(source))
    score, tokens = mod['score_late'], mod['content_tokens']
    phrase, norm = mod['normalized_phrase_occurs'], mod['norm_text']
    # Guard against arbitrary natural-language guessing in the parser.
    assert split_answer('book a hotel; time: 12:30 pm') == ('book a hotel', 'time', '12:30 pm')
    for bad in ['book a hotel', '; area: east', 'book; east', 'book; area:', 'book; area: east; extra']:
        assert split_answer(bad) is None
    def segmented(gold_goal, gold_value, pred):
        parts = split_answer(pred)
        if parts is None:
            return 0, 0, 0
        g, _, v = parts
        gp = int(bool(tokens(gold_goal) & tokens(g)))
        vp = int(phrase(norm(gold_value), norm(v)))
        vl = int(vp or (bool(tokens(gold_value)) and
                       len(tokens(gold_value) & tokens(v)) >= max(1, len(tokens(gold_value)) // 2)))
        return gp, vp, vl
    assert segmented('find a hotel in the east', 'east', 'book a restaurant; area: east') == (0, 1, 1)
    assert segmented('find a hotel', 'east', 'find a hotel in the east; area: west') == (1, 0, 0)
    groups, all_rows, manifest, seen = defaultdict(list), [], {}, set()
    for reader in ['llama', 'mistral', 'chunkkv']:
        paths = sorted((root / 'results/diagnostics/update_evidence').glob(f'p3_*_scored_{reader}.jsonl'))
        assert len(paths) == (2 if reader == 'chunkkv' else 6)
        for path in paths:
            manifest[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.read_text().splitlines():
                r = json.loads(line)
                seed, dataset, method = r['meta']['sample_seed'], r['dataset'], r['method']
                key = reader, dataset, seed, r['dialogue_id'], method
                assert key not in seen, key
                seen.add(key)
                gold = {'gold_story_label': r['gold_story_label'], 'gold_value': r['gold_value']}
                saved = score(gold, {'value': r['pred_value'], 'abstain': r['abstain']})
                for field in ['p1late_story_label_correct', 'p1late_value_correct',
                              'p1late_value_correct_strict', 'p1late_combined', 'p1late_combined_strict']:
                    assert saved[field] == r[field], (key, field)
                pred = '' if r['abstain'] else r['pred_value']
                parts = split_answer(pred)
                gp, vs, vl = segmented(r['gold_story_label'], r['gold_value'], pred)
                original_s, original_l = r['p1late_combined_strict'], r['p1late_combined']
                separated_s, separated_l = gp * vs, gp * vl
                assert separated_s <= original_s and separated_l <= original_l
                row = dict(reader=reader, dataset=dataset, seed=seed, method=method,
                           dialogue_id=r['dialogue_id'], source_dialogue=r['dialogue_id'].split('__qa_')[0],
                           gold_goal=r['gold_story_label'], gold_value=r['gold_value'], pred=pred,
                           parsed=parts is not None, goal_part=parts[0] if parts else None,
                           value_part=parts[2] if parts else None,
                           original_strict=original_s, original_loose=original_l,
                           separated_strict=separated_s, separated_loose=separated_l,
                           original_goal=r['p1late_story_label_correct'], separated_goal=gp,
                           goal_cross_segment=int(parts is not None and r['p1late_story_label_correct'] and not gp),
                           value_cross_segment_strict=int(parts is not None and r['p1late_value_correct_strict'] and not vs))
                all_rows.append(row)
                groups[reader, dataset, seed, method].append(row)
    assert len(all_rows) == 19600
    cells = []
    metrics = ['original_strict', 'original_loose', 'separated_strict', 'separated_loose']
    for (reader, ds, seed, method), rows in sorted(groups.items()):
        assert len(rows) == 200
        parsed = [r for r in rows if r['parsed']]
        cell = dict(reader=reader, dataset=ds, seed=seed, method=method, n=len(rows), parsed_n=len(parsed))
        for m in metrics:
            cell[m] = statistics.mean(r[m] for r in rows)
            cell['parsed_' + m] = statistics.mean(r[m] for r in parsed) if parsed else None
        for kind in ['strict', 'loose']:
            cell['lost_unparseable_' + kind] = sum(r['original_' + kind] for r in rows if not r['parsed'])
            cell['lost_cross_segment_' + kind] = sum(r['original_' + kind] - r['separated_' + kind] for r in parsed)
            assert sum(r['original_' + kind] - r['separated_' + kind] for r in rows) == (
                cell['lost_unparseable_' + kind] + cell['lost_cross_segment_' + kind])
        cell['goal_cross_segment_n'] = sum(r['goal_cross_segment'] for r in rows)
        cell['value_cross_segment_strict_n'] = sum(r['value_cross_segment_strict'] for r in rows)
        cells.append(cell)
    method_groups = defaultdict(list)
    for c in cells:
        method_groups[c['reader'], c['dataset'], c['method']].append(c)
    means = []
    for (reader, ds, method), cs in sorted(method_groups.items()):
        d = dict(reader=reader, dataset=ds, method=method, seeds=[c['seed'] for c in cs])
        for m in metrics:
            d[m] = statistics.mean(c[m] for c in cs)
            d[m + '_seed_std'] = statistics.stdev(c[m] for c in cs) if len(cs)>1 else None
        d['parsed_rate'] = statistics.mean(c['parsed_n']/c['n'] for c in cs)
        for k in ['strict', 'loose']:
            for reason in ['unparseable', 'cross_segment']:
                d['lost_' + reason + '_' + k] = sum(c['lost_' + reason + '_' + k] for c in cs)
        means.append(d)
    # Paired, source-dialogue cluster bootstrap of full minus selector differences.
    # The same source sampled in multiple seed sets remains one cluster.
    import numpy as np
    rng = np.random.default_rng(20260928)
    contrasts, common_parseable = [], []
    by_setup = defaultdict(list)
    for r in all_rows:
        if r['reader'] != 'chunkkv':
            by_setup[r['reader'], r['dataset']].append(r)
    for (reader, ds), rows in sorted(by_setup.items()):
        full = {(r['seed'], r['dialogue_id']): r for r in rows if r['method']=='full_context'}
        for method in sorted({r['method'] for r in rows} - {'full_context'}):
            cl = defaultdict(list)
            common = []
            for r in rows:
                if r['method'] == method:
                    f = full[r['seed'], r['dialogue_id']]
                    cl[r['source_dialogue']].append([f[m]-r[m] for m in metrics])
                    if f['parsed'] and r['parsed']:
                        common.append((f,r))
            common_parseable.append(dict(reader=reader,dataset=ds,method=method,n=len(common),
                metrics={m:dict(full=statistics.mean(f[m] for f,r in common),
                                compressed=statistics.mean(r[m] for f,r in common)) for m in metrics} if common else {}))
            counts = np.array([len(v) for v in cl.values()])
            sums = np.array([np.array(v).sum(axis=0) for v in cl.values()])
            draws = rng.integers(len(cl), size=(5000,len(cl)))
            boots = sums[draws].sum(axis=1)/counts[draws].sum(axis=1)[:,None]
            ci = np.quantile(boots,[.025,.975],axis=0)
            contrasts.append(dict(reader=reader,dataset=ds,method=method,clusters=len(cl),n=int(counts.sum()),
                                  differences={m:dict(mean=float(sums[:,i].sum()/counts.sum()),
                                                     ci95=[float(ci[0,i]),float(ci[1,i])]) for i,m in enumerate(metrics)}))
    for p in [source, root/'scripts/normalize.py']:
        manifest[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    result = dict(n=len(all_rows), parser='one semicolon; nonempty goal; nonempty label:value; time colons preserved',
                  interpretation='Primary segment-local lexical scoring compared with the original whole-field scorer as sensitivity; neither is semantic evaluation. Unparseable is a format outcome, not evidence of semantic error.',
                  bootstrap=dict(repetitions=5000, seed=20260928, unit='source dialogue, paired, pooled across three seed sets'),
                  cells=cells, means=means, full_context_contrasts=contrasts,
                  common_parseable_contrasts=common_parseable, input_sha256=manifest)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'per_answer.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_rows))
    changed = [r for r in all_rows if r['original_strict'] != r['separated_strict'] or r['original_loose'] != r['separated_loose']]
    (out/'changed_answers.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in changed))
    lines=['# Primary P3 segment scoring and whole-field sensitivity', '', 'Both scorers are recomputed from the saved reader outputs. Segment-local scores are primary; original whole-field scores are a sensitivity analysis.', '',
           'Unparseable answers receive zero under primary segment-local scoring. Format loss and parsed cross-segment loss are reported separately.', '',
           '| Reader | Dataset | Method | Original loose / strict | Segment-local loose / strict | Parse rate |',
           '|---|---|---|---|---|---|']
    for d in means:
        lines.append(f"| {d['reader']} | {d['dataset']} | {d['method']} | {d['original_loose']:.3f} / {d['original_strict']:.3f} | {d['separated_loose']:.3f} / {d['separated_strict']:.3f} | {d['parsed_rate']:.3f} |")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(verified_answers=len(all_rows), changed_answers=len(changed), output=str(out))))


if __name__ == '__main__':
    main()

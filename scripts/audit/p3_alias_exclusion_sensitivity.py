"""Exclude P2-alias-equivalent examples from original P3 saved generations.

This is a heuristic sensitivity subset, not a replacement n=200 experiment.
No reader is called, no pool is resampled, and original contexts/prompts stay
fixed. Per-seed counts and original/excluded scores are reported explicitly.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/builders'))
sys.path.insert(0, str(ROOT / 'scripts/scorer'))
from construction import is_alias_equivalent
from scorer_paper_p3 import load_jsonl, score_saved_row, SCORER


def evaluate(root):
    labels, hashes = {}, {}
    for dataset in ('sgd', 'multiwoz'):
        path = root / f'results/diagnostics/update_evidence/{dataset}_goal_labels.jsonl'
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in load_jsonl(path):
            labels[dataset, row['dialogue_id']] = row
    groups, seen, excluded = defaultdict(list), set(), {}
    for reader in ('llama', 'mistral', 'chunkkv'):
        paths = sorted((root / 'results/diagnostics/update_evidence').glob(f'p3_*_scored_{reader}.jsonl'))
        if len(paths) != (2 if reader == 'chunkkv' else 6):
            raise ValueError(f'incomplete original P3 saved evidence for {reader}')
        for path in paths:
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
            for row in load_jsonl(path):
                ds, seed, did, method = row['dataset'], row['meta']['sample_seed'], row['dialogue_id'], row['method']
                key = reader, ds, seed, did, method
                if key in seen:
                    raise ValueError(f'duplicate P3 answer: {key}')
                seen.add(key)
                label = labels[ds, did]
                is_alias = is_alias_equivalent(label['old_value'], label['new_value'], ds)
                groups[reader, ds, seed, method].append((score_saved_row(row), is_alias))
                if is_alias:
                    excluded[ds, seed, did] = dict(dataset=ds, seed=seed, dialogue_id=did,
                        slot=label['slot'], old_value=label['old_value'], new_value=label['new_value'])
    cells = []
    for (reader, ds, seed, method), rows in sorted(groups.items()):
        retained = [r for r, alias in rows if not alias]
        if len(rows) != 200 or not retained:
            raise ValueError(f'invalid original/sensitivity count: {(reader, ds, seed, method)}')
        cell = dict(reader=reader, dataset=ds, seed=seed, method=method,
            original_n=len(rows), excluded_n=len(rows) - len(retained), n=len(retained))
        for name, field in [('strict', 'p3_joint_strict'), ('loose', 'p3_joint_loose')]:
            cell[name] = statistics.mean(r[field] for r in retained)
            cell['original_' + name] = statistics.mean(r[field] for r, _ in rows)
        cells.append(cell)
    method_groups = defaultdict(list)
    for cell in cells:
        method_groups[cell['reader'], cell['dataset'], cell['method']].append(cell)
    means = []
    for (reader, ds, method), cs in sorted(method_groups.items()):
        row = dict(reader=reader, dataset=ds, method=method, seeds=[c['seed'] for c in cs],
            n_by_seed=[c['n'] for c in cs], excluded_n_by_seed=[c['excluded_n'] for c in cs])
        for name in ('strict', 'loose', 'original_strict', 'original_loose'):
            values = [c[name] for c in cs]
            row[name] = statistics.mean(values)
            row[name + '_seed_std'] = statistics.stdev(values) if len(values) > 1 else None
        means.append(row)
    for path in [root / 'scripts/builders/construction.py', root / 'scripts/normalize.py',
                 root / 'scripts/scorer/scorer_p1_late.py',
                 root / 'scripts/semantic_evaluation/audit_p3_segments.py',
                 Path(__file__).resolve(), ROOT / 'scripts/scorer/scorer_paper_p3.py']:
        # A patched overlay may reside outside the immutable original artifact.
        relative = str(path.relative_to(root)) if path.is_relative_to(root) else str(path.relative_to(ROOT))
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(scorer=SCORER, saved_answers=len(seen),
        interpretation='Heuristic sensitivity subset of original P3 generations; not a rebuilt n=200 pool or a new inference run.',
        exclusion='The same is_alias_equivalent(old_value,new_value,dataset) function used by P2; domain-sensitive heuristic, not independent semantic labels.',
        averaging='Unweighted mean of per-seed accuracies; variable denominators shown for every seed.',
        cells=cells, means=means, excluded_examples=[excluded[k] for k in sorted(excluded)], input_sha256=hashes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.artifact.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(saved_answers=result['saved_answers'], cells=len(result['cells']), output=str(args.output))))


if __name__ == '__main__':
    main()

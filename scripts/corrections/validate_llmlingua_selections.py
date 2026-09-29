"""Replay both LLMLingua selectors from saved word labels, without inference."""
import hashlib
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/fmts_refpool/core'))
from compression_methods import mean_llmlingua_word_labels


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    data = ROOT / 'data/llmlingua_correction'
    labels = read(data / 'turn_word_labels.jsonl')
    cache = {r['sha256']: r for r in labels}
    assert len(cache) == len(labels) == 33166
    for row in labels:
        assert row['sha256'] == digest(row['text'])
        corrected = mean_llmlingua_word_labels(row['labeled_text']) if row['text'] else 0.
        values = [float(x) for x in re.findall(r'(\d+(?:\.\d+)?)', row['labeled_text'])]
        legacy = statistics.mean(values) if values else 0.
        assert corrected == row['corrected_score'], row['sha256']
        assert legacy == row['legacy_score'], row['sha256']
    source = {}
    for ds in ('sgd', 'multiwoz'):
        for row in read(ROOT / f'data/construction/{ds}_compressed_r30.jsonl'):
            turns = re.split(r'^\[T\d+\] (?:USER|SYSTEM): ', row['methods']['full_context'], flags=re.M)[1:]
            source['main', ds, row['dialogue_id']] = [t.rstrip('\n') for t in turns]
    for ds in ('indirect_seed42', 'sgd_scope', 'multiwoz_scope'):
        for row in read(ROOT / f'data/fmts/fmts_{ds}.jsonl'):
            source['fmts', ds, row['dialogue_id']] = [t['text'] for t in row['turns']]
    comparisons = read(data / 'context_comparisons.jsonl')
    assert len(comparisons) == 4374
    keys = set()
    groups = defaultdict(list)
    for row in comparisons:
        key = row['family'], row['dataset'], row['dialogue_id'], row['ratio']
        assert key not in keys
        keys.add(key)
        turns = source[key[:3]]
        assert len(turns) == row['n_turns']
        k = max(1, int(len(turns) * row['ratio']))
        assert row['k'] == k
        scores = [cache[digest(t.strip())] for t in turns]
        for prefix, score in [('corrected', 'corrected_score'), ('legacy', 'legacy_score')]:
            indices = sorted(sorted(range(len(turns)), key=lambda i: scores[i][score], reverse=True)[:k])
            text = ' '.join(turns[i] for i in indices)
            assert indices == row[f'{prefix}_indices'], key
            assert text == row[f'{prefix}_context'], key
            assert (text == row['saved_context']) is row[f'{prefix}_matches_saved'], key
        assert row['legacy_matches_saved'], key
        groups[row['family'], row['dataset'], row['ratio']].append(row)
    actual = [dict(family=k[0], dataset=k[1], ratio=k[2], n=len(rs),
                   legacy_matches_saved=sum(r['legacy_matches_saved'] for r in rs),
                   corrected_matches_saved=sum(r['corrected_matches_saved'] for r in rs),
                   corrected_differs_legacy=sum(r['corrected_context'] != r['legacy_context'] for r in rs))
              for k, rs in sorted(groups.items())]
    expected = json.loads((data / 'selection_impact_summary.json').read_text())
    assert actual == expected['groups']
    assert sum(r['n'] for r in actual if r['family'] == 'main') == 3374
    assert sum(r['corrected_differs_legacy'] for r in actual if r['family'] == 'main') == 3300
    assert sum(r['n'] for r in actual if r['family'] == 'fmts') == 1000
    assert sum(r['corrected_differs_legacy'] for r in actual if r['family'] == 'fmts') == 792
    print('OK: 33,166 saved word-label rows; legacy selections exactly replay 3,374 main / 1,000 FMTS contexts; corrected contexts differ in 3,300 / 792 cases')


if __name__ == '__main__':
    main()

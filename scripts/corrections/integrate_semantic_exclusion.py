"""Stage the explicitly covered P1 semantic subset; never infer or call an API."""
import argparse
import json
import shutil
from pathlib import Path


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', type=Path, required=True)
    p.add_argument('--staged', type=Path, required=True)
    a = p.parse_args()
    archive = Path('results/historical/semantic_p1_legacy_20260929')
    paths = ['data/semantic_evaluation/p1_key.jsonl', 'data/semantic_evaluation/p1_inputs.jsonl',
             'results/semantic_evaluation/p1_summary.json',
             'results/semantic_evaluation/p1/judgments.jsonl',
             'results/semantic_evaluation/p1/raw_responses.jsonl',
             'results/semantic_evaluation/p1/model_record.json']
    for rel in paths:
        target = a.staged / archive / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(a.original / rel, target)
    key = [r for r in read(a.original / paths[0]) if r['method'] != 'llmlingua2_cache']
    ids = {r['semantic_id'] for r in key}
    assert len(key) == 8400 and len(ids) == 3049
    write(a.staged / paths[0], key)
    for rel in [paths[1], paths[3], paths[4]]:
        rows = [r for r in read(a.original / rel) if r['id'] in ids]
        assert len(rows) == len(ids)
        write(a.staged / rel, rows)
    scope = {
        'status': 'covered_subset_excluding_corrected_llmlingua',
        'expected_pairs': 3049, 'expected_answers': 8400,
        'conditions_including_full_context': 7, 'compressed_selectors': 6,
        'excluded_methods': ['llmlingua2_cache'],
        'reason': 'Corrected LLMLingua reader answers require new semantic judgments. No new paid judgments were obtained. Historical judgments do not validate changed answers.',
        'selection_based_on_evaluation_coverage_not_scores': True,
        'original_pairs': 3500, 'original_answers': 9600,
        'historical_archive': str(archive),
        'new_llmlingua_unique_pairs_without_judgment': len(read(a.staged / 'data/llmlingua_correction/semantic_pending_inputs.jsonl')),
        'pending_inputs': 'data/llmlingua_correction/semantic_pending_inputs.jsonl',
        'pending_inputs_status': 'Awaiting semantic judge; not included in headline scores',
    }
    (a.staged / 'data/semantic_evaluation/p1_scope.json').write_text(json.dumps(scope, indent=2) + '\n')
    (a.staged / archive / 'README.md').write_text(
        '# Historical P1 semantic evidence\n\n'
        'These original 9,600 expanded answers and 3,500 pair judgments were produced '
        'before the LLMLingua selection-parser correction. They reproduce the historical '
        'release, not the corrected LLMLingua condition. The current semantic headline '
        'covers the unchanged six selectors and full context only (8,400 expanded answers; '
        '3,049 distinct judged pairs). No new judgments are claimed.\n')
    print(json.dumps(scope, indent=2))


if __name__ == '__main__':
    main()

"""Regression checks against all saved P3 answers and a raw-reader CLI round trip."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/scorer'))
from scorer_paper_p3 import load_jsonl, score_prediction, score_saved_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, default=ROOT)
    args = parser.parse_args()
    artifact = args.artifact.resolve()
    # Prevent cross-segment credit; preserve colons inside a value; abstention is zero.
    assert score_prediction('find a hotel', 'east', 'find a hotel in east; area: west')['p3_joint_strict'] == 0
    assert score_prediction('find a hotel', 'east', 'book a restaurant; area: east')['p3_joint_loose'] == 0
    assert score_prediction('book a hotel', '12:30 pm', 'book a hotel; time: 12:30 pm')['p3_joint_strict'] == 1
    assert score_prediction('book a hotel', 'east', 'book a hotel; area: east', True)['p3_joint_strict'] == 0
    for bad in ('book a hotel', '; area: east', 'book; area:', 'book; area: east; extra'):
        assert score_prediction('book a hotel', 'east', bad)['p3_segment_parse'] == 0
    expected = {(r['reader'], r['dataset'], r['seed'], r['dialogue_id'], r['method']): r
        for r in load_jsonl(artifact / 'results/p3_segment_sensitivity/per_answer.jsonl')}
    count = 0
    for reader in ('llama', 'mistral', 'chunkkv'):
        for path in sorted((artifact / 'results/diagnostics/update_evidence').glob(f'p3_*_scored_{reader}.jsonl')):
            for row in load_jsonl(path):
                key = reader, row['dataset'], row['meta']['sample_seed'], row['dialogue_id'], row['method']
                result, original = score_saved_row(row), expected.pop(key)
                assert result['p3_joint_strict'] == original['separated_strict'], key
                assert result['p3_joint_loose'] == original['separated_loose'], key
                assert result['p3_segment_parse'] == original['parsed'], key
                count += 1
    assert count == 19600 and not expected
    # Test the public command on raw saved inference, not only the helper function.
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        subprocess.run([sys.executable, str(ROOT / 'scripts/scorer/scorer_paper_p3.py'),
            '--probes', str(artifact / 'data/probes/probes_sgd_s42_r30_n200_late_intent.jsonl'),
            '--reader-out', str(artifact / 'results/diagnostics/update_evidence/p3_sgd_s42_reader_llama.jsonl'),
            '--scored-out', str(out / 'scored.jsonl'), '--aggregate-out', str(out / 'aggregate.json')], check=True)
        actual = {r['dialogue_id'] + '|' + r['method']: r for r in load_jsonl(out / 'scored.jsonl')}
        for row in load_jsonl(artifact / 'results/diagnostics/update_evidence/p3_sgd_s42_scored_llama.jsonl'):
            expected_row = score_saved_row(row)
            assert actual[row['dialogue_id'] + '|' + row['method']] == expected_row
        assert len(actual) == 1600
        aggregate = json.loads((out / 'aggregate.json').read_text())
        assert len(aggregate['cells']) == 8 and all(r['n'] == 200 for r in aggregate['cells'])
    print('OK: canonical paper-P3 scores reproduce 19,600 saved answers and 1,600 raw-reader CLI results')


if __name__ == '__main__':
    main()

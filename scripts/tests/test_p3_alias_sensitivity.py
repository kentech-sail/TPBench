"""Check released alias-exclusion evidence against saved scores and heuristic labels."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/audit'))
from p3_alias_exclusion_sensitivity import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, default=ROOT)
    parser.add_argument('--expected', type=Path)
    args = parser.parse_args()
    result = evaluate(args.artifact.resolve())
    expected_n = {'sgd': {42: 152, 43: 147, 44: 147}, 'multiwoz': {42: 181, 43: 183, 44: 180}}
    assert result['saved_answers'] == 19600 and len(result['cells']) == 98
    for row in result['cells']:
        assert row['original_n'] == 200
        assert row['n'] == expected_n[row['dataset']][row['seed']]
        assert row['excluded_n'] == 200 - row['n']
    if args.expected:
        assert result == json.loads(args.expected.read_text(encoding='utf-8'))
    print('OK: original P3 alias-exclusion sensitivity regenerates, with explicit seed counts and unchanged generations')


if __name__ == '__main__':
    main()

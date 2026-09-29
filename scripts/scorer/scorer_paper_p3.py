"""Canonical paper-P3 segment-local lexical scorer, with no model dependencies.

Paper P3 is serialized as P1_LATE. scorer_p1_late.py remains the original
whole-field sensitivity scorer. This CLI preserves the reported segment-local
semantics, including one goal-token match, phrase-bounded value matching, the
original loose token threshold, and zero credit for unparseable answers.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts/scorer'))
from scorer_p1_late import content_tokens, norm_text, normalized_phrase_occurs, parse_reader_text
from semantic_evaluation.audit_p3_segments import split_answer

SCORER = 'paper_p3_segment_local_v1'


def score_prediction(gold_goal, gold_value, prediction, abstain=False):
    """Apply exactly the segment-local rule used by audit_p3_segments.py."""
    pred = '' if abstain else str(prediction or '')
    parts = split_answer(pred)
    goal, strict, loose = 0, 0, 0
    if parts is not None:
        goal_part, _, value_part = parts
        goal = int(bool(content_tokens(gold_goal) & content_tokens(goal_part)))
        strict = int(normalized_phrase_occurs(norm_text(gold_value), norm_text(value_part)))
        target_tokens = content_tokens(gold_value)
        loose = int(strict or (bool(target_tokens) and
            len(target_tokens & content_tokens(value_part)) >= max(1, len(target_tokens) // 2)))
    return dict(p3_goal_correct=goal, p3_value_strict=strict, p3_value_loose=loose,
        p3_joint_strict=goal * strict, p3_joint_loose=goal * loose,
        p3_segment_parse=int(parts is not None), pred_value=pred,
        goal_segment=parts[0] if parts else None,
        value_segment=parts[2] if parts else None)


def load_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def identity(row):
    return row['dataset'], row.get('meta', {}).get('sample_seed'), row['dialogue_id'], row['method']


def score_saved_row(row):
    if row.get('probe_type') != 'P1_LATE':
        raise ValueError('paper P3 requires serialized probe_type=P1_LATE, not current-value P3')
    result = score_prediction(row['gold_story_label'], row['gold_value'], row.get('pred_value'), row.get('abstain', False))
    return dict(dataset=row['dataset'], dialogue_id=row['dialogue_id'], method=row['method'],
        meta=row.get('meta', {}), paper_probe='P3', probe_type='P1_LATE', scorer=SCORER, **result)


def score_reader_rows(probes, readers):
    """Join by example/method, rejecting duplicates, missing, or unrelated rows."""
    index = {}
    for probe in probes:
        if probe.get('probe_type') != 'P1_LATE':
            raise ValueError('all probe rows must be paper P3 (P1_LATE)')
        key = identity(probe)
        if key in index:
            raise ValueError(f'duplicate probe: {key}')
        index[key] = probe
    seen, scored = set(), []
    for reader in readers:
        if reader.get('probe_type') != 'P1_LATE':
            raise ValueError('all reader rows must be paper P3 (P1_LATE)')
        key = identity(reader)
        if key in seen or key not in index:
            raise ValueError(f'duplicate or unknown reader answer: {key}')
        seen.add(key)
        probe = index[key]
        parsed = parse_reader_text(reader.get('reader_output_text') or '')
        parsed = parsed if isinstance(parsed, dict) else None
        # The saved protocol treats absent JSON / true abstain / empty value as no answer.
        abstain = parsed is None or parsed.get('abstain') is True
        prediction = '' if abstain else str(parsed.get('value') or '')
        row = dict(probe, pred_value=prediction, abstain=abstain or not prediction)
        scored.append(score_saved_row(row))
    if seen != set(index):
        raise ValueError(f'missing reader answers: {len(set(index) - seen)}')
    return scored


def aggregate(scored):
    groups = defaultdict(list)
    for row in scored:
        groups[row['dataset'], row.get('meta', {}).get('sample_seed'), row['method']].append(row)
    cells = []
    for (dataset, seed, method), rows in sorted(groups.items()):
        cells.append(dict(dataset=dataset, seed=seed, method=method, n=len(rows),
            strict=statistics.mean(r['p3_joint_strict'] for r in rows),
            loose=statistics.mean(r['p3_joint_loose'] for r in rows),
            parse_rate=statistics.mean(r['p3_segment_parse'] for r in rows)))
    return dict(paper_probe='P3', serialized_probe_type='P1_LATE', scorer=SCORER, cells=cells)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probes', type=Path)
    parser.add_argument('--reader-out', type=Path)
    parser.add_argument('--saved-scored', type=Path, nargs='+',
        help='Rescore released P1_LATE predictions instead of raw reader outputs')
    parser.add_argument('--scored-out', type=Path, required=True)
    parser.add_argument('--aggregate-out', type=Path, required=True)
    args = parser.parse_args()
    if args.saved_scored:
        if args.probes or args.reader_out:
            parser.error('--saved-scored cannot be combined with --probes/--reader-out')
        scored, seen = [], set()
        for path in args.saved_scored:
            for row in load_jsonl(path):
                key = identity(row)
                if key in seen:
                    parser.error(f'duplicate saved prediction: {key}')
                seen.add(key)
                scored.append(score_saved_row(row))
    else:
        if not args.probes or not args.reader_out:
            parser.error('provide both --probes and --reader-out, or --saved-scored')
        scored = score_reader_rows(load_jsonl(args.probes), load_jsonl(args.reader_out))
    for path in (args.scored_out, args.aggregate_out):
        path.parent.mkdir(parents=True, exist_ok=True)
    args.scored_out.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in scored), encoding='utf-8')
    args.aggregate_out.write_text(json.dumps(aggregate(scored), indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(scorer=SCORER, scored_answers=len(scored))))


if __name__ == '__main__':
    main()

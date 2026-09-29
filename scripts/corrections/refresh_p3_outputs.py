"""Refresh primary P3 fields, whole-field/parse diagnostics, and alias sensitivity."""
from collections import defaultdict
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'scripts/scorer'),str(ROOT/'scripts')]
from scorer_paper_p3 import score_saved_row, SCORER


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    count=0
    for path in sorted((ROOT/'results/diagnostics/update_evidence').glob('p3_*_s*_scored_*.jsonl')):
        rows=read(path)
        groups=defaultdict(list)
        for row in rows:
            primary=score_saved_row(row)
            for field in ['p3_goal_correct','p3_value_strict','p3_value_loose',
                          'p3_joint_strict','p3_joint_loose','p3_segment_parse']:
                row[field]=primary[field]
            row['primary_scorer']=SCORER
            groups[row['method']].append(row)
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        aggregate_path=path.with_name(path.name.replace('_scored_','_aggregate_').replace('.jsonl','.json'))
        aggregate=json.loads(aggregate_path.read_text())
        for method,rs in groups.items():
            assert aggregate[method]['n']==len(rs)
            aggregate[method].update(
                p3_joint_strict=statistics.mean(r['p3_joint_strict'] for r in rs),
                p3_joint_loose=statistics.mean(r['p3_joint_loose'] for r in rs),
                p3_segment_parse_rate=statistics.mean(r['p3_segment_parse'] for r in rs),
                primary_scorer=SCORER)
        aggregate_path.write_text(json.dumps(aggregate,indent=2)+'\n')
        count+=len(rows)
    assert count==19600,count
    subprocess.run([sys.executable,str(ROOT/'scripts/semantic_evaluation/audit_p3_segments.py'),
                    '--artifact',str(ROOT),'--output',str(ROOT/'results/p3_segment_sensitivity')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/audit/p3_alias_exclusion_sensitivity.py'),
                    '--artifact',str(ROOT),'--output',str(ROOT/'results/p3_alias_exclusion/summary.json')],check=True)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_aggregates_flat.py')],check=True)
    print('OK: 19,600 primary P3 fields and all joint aggregates/diagnostics/alias subsets refreshed; flat joint scores use segment-local criteria')


if __name__=='__main__':main()

"""Verify reconstruction against every released main probe cell."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/builders'))
from build_probes import build
from build_deletions import build as deletions

def projection(rows):
    fields=['dialogue_id','method','gold','prompt_system','prompt_user']
    result={tuple(r[k] for k in fields) for r in rows}
    assert len(result)==len(rows)
    return result

def main():
    cells=0
    for dataset in ['sgd','multiwoz']:
        for seed in [42,43,44]:
            for ratio in [.1,.3,.5,.7]:
                probes=[('p1','p1_n200'),('p2','p3_n200')]
                if ratio==.3:probes.append(('p3','n200_late_intent'))
                for probe,tag in probes:
                    path=ROOT/f'data/probes/probes_{dataset}_s{seed}_r{round(ratio*100)}_{tag}.jsonl'
                    if not path.exists():continue
                    expected=[json.loads(l) for l in path.open()]
                    actual=build(dataset,probe,ratio,seed,200)
                    assert projection(actual)==projection(expected),path
                    cells+=1
    assert cells==36,cells
    path=ROOT/'results/diagnostics/update_evidence/p2_reader_input.jsonl'
    expected=[json.loads(l) for l in path.open()]
    actual=deletions('sgd')+deletions('multiwoz')
    assert projection(actual)==projection(expected)
    print(f'OK: {cells} main cells and {len(actual)} paired-deletion prompts reconstructed')

if __name__=='__main__':main()

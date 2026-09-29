"""Regenerate all positional FMTS contexts and verify saved diagnostics."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
METHODS=['full_context','recency','random_seed42','first_n','uniform_stride']

def main():
    count=0
    with tempfile.TemporaryDirectory(prefix='tpbench-fmts-test.') as tmp:
        for split in ['indirect_seed42','multiwoz_scope','sgd_scope']:
            for ratio in [.1,.3]:
                tag=f'{split}_refpool_r{round(ratio*100):03d}'
                subprocess.run([sys.executable,str(ROOT/'scripts/fmts_refpool/run_compression.py'),
                    '--fmts',str(ROOT/f'data/fmts/fmts_{split}.jsonl'),'--ratio',str(ratio),
                    '--seed','42','--tag',tag,'--out_dir',tmp,'--only',*METHODS],
                    check=True,stdout=subprocess.DEVNULL)
                actual=json.loads((Path(tmp)/f'{tag}.json').read_text())
                expected=json.loads((ROOT/f'results/fmts_refpool/{tag}.json').read_text())
                for method in METHODS:
                    ar=actual['raw'][method];er=expected['raw'][method]
                    assert len(ar)==len(er)
                    for a,b in zip(ar,er):
                        for field in ['dialogue_id','compressed_text','kept_turn_ids','b_qa_exact','boundary_recall']:
                            assert a[field]==b[field],(tag,method,field)
                        count+=1
    subprocess.run([sys.executable,str(ROOT/'scripts/fmts_refpool/run_answers.py'),'--dry-run'],check=True)
    print(f'OK: {count} FMTS positional contexts/diagnostics reproduced across six settings')

if __name__=='__main__':main()

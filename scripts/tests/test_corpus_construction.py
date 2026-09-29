"""Reconstruct both all-eligible corpus comparisons from bundled source inputs."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ('dialogue_id', 'method', 'probe_type', 'gold', 'prompt_system', 'prompt_user')

def projection(path):
    return [tuple(json.loads(line)[k] for k in FIELDS) for line in path.open()]

def main():
    with tempfile.TemporaryDirectory(prefix='tpbench-corpora-') as tmp:
        tmp = Path(tmp)
        subprocess.run([sys.executable, str(ROOT/'scripts/seed_robustness/build_seed_robustness.py'),
            '--risawoz-src', str(ROOT/'data/sources/risawoz_test.json'),
            '--longmemeval-src', str(ROOT/'data/sources/longmemeval_knowledge_updates.json'),
            '--risawoz-out', str(tmp/'risawoz.jsonl'),
            '--longmemeval-out', str(tmp/'lme_ku.jsonl')], check=True, stdout=subprocess.DEVNULL)
        for corpus, count in [('risawoz', 207), ('lme_ku', 72)]:
            assert projection(tmp/f'{corpus}.jsonl') == projection(
                ROOT/f'data/probes/probes_{corpus}_all{count}_r30_s42-46.jsonl'), corpus
    print('OK: 2,511 RiSAWOZ/LongMemEval probe rows reconstructed from bundled sources')

if __name__ == '__main__': main()

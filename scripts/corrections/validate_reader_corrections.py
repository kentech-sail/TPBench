"""Recompute corrected reader evidence from public saved answers, without GPU/API."""
import argparse
import json
import re
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[2]


def read(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reader',choices=['llama','mistral'],nargs='+',default=['llama','mistral'])
    a=p.parse_args()
    for reader in a.reader:
        data=ROOT/f'data/llmlingua_correction/{reader}'
        results=ROOT/f'results/llmlingua_correction/{reader}'
        inputs,answers,key=read(data/'inputs.jsonl'),read(results/'answers.jsonl'),read(data/'key.jsonl')
        assert len(inputs)==len(answers)==(2780 if reader=='llama' else 1938)
        assert {r['id'] for r in inputs}=={r['id'] for r in answers}
        assert len(key)==(4164 if reader=='llama' else 2964)
        assert all(not r.get('error') and 0<r['input_tokens']<=7168 for r in answers)
        assert all(r['meta']['k_turns']==len(r['corrected_kept_turn_indices']) for r in key)
        assert all(len(r['compressed_text_used'])==r['compressed_text_chars_used']==r['compressed_text_chars'] for r in key)
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)
            subprocess.run([sys.executable,str(ROOT/'scripts/corrections/score_main_reader.py'),
                '--artifact',str(ROOT),'--prepared',str(data),'--answers',str(results/'answers.jsonl'),
                '--canonical-scorer-dir',str(ROOT/'scripts/scorer'),'--reader-name',reader,
                '--output',str(out)],check=True,stdout=subprocess.DEVNULL)
            for name in ['expanded_scored.jsonl','summary.json']:
                assert (out/name).read_bytes()==(results/name).read_bytes(),(reader,name)
            if reader=='llama':
                for name,expected in [('missing_inputs.jsonl','semantic_pending_inputs.jsonl'),
                                     ('reused_judgments.jsonl','semantic_cached_exact_judgments.jsonl'),
                                     ('p1_corrected_key.jsonl','semantic_corrected_p1_key.jsonl')]:
                    assert (out/'semantic'/name).read_bytes()==(ROOT/'data/llmlingua_correction'/expected).read_bytes()
        print(f'OK: {reader} {len(answers):,} fresh answers / {len(key):,} expanded corrected scores regenerate; exact reuse and pending semantic coverage verified')
    flat=[r for r in read(ROOT/'results/aggregates_flat.jsonl') if r['probe_type']=='P1_LATE']
    assert len(flat)==98
    aggregates={}
    for fp in (ROOT/'results/diagnostics/update_evidence').glob('p3_*_s*_aggregate_*.json'):
        match=re.fullmatch(r'p3_(sgd|multiwoz)_s(42|43|44)_aggregate_(llama|mistral|chunkkv)',fp.stem)
        if not match:continue
        ds,seed,reader=match.groups()
        for method,row in json.loads(fp.read_text()).items():
            aggregates[ds,int(seed),reader,method]=row
    assert len(aggregates)==98
    zeros_with_legacy_credit=0
    for row in flat:
        reader='chunkkv' if 'chunkkv' in row['reader'] else 'mistral' if 'mistral' in row['reader'] else 'llama'
        source=aggregates[row['dataset'],row['seed'],reader,row['method']]
        assert row['strict_acc']==source['p3_joint_strict'],row['cell_id']
        assert row['loose_acc']==source['p3_joint_loose'],row['cell_id']
        if source['p3_joint_strict']==0 and source['p1late_combined_rate_strict']>0:
            zeros_with_legacy_credit+=1
    # Focused zero-rate regression also covers releases without a real zero cell.
    emit=runpy.run_path(str(ROOT/'scripts/build_aggregates_flat.py'))['emit']
    trial=[]
    emit(trial,cell_id='zero_regression',dataset='sgd',ratio=.3,seed=42,
         probe='P1_LATE',reader='test',method='test',m_data=dict(n=1,
         p3_joint_strict=0.,p3_joint_loose=0.,p1late_combined_rate_strict=.5,p1late_combined_rate=.6))
    assert trial[0]['strict_acc']==trial[0]['loose_acc']==0.
    print(f'OK: all 98 flattened joint P3 cells equal primary segment aggregates; zero-rate fallback regression passes ({zeros_with_legacy_credit} real zero/legacy-credit cells)')


if __name__=='__main__':main()

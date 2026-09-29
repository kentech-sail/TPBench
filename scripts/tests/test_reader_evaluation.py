"""Reconstruct first-user-plus-recent contexts and validate current result tables."""
import json,sys,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/builders'))
from construction import split_turns,render_subset
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def main():
 key=read(ROOT/'data/reader_evaluation/key.jsonl')
 inputs=read(ROOT/'data/reader_evaluation/inputs.jsonl');answers=read(ROOT/'results/reader_evaluation/answers.jsonl')
 assert {r['id'] for r in inputs}=={r['id'] for r in answers}=={r['check_id'] for r in key}
 full={(r['dataset'],r['sample_seed'],r['paper_probe'],r['dialogue_id']):r for r in key if r['method']=='full_context'}
 requests={r['id']:r for r in inputs}
 for r in key:
  assert r['prompt_user']==requests[r['check_id']]['prompt_user']
  if r['paper_probe']=='P1':assert "What was the user's initial goal at the beginning of this conversation?" in r['prompt_user']
  if r['method']!='first_user_recent':continue
  source=full[r['dataset'],r['sample_seed'],r['paper_probe'],r['dialogue_id']]
  header,turns=split_turns(source['compressed_text_used']);k=max(1,round(len(turns)*.3))
  first=next(t['turn_id'] for t in turns if t['speaker']=='USER')
  keep=sorted([first]+[t['turn_id'] for t in reversed(turns) if t['turn_id']!=first][:k-1])
  assert render_subset(header,turns,keep)==r['compressed_text_used']
 with tempfile.TemporaryDirectory() as temp:
  out=Path(temp)
  subprocess.run([sys.executable,str(ROOT/'scripts/reader_evaluation/score_answers.py'),'--artifact',str(ROOT),'--key',str(ROOT/'data/reader_evaluation/key.jsonl'),'--answers',str(ROOT/'results/reader_evaluation/answers.jsonl'),'--output',str(out)],check=True)
  for n in ['summary.json','scored.jsonl']:assert (out/n).read_bytes()==(ROOT/'results/reader_evaluation/scored'/n).read_bytes(),n
  subprocess.run([sys.executable,str(ROOT/'scripts/semantic_evaluation/aggregate_equivalence.py'),'--output',str(out/'semantic.json')],check=True)
  assert (out/'semantic.json').read_bytes()==(ROOT/'results/semantic_evaluation/p1_summary.json').read_bytes()
 summary=json.loads((ROOT/'results/main/seed_summary.json').read_text())['p1']
 scored=read(ROOT/'results/reader_evaluation/scored/scored.jsonl')
 for ds,methods in summary.items():
  for method,metrics in methods.items():
   rows=[r for r in scored if r['probe']=='P1' and r['dataset']==ds.removesuffix('_r30') and r['method']==method]
   assert len(rows)==600
   for label,field in [('p1_em_strict','strict'),('p1_em_loose','loose'),('p1_token_f1','token_f1')]:assert abs(sum(r[field] for r in rows)/600-metrics[label]['mean'])<1e-12
 print('OK: P1 headline, saved inference, semantic tables, and first-plus-recent contexts reproduced')
if __name__=='__main__':main()

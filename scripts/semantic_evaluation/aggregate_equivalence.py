"""Recompute P1 full-equivalence scores from saved, blinded judgments."""
import argparse,json,sys,statistics
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts/reader_evaluation'))
from score_answers import cluster_interval
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 data=ROOT/'data/semantic_evaluation';res=ROOT/'results/semantic_evaluation/p1'
 inputs=read(data/'p1_inputs.jsonl');key=read(data/'p1_key.jsonl');judgments=read(res/'judgments.jsonl')
 labels={r['id']:r for r in judgments};raw={r['id']:r for r in read(res/'raw_responses.jsonl')}
 scope=json.loads((data/'p1_scope.json').read_text())
 assert len(labels)==len(judgments)==len(inputs)==scope['expected_pairs']
 assert set(labels)=={r['id'] for r in inputs}=={r['semantic_id'] for r in key}==set(raw)
 assert len(key)==scope['expected_answers']
 assert not any(r['method'] in scope['excluded_methods'] for r in key)
 for r in inputs:
  record=raw[r['id']]
  assert record['judgment']==labels[r['id']]
  payload=json.loads(record['request']['input'][1]['content'])
  assert payload==r,(r['id'],'request mismatch')
  generated=''.join(c.get('text','') for out in record['response']['output'] for c in out.get('content',[]) if c.get('type')=='output_text')
  answer=json.loads(generated)
  assert {k:v for k,v in answer.items() if k!='id'}=={k:v for k,v in labels[r['id']].items() if k!='id'},(r['id'],'response/label mismatch')
  if answer.get('id')!=r['id']:assert record.get('identifier_mapping'),r['id']
 groups=defaultdict(list);cells=defaultdict(list)
 for r in key:
  j=labels[r['semantic_id']];assert j['equivalent'] is None or type(j['equivalent']) is bool
  row=dict(r,equivalent=int(j['equivalent'] is True),uncertain=int(j['equivalent'] is None))
  groups[r['dataset'],r['method']].append(row);cells[r['dataset'],r['seed'],r['method']].append(row)
 result={'pairs':len(labels),'answers':len(key),'criterion':'AI-assisted full equivalence','evaluation_scope':scope,
  'methods':[dict(dataset=k[0],method=k[1],equivalent=cluster_interval(rs,'equivalent'),uncertain=cluster_interval(rs,'uncertain')) for k,rs in sorted(groups.items())],
  'cells':[dict(dataset=k[0],seed=k[1],method=k[2],n=len(rs),equivalent=statistics.mean(r['equivalent'] for r in rs)) for k,rs in sorted(cells.items())]}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
 print(f"OK: {len(labels):,} request/response/label pairs; {len(key):,} semantic answers aggregated; corrected LLMLingua excluded")
if __name__=='__main__':main()

"""CPU-only audit of corrected FMTS retention and excluded historical reader condition."""
import json,re,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
H=ROOT/'results/historical/fmts_legacy_20260929'
METHOD='llmlingua2_cache'
read=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s]
tokens=lambda t:set(re.findall('[a-z0-9]+',t.lower()))|set(re.findall('[가-힣]+',t))
def metrics(rec,ctx,kept):
 gold=rec['probe']['gold_answer'].lower();span=bool(gold) and gold in ctx.lower();kw=tokens(gold)
 support=rec['transition'].get('support_turn_ids',[]);boundary=rec.get('gt_boundaries') or rec['transition'].get('boundary_turn_ids') or [];adv=rec.get('adv_positions',[])
 recall=lambda ids:sum(i in kept for i in ids)/len(ids) if ids else 1.
 full=' '.join(t['text'] for t in rec['turns']);fallback=not span and len(kw&tokens(ctx))/max(len(kw),1)>.5
 return dict(qa_exact=float(span or fallback),answer_span_exact=float(span),fallback_overlap_only=float(fallback),boundary_recall=recall(boundary),support_recall=recall(support),adv_fp=sum(i in kept for i in adv)/len(adv) if adv else 0.,compression_ratio=len(kept)/len(rec['turns']),token_ratio=len(ctx.split())/len(full.split()))
def main():
 report=json.loads((ROOT/'results/fmts_refpool/corrected_diagnostics.json').read_text());groups={(r['dataset'],r['ratio'],r['method']):r for r in report['groups']}
 count=changed=0;contexts={}
 for ds in ['indirect_seed42','sgd_scope','multiwoz_scope']:
  source={r['dialogue_id']:r for r in read(ROOT/f'data/fmts/fmts_{ds}.jsonl')}
  for ratio in [.1,.3]:
   tag=f'{ds}_refpool_r{round(ratio*100):03d}';path=Path(f'results/fmts_refpool/{tag}.json');cur=json.loads((ROOT/path).read_text());old=json.loads((H/path).read_text())
   for method,rows in cur['raw'].items():
    if method!=METHOD:assert rows==old['raw'][method]
    vals=[]
    for row,original in zip(rows,old['raw'][method]):
     rec=source[row['dialogue_id']];ctx=row['compressed_text'];kept=row['kept_turn_ids'];assert ctx==' '.join(rec['turns'][i]['text'] for i in kept)
     d=metrics(rec,ctx,kept);vals.append(d)
     for metric,field in [('qa_exact','qa_accuracy_exact'),('qa_exact','b_qa_exact')]+[(k,k) for k in ['boundary_recall','support_recall','adv_fp','token_ratio']]:assert abs(d[metric]-row[field])<1e-12,(tag,method,metric)
     contexts[tag,row['dialogue_id']+'::'+method]=ctx
     if method==METHOD:changed+=ctx!=original['compressed_text']
     count+=1
    for k in vals[0]:assert abs(statistics.mean(v[k] for v in vals)-groups[ds,ratio,method]['corrected'][k])<1e-12,(tag,method,k)
 assert count==8000 and changed==792,(count,changed)
 tasks=read(ROOT/'results/fmts_execution/tasks.jsonl');answers=read(ROOT/'results/fmts_execution/answers.jsonl')
 legacy_tasks=read(H/'results/fmts_execution/tasks.jsonl');legacy_answers=read(H/'results/fmts_execution/answers.jsonl')
 assert tasks==[x for x in legacy_tasks if not x['item_id'].endswith('::'+METHOD)]
 assert answers==[x for x in legacy_answers if not x['item_id'].endswith('::'+METHOD)]
 assert len(tasks)==len(answers)==7000 and len(legacy_tasks)==len(legacy_answers)==8000
 for t in tasks:assert t['context']==contexts[t['batch_tag'],t['item_id']]
 print('OK: 8,000 deterministic records; 792 corrected contexts; 7,000 unchanged reader task/answer records; historical LLMLingua answers excluded')
if __name__=='__main__':main()

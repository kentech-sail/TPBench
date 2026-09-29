"""Score reader answers with the released metrics and paired cluster intervals."""
import argparse,hashlib,json,re,statistics,sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from audit_p3_segments import split_answer

def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def parsed_output(text):
    for part in reversed(re.findall(r'\{[^{}]*\}',text,re.S)):
        try:
            obj=json.loads(part)
            if isinstance(obj,dict):return obj
        except ValueError:pass
    return None

def cluster_interval(rows,metric):
    # Equal weight for each of the three seed estimates, with source-level resampling.
    ns=Counter(r['seed'] for r in rows);clusters=defaultdict(lambda:[0.,0.])
    for r in rows:
        weight=1/ns[r['seed']]
        clusters[r['source']][0]+=weight*r[metric];clusters[r['source']][1]+=weight
    arr=np.array(list(clusters.values()));rng=np.random.default_rng(20260928)
    idx=rng.integers(len(arr),size=(5000,len(arr)));sample=arr[idx].sum(axis=1)
    boot=sample[:,0]/sample[:,1]
    return {'mean':float(arr[:,0].sum()/arr[:,1].sum()),'ci95':np.quantile(boot,[.025,.975]).tolist(),'clusters':len(arr),'n':len(rows)}

def main():
    ap=argparse.ArgumentParser()
    for k in ['artifact','key','answers','output']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--probe',choices=['P1','P2','P3'])
    ap.add_argument('--allow-partial',action='store_true');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    sys.path[:0]=[str(a.artifact/'scripts/scorer'),str(a.artifact/'scripts')]
    from scorer_p1 import score_p1
    from scorer_p3 import score_p3
    from scorer_p1_late import score_late,content_tokens,norm_text,normalized_phrase_occurs
    key=read(a.key);raw=read(a.answers);answers={r['id']:r for r in raw};assert len(answers)==len(raw)
    assert set(answers)<={r['check_id'] for r in key}
    if a.probe:
        key=[r for r in key if r['paper_probe']==a.probe]
        keep={r['check_id'] for r in key};answers={k:v for k,v in answers.items() if k in keep}
    expected={r['check_id'] for r in key};assert set(answers)<=expected
    missing=expected-set(answers)
    if missing and not a.allow_partial:raise ValueError(f'Missing {len(missing)} reader requests; no final report produced')
    scored=[];groups=defaultdict(list)
    for r in key:
        if r['check_id'] not in answers:continue
        o=answers[r['check_id']];parsed=parsed_output(o['reader_output_text']);probe=r['paper_probe']
        row={'id':r['check_id'],'dataset':r['dataset'],'seed':r['sample_seed'],'dialogue_id':r['dialogue_id'],
             'source':r['dialogue_id'].split('__qa_')[0],'method':r['method'],'probe':probe,'condition':r['question_condition'],
             'format_error':int(parsed is None),'input_tokens':o['input_tokens']}
        if probe=='P1':
            s=score_p1(r,parsed);row.update(strict=s['p1_correct'],loose=s['p1_correct_loose'],token_f1=s['p1_token_f1'],pred=s['pred'],gold=s['gold'],abstain=int(s['abstain']))
        elif probe=='P2':
            s=score_p3(json.dumps(parsed) if parsed else '',r['gold'],r['compressed_text_used'],r['dataset'])
            row.update(strict=s['p3_correct'],loose=s['p3_correct_loose'],pred=s['pred_value'] or '',gold=r['gold'],abstain=s['abstain'])
        else:
            s=score_late(r,parsed);pred=s['pred'];parts=split_answer(pred);g=v=l=0
            if parts:
                g=int(bool(content_tokens(r['gold_story_label'])&content_tokens(parts[0])))
                v=int(normalized_phrase_occurs(norm_text(r['gold_value']),norm_text(parts[2])))
                vt=content_tokens(r['gold_value']);l=int(v or (bool(vt) and len(vt&content_tokens(parts[2]))>=max(1,len(vt)//2)))
            row.update(strict=s['p1late_combined_strict'],loose=s['p1late_combined'],segment_strict=g*v,segment_loose=g*l,
                segment_parse=int(parts is not None),pred=pred,gold=r['gold'],abstain=int(s['abstain']))
        scored.append(row);groups[row['dataset'],row['seed'],probe,row['condition'],row['method']].append(row)
    (a.output/'scored.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in scored))
    cells=[]
    for (ds,seed,probe,condition,method),rs in sorted(groups.items()):
        ms=['strict','loose','format_error','input_tokens']+(['token_f1'] if probe=='P1' else ['segment_strict','segment_loose','segment_parse'] if probe=='P3' else [])
        cells.append(dict(dataset=ds,seed=seed,probe=probe,condition=condition,method=method,n=len(rs),**{m:statistics.mean(r[m] for r in rs) for m in ms}))
    report={'complete':not missing,'scope_probes':[a.probe] if a.probe else ['P1','P2','P3'],'expected_requests':len(expected),'completed_requests':len(answers),'expanded_scored':len(scored),'cells':cells}
    if not missing:
        methods=defaultdict(list)
        for r in scored:methods[r['dataset'],r['probe'],r['condition'],r['method']].append(r)
        report['methods']=[]
        for (ds,probe,condition,method),rs in sorted(methods.items()):
            ms=['strict','loose']+(['token_f1'] if probe=='P1' else ['segment_strict','segment_loose'] if probe=='P3' else [])
            report['methods'].append(dict(dataset=ds,probe=probe,condition=condition,method=method,**{m:cluster_interval(rs,m) for m in ms}))
        index={(r['dataset'],r['seed'],r['dialogue_id'],r['probe'],r['condition'],r['method']):r for r in scored}
        pairs=defaultdict(list)
        for r in scored:
            base_key=(r['dataset'],r['seed'],r['dialogue_id'],r['probe'])
            if r['method']=='first_user_recent':
                for control in ['full_context','first_n','recency','uniform_stride']:
                    other=index[(*base_key,r['condition'],control)]
                    ms=['strict','loose']+(['segment_strict','segment_loose'] if r['probe']=='P3' else [])
                    pairs[r['dataset'],r['probe'],r['condition'],f'first_user_recent_minus_{control}'].append(dict(seed=r['seed'],source=r['source'],**{m:r[m]-other[m] for m in ms}))
        report['paired_comparisons']=[dict(dataset=k[0],probe=k[1],condition=k[2],comparison=k[3],**{m:cluster_interval(rs,m) for m in rs[0] if m not in {'source','seed'}}) for k,rs in sorted(pairs.items())]
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'cells','methods','paired_comparisons'}}))

if __name__=='__main__':main()

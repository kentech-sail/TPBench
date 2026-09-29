"""Resumable, bounded API evaluation; never reads human labels or method keys.

Only --inputs and a fixed prompt are transmitted. Credentials are read in place
from OPENAI_API_KEY or --env-file, never included in outputs. Stdlib only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

MODEL = 'gpt-5.4-mini-2026-03-17'
SCHEMA = dict(type='object', properties={
    'id': {'type':'string'}, 'equivalent': {'type':['boolean','null']},
    'core_goal': {'type':'string','enum':['correct','partial','wrong','uncertain']},
    'reference_type': {'type':'string','enum':['explicit_goal','implicit_need','not_goal','uncertain']},
    'reason': {'type':'string'}},
    required=['id','equivalent','core_goal','reference_type','reason'], additionalProperties=False)

def sha(b): return hashlib.sha256(b).hexdigest()
def stamp(): return datetime.now(timezone.utc).isoformat()
def rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []

def get_key(path):
    key=os.environ.get('OPENAI_API_KEY','')
    if not key and path:
        for line in path.read_text().splitlines():
            line=line.strip().removeprefix('export ')
            if line.startswith('OPENAI_API_KEY='):
                key=line.split('=',1)[1].strip()
                if key[:1] in {'"',"'"}: key=key[1:].split(key[0],1)[0]
                else: key=key.split(' #',1)[0].strip()
                break
    if not key or not key.startswith('sk-'): raise SystemExit('Usable OPENAI_API_KEY not found; no request sent.')
    return key

def valid(j, ident):
    assert set(j)==set(SCHEMA['required']) and j['id']==ident
    assert j['equivalent'] is None or type(j['equivalent']) is bool
    assert j['core_goal'] in SCHEMA['properties']['core_goal']['enum']
    assert j['reference_type'] in SCHEMA['properties']['reference_type']['enum']
    assert isinstance(j['reason'],str) and j['reason'].strip()

def main():
    ap=argparse.ArgumentParser()
    for k in ['inputs','prompt','output']: ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--env-file',type=Path)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--workers',type=int,default=6)
    ap.add_argument('--budget-usd',type=float,default=1.0)
    ap.add_argument('--dry-run',action='store_true')
    a=ap.parse_args(); data=rows(a.inputs); prompt=a.prompt.read_text()
    assert len({r['id'] for r in data})==len(data)
    for r in data:
        assert set(r) in ({'id','question','reference','candidate'}, {'id','question','context','reference','candidate'}), 'Unblinded or unexpected input field'
        assert all(isinstance(v,str) for v in r.values())
    if a.limit: data=data[:a.limit]
    config=dict(model=MODEL,provider='OpenAI',endpoint='https://api.openai.com/v1/responses',
        prompt_sha256=sha(a.prompt.read_bytes()),input_sha256=sha(a.inputs.read_bytes()),
        schema_sha256=sha(json.dumps(SCHEMA,sort_keys=True).encode()),
        decoding=dict(reasoning_effort='none',temperature=0,max_output_tokens=512),store=False,
        rates_usd_per_million=dict(input=0.75,output=4.50),runner_sha256=sha(Path(__file__).read_bytes()))
    if a.dry_run:
        print(json.dumps(dict(inputs=len(data),config=config)));return
    key=get_key(a.env_file); a.output.mkdir(parents=True,exist_ok=True)
    meta=a.output/'model_record.json'
    if meta.exists():
        old=json.loads(meta.read_text())
        assert all(old.get(k)==v for k,v in config.items()), 'Run configuration changed; use a new output directory'
    else:
        meta.write_text(json.dumps(dict(config,execution_timestamp=stamp()),indent=2)+'\n')
    rawpath=a.output/'raw_responses.jsonl'; resultpath=a.output/'judgments.jsonl'
    previous=rows(rawpath); saved=rows(resultpath)
    assert len({r['id'] for r in saved})==len(saved), 'Duplicate saved output'
    done={r['id'] for r in saved}
    # Recover a valid response if a process stopped between raw and parsed writes.
    with resultpath.open('a') as f:
        for r in previous:
            if r.get('judgment') and r['id'] not in done:
                valid(r['judgment'],r['id']); f.write(json.dumps(r['judgment'])+'\n'); done.add(r['id'])
    spent=sum(r.get('estimated_cost_usd',0) for r in previous)
    lock=threading.Lock(); stop=threading.Event(); reserved=0.; completed=0
    pending=[r for r in data if r['id'] not in done]
    print(json.dumps(dict(event='start',pending=len(pending),saved=len(done),budget_usd=a.budget_usd)),flush=True)
    def run(row):
        nonlocal spent,reserved,completed
        request=dict(model=MODEL,input=[{'role':'system','content':prompt},
            {'role':'user','content':json.dumps(row,ensure_ascii=False)}],
            text={'format':dict(type='json_schema',name='semantic_judgment',strict=True,schema=SCHEMA)},
            reasoning={'effort':'none'},temperature=0,max_output_tokens=512,store=False)
        body=json.dumps(request,ensure_ascii=False).encode()
        # Byte count plus margin upper-bounds ordinary UTF-8 text tokenization;
        # output is hard-capped. Outstanding requests reserve this amount.
        reserve=(len(body)+2048)*.75/1e6+512*4.50/1e6
        for attempt in range(3):
            if stop.is_set(): return
            with lock:
                if spent+reserved+reserve>a.budget_usd:
                    stop.set();print(json.dumps(dict(event='budget_stop',estimated_usd=spent)),flush=True);return
                reserved+=reserve
            rec=dict(id=row['id'],timestamp=stamp(),attempt=attempt,request=request)
            fatal=False
            try:
                req=urllib.request.Request(config['endpoint'],data=body,headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=90) as response: raw=json.load(response)
                rec['response']=raw
                usage=raw.get('usage') or {}
                rec['estimated_cost_usd']=(usage.get('input_tokens',0)*.75+usage.get('output_tokens',0)*4.50)/1e6
                assert raw.get('status')=='completed', 'Incomplete response'
                assert raw.get('model')==MODEL, 'Unexpected model snapshot'
                output=''.join(c.get('text','') for m in raw.get('output',[]) if m.get('type')=='message' for c in m.get('content',[]) if c.get('type')=='output_text')
                j=json.loads(output);valid(j,row['id']);rec['judgment']=j
            except urllib.error.HTTPError as e:
                rec['error_type']='HTTPError';rec['http_status']=e.code
                rec['estimated_cost_usd']=0;fatal=e.code not in {429,500,502,503,504}
            except Exception as e:
                rec['error_type']=type(e).__name__
                rec.setdefault('estimated_cost_usd',reserve)
            with lock:
                reserved-=reserve;spent+=rec['estimated_cost_usd']
                with rawpath.open('a') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
                if rec.get('judgment'):
                    with resultpath.open('a') as f:f.write(json.dumps(rec['judgment'],ensure_ascii=False)+'\n');f.flush()
                    completed+=1
                    if completed%25==0:print(json.dumps(dict(event='progress',completed=completed,estimated_usd=round(spent,4))),flush=True)
            if rec.get('judgment'):return
            if fatal:
                stop.set();print(json.dumps(dict(event='fatal_api_error',http_status=rec.get('http_status'))),flush=True);return
            time.sleep(2**attempt)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run,r) for r in pending]):future.result()
    final=rows(resultpath);remaining=[r['id'] for r in data if r['id'] not in {j['id'] for j in final}]
    report=dict(expected=len(data),completed=len(data)-len(remaining),remaining=remaining,
        estimated_cost_usd=spent,finished_timestamp=stamp())
    (a.output/'execution_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='remaining'}),flush=True)
    if remaining:raise SystemExit(2)

if __name__=='__main__':main()

"""Corrected Mistral run; current snapshot pinned because original revision is unrecorded.
Derived from the released resumable reader; serial inference preserves the original
single-example execution pattern. No paid API calls.
"""
import argparse,fcntl,hashlib,json,os,platform,sys,time
from pathlib import Path
import torch,transformers
from transformers import AutoTokenizer,AutoModelForCausalLM

MODEL='mistralai/Mistral-7B-Instruct-v0.3'
REV='c170c708c41dac9275d15a8fff4eca08d52bab71'
def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    for k in ['inputs','output','model-path']:ap.add_argument('--'+k,type=Path,required=True)
    ap.add_argument('--attn',choices=['eager','sdpa'],required=True)
    ap.add_argument('--batch-size',type=int,default=1);ap.add_argument('--batch-tokens',type=int,default=6000)
    ap.add_argument('--limit',type=int);ap.add_argument('--serial-check',type=int,default=0)
    ap.add_argument('--coverage-check',action='store_true')
    a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    lock=(a.output.parent/'gpu-reader.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert torch.cuda.device_count()==1,'Expose only the assigned GPU via CUDA_VISIBLE_DEVICES'
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(a.model_path,local_files_only=True,use_fast=True)
    tok.pad_token=tok.eos_token;tok.padding_side='left';tok.truncation_side='left'
    data=[r for r in read(a.inputs) if r['attn_impl']==a.attn]
    records=read(a.output/'answers.jsonl');done={r['id'] for r in records};assert len(done)==len(records)
    pending=[r for r in data if r['id'] not in done]
    for r in pending:
        prompt=tok.apply_chat_template([{'role':'system','content':r['prompt_system']},{'role':'user','content':r['prompt_user']}],tokenize=False,add_generation_prompt=True)
        r['_ids']=tok(prompt,truncation=True,max_length=r['max_input_tokens'])['input_ids']
    pending.sort(key=lambda r:(len(r['_ids']),r['id']))
    if a.limit:
        pending=([pending[round(i*(len(pending)-1)/(a.limit-1))] for i in range(a.limit)]
                 if a.coverage_check and a.limit>1 else pending[:a.limit])
    config=dict(model=MODEL,revision=REV,input_sha256=sha(a.inputs),attention=a.attn,dtype='bfloat16',greedy=True,
        batch_size=a.batch_size,batch_tokens=a.batch_tokens,torch=torch.__version__,transformers=transformers.__version__,
        python=platform.python_version(),gpu=torch.cuda.get_device_name(0),runner_sha256=sha(Path(__file__)),
        model_files={p.name:sha(p) for p in a.model_path.iterdir() if p.name.endswith(('.json','.safetensors'))})
    mp=a.output/'model_record.json'
    if mp.exists():assert json.loads(mp.read_text())==config,'Resume configuration mismatch'
    else:mp.write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'pending':len(pending),'completed':len(done),'min_tokens':min((len(r['_ids']) for r in pending),default=0),'max_tokens':max((len(r['_ids']) for r in pending),default=0),'config':{k:v for k,v in config.items() if k!='model_files'}}),flush=True)
    if not pending:return
    model=AutoModelForCausalLM.from_pretrained(a.model_path,local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation=a.attn).to('cuda:0').eval()
    def generate(batch):
        enc=tok.pad({'input_ids':[r['_ids'] for r in batch]},padding=True,return_tensors='pt')
        enc={k:v.to('cuda:0') for k,v in enc.items()}
        assert len({r['max_new_tokens'] for r in batch})==1
        with torch.inference_mode():
            outputs=model.generate(**enc,max_new_tokens=batch[0]['max_new_tokens'],do_sample=False,temperature=1.,top_p=1.,pad_token_id=tok.pad_token_id)
        return tok.batch_decode(outputs[:,enc['input_ids'].shape[1]:],skip_special_tokens=True)
    start=time.time();finished=0;serial=[];pos=0
    with (a.output/'answers.jsonl').open('a') as f:
        while pos<len(pending):
            end=min(pos+a.batch_size,len(pending))
            while end>pos+1 and (end-pos)*len(pending[end-1]['_ids'])>a.batch_tokens:end-=1
            batch=pending[pos:end]
            try:answers=generate(batch)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache();raise RuntimeError('GPU memory limit reached; no silent configuration change')
            for row,answer in zip(batch,answers):
                if len(serial)<a.serial_check:
                    single=generate([row])[0];serial.append({'id':row['id'],'identical':single==answer,'batch_answer':answer,'single_answer':single})
                out=dict(id=row['id'],reader_output_text=answer,error=None,model=MODEL,input_tokens=len(row['_ids']),batch_size=len(batch))
                f.write(json.dumps(out,ensure_ascii=False)+'\n');done.add(row['id'])
            f.flush();finished+=len(batch);pos=end
            if finished%100<len(batch) or pos==len(pending):
                elapsed=time.time()-start;rate=finished/elapsed
                status={'completed':len(done),'expected':len(data),'processed_now':finished,'rate_per_second':rate,'eta_minutes':(len(pending)-pos)/max(rate,.001)/60,'elapsed_seconds':elapsed}
                (a.output/'progress.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status),flush=True)
    if serial:(a.output/'batch_serial_check.json').write_text(json.dumps(serial,ensure_ascii=False,indent=2)+'\n')
    final={'completed':len(done),'expected':len(data),'remaining':len(data)-len(done),'elapsed_seconds':time.time()-start}
    (a.output/'completion.json').write_text(json.dumps(final,indent=2)+'\n');print(json.dumps(final),flush=True)

if __name__=='__main__':main()

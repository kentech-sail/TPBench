"""Answer the included FMTS tasks using the recorded protocol and Codex CLI.

--dry-run validates inputs without invoking a model. Output is append-only;
successful task identifiers already in the output are skipped on restart.
"""
import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tasks',type=Path,default=ROOT/'results/fmts_execution/tasks.jsonl')
    p.add_argument('--protocol',type=Path,default=ROOT/'results/fmts_execution/protocol.json')
    p.add_argument('--out',type=Path,default=ROOT/'outputs/fmts_answers.jsonl')
    p.add_argument('--limit',type=int,default=0)
    p.add_argument('--codex',default='codex')
    p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args(); protocol=json.loads(a.protocol.read_text())
    tasks=[json.loads(l) for l in a.tasks.open() if l.strip()]
    if a.limit: tasks=tasks[:a.limit]
    for t in tasks:
        assert all(k in t for k in ['task_id','batch_tag','item_id','context','question'])
    if a.dry_run:
        print(f'Validated {len(tasks)} tasks; model={protocol["model"]}; no model calls')
        return
    done=set()
    if a.out.exists():
        done={r['task_id'] for r in map(json.loads,a.out.open()) if not r.get('error')}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for task in tasks:
        if task['task_id'] in done:continue
        with tempfile.TemporaryDirectory(prefix='tpbench-fmts.') as tmp:
            work=Path(tmp); schema=work/'schema.json'; answer=work/'answer.json'
            schema.write_text(json.dumps(protocol['output_schema']))
            user=protocol['user_template'].format(context=task['context'],question=task['question'])
            cmd=[a.codex,'exec','--ignore-user-config','--ephemeral',
                 '--skip-git-repo-check','--sandbox','read-only','--model',protocol['model'],
                 '-c','base_instructions='+json.dumps(protocol['system_prompt']),
                 '--output-schema',str(schema),'-o',str(answer),'-']
            start=time.monotonic(); raw=''; value=''; error=None
            try:
                subprocess.run(cmd,input=user,text=True,cwd=work,capture_output=True,
                               check=True,timeout=a.timeout)
                raw=answer.read_text(); value=json.loads(raw)['answer']
                if not isinstance(value,str) or not value.strip():raise ValueError('Empty answer')
            except Exception as exc:
                error=f'{type(exc).__name__}: {exc}'
            row={k:task[k] for k in ['task_id','batch_tag','item_id']}
            row.update(answer=value,raw=raw,error=error,retry_count=0,
                       latency_sec=time.monotonic()-start,model=protocol['model'])
            with a.out.open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if error:raise SystemExit(f'Task {task["task_id"]} failed; error recorded in {a.out}')
    print(f'Answers written to {a.out}')

if __name__=='__main__':main()

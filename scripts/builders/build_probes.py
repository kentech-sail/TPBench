"""Construct P1/P2/P3 prompts from the included annotations and contexts.

The construction directory contains the ordered candidate context pools.
Sample seeds select candidates; final-update and value filters determine P2.
P3 samples the eligible late-update pool. Saved contexts fix the compressor
outputs independently of reader hardware and package versions.
"""
import argparse
import json
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from normalize import value_in_turns
from construction import (make_p1_user, make_p3_user, is_alias_equivalent,
    normalize_task, p3_user_prompt, split_turns, evidence_turns)

SYSTEM='You are reading a conversation that may be partial or compressed. Answer ONLY using the provided context. Reply with a single JSON object and no other text. Do not add markdown fences.'

def load(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]

def build(dataset,probe,ratio,seed,n):
    labels={r['dialogue_id']:r for r in load(ROOT/f'results/diagnostics/update_evidence/{dataset}_goal_labels.jsonl')}
    pool=load(ROOT/f'data/construction/{dataset}_compressed_r{round(ratio*100):02d}.jsonl')
    if probe=='p3':
        assert ratio==0.3, 'P3 construction uses r=0.30'
        candidates={r['dialogue_id']:r for r in pool}
        eligible=sorted(d for d,l in labels.items() if d in candidates
            and l.get('origin_status')=='resolved' and l.get('is_final_slot_update')
            and l['n_turns']>=10 and l['origin_turn_id']>=6
            and l['origin_turn_id']/max(1,l['n_turns']-1)>=0.5)
        pool=[candidates[d] for d in sorted(random.Random(seed).sample(eligible,n))]
    else:
        random.Random(seed).shuffle(pool)
        pool=pool[:n]
    rows=[]
    for record in pool:
        did=record['dialogue_id']; label=labels[did]
        _,turns=split_turns(record['methods']['full_context'])
        if probe=='p2' and (not label['is_final_slot_update'] or
            is_alias_equivalent(label['old_value'],label['new_value'],dataset) or
            not value_in_turns(label['new_value'],[t['text'] for t in turns],dataset)):
            continue
        for method,context in record['methods'].items():
            meta={'n_turns':record['n_turns'],'k_turns':record['k_turns'],
                  'ratio':ratio,'sample_seed':seed,'budget_unit':'dialogue_turns',
                  'origin_turn_id':label['origin_turn_id'],
                  'is_final_slot_update':label['is_final_slot_update'],
                  'final_slot_update_turn_id':label['final_slot_update_turn_id']}
            row={'dialogue_id':did,'dataset':dataset,'slot':label['slot'],
                 'slot_human':label['slot_human'],'old_value':label['old_value'],
                 'new_value':label['new_value'],'method':method,'meta':meta,
                 'prompt_system':SYSTEM,'compressed_text_used':context}
            if probe=='p1':
                row.update(probe_type='P1',gold=label['goal_first_sentence'],prompt_user=make_p1_user(context))
            elif probe=='p2':
                evidence=evidence_turns(turns,label['new_value'],dataset)
                meta.update(s_origin=[label['origin_turn_id']],s_evidence=evidence,
                            s_transition_support=sorted(set(evidence)|{label['origin_turn_id']}))
                row.update(probe_type='P3',gold=label['new_value'],prompt_user=make_p3_user(context,label['slot_human']))
            else:
                goal=normalize_task(label.get('goal_first_sentence') or label.get('goal_text') or '')
                row.update(probe_type='P1_LATE',gold=f"{goal}; {label['slot_human']}: {label['new_value']}",
                    gold_story_label=goal,gold_value=label['new_value'],story_label_phrase=goal,
                    prompt_user=p3_user_prompt(context,label['slot_human']))
            rows.append(row)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['sgd','multiwoz'],required=True)
    p.add_argument('--probe',choices=['p1','p2','p3'],required=True)
    p.add_argument('--ratio',type=float,default=0.30)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--n',type=int,default=200)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); rows=build(a.dataset,a.probe,a.ratio,a.seed,a.n)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(f'Wrote {len(rows)} rows to {a.out}')

if __name__=='__main__': main()

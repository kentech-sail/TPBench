"""Reconstruct the paired P2 deletion contexts from released evidence fields."""
import argparse
import json
from pathlib import Path
from construction import prompt_parts,split_turns,render_subset,evidence_turns,matched_control

ROOT=Path(__file__).resolve().parents[2]

def build(dataset):
    source=ROOT/f'data/probes/probes_{dataset}_s42_r30_p3_n200.jsonl'
    rows=[]
    for row in map(json.loads,source.open()):
        if row['method']!='full_context':continue
        meta=row['meta']; context,question=prompt_parts(row['prompt_user'])
        header,turns=split_turns(context)
        evidence=evidence_turns(turns,row['gold'],dataset)
        assert evidence==meta['s_evidence']
        origin=meta['origin_turn_id']; support=sorted(set(evidence)|{origin})
        assert support==meta['s_transition_support']
        forbidden=set(support)
        one=matched_control(turns,[origin],forbidden,row['old_value'],row['new_value'],dataset)
        all_control=matched_control(turns,support,forbidden,row['old_value'],row['new_value'],dataset)
        assert one==meta['control_origin'] and all_control==meta['control_transition_support']
        conditions=[('full_context',[]),('cf_origin_minus_origin',[origin]),
                    ('cf_origin_minus_control_origin',one)]
        if meta['transition_support_control_status']=='matched':
            conditions += [('cf_origin_minus_transition_support',support),
                           ('cf_origin_minus_control_support',all_control)]
        for method,deleted in conditions:
            new=dict(row);new['meta']=dict(meta,removed_turn_ids=deleted)
            keep=[t['turn_id'] for t in turns if t['turn_id'] not in deleted]
            new['method']=method;new['prompt_user']=render_subset(header,turns,keep)+question
            rows.append(new)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();rows=build('sgd')+build('multiwoz')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    print(f'Wrote {len(rows)} paired-deletion inputs')

if __name__=='__main__':main()

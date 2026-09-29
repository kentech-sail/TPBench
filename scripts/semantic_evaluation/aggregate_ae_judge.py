"""Compare blinded AI equivalence judgments with held-out human ratings.

Majority ties are unresolved. Abstentions count as incorrect in headline
accuracy, and are separately reported. Bootstrap resamples question clusters.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np

def read(p):return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def metrics(rs):
    labeled=[r for r in rs if r['majority_label'] is not None]
    d=dict(n=len(rs),resolved=len(labeled),human_ties=len(rs)-len(labeled))
    for method in ['judge','exact_match','token_f1_ge_half']:
        tp=tn=fp=fn=up=un=0
        for r in labeled:
            y=r['majority_label'];p=r[method]
            if p is None:
                up+=int(y);un+=int(not y)
            else:
                tp+=int(p and y);tn+=int(not p and not y)
                fp+=int(p and not y);fn+=int(not p and y)
        n=len(labeled);pos=tp+fn+up;neg=tn+fp+un
        d[method]=dict(tp=tp,tn=tn,fp=fp,fn=fn,uncertain_positive=up,uncertain_negative=un,
            accuracy=(tp+tn)/n if n else None,
            balanced_accuracy=(tp/pos+tn/neg)/2 if pos and neg else None,
            uncertainty_rate=(up+un)/n if n else None)
    return d

def main():
    ap=argparse.ArgumentParser()
    for k in ['inputs','key','judgments','model-record','output']:ap.add_argument('--'+k,type=Path,required=True)
    a=ap.parse_args();inputs=read(a.inputs);keys=read(a.key);js=read(a.judgments)
    assert len({r['id'] for r in inputs})==len(inputs)
    assert len({r['id'] for r in keys})==len(keys)==len(inputs)
    assert len({r['id'] for r in js})==len(js)==len(inputs)
    assert {r['id'] for r in inputs}=={r['id'] for r in keys}=={r['id'] for r in js}
    labels={r['id']:r['equivalent'] for r in js}
    assert all(x is None or type(x) is bool for x in labels.values())
    rs=[dict(r,judge=labels[r['id']]) for r in keys]
    groups=defaultdict(list)
    for r in rs:
        if r['majority_label'] is not None:groups[r['qid']].append(r)
    # n, positives, negatives, and correct/TP/TN counts for three methods.
    vals=[]
    for cl in groups.values():
        m=metrics(cl);v=[len(cl),sum(r['majority_label'] for r in cl),sum(not r['majority_label'] for r in cl)]
        for name in ['judge','exact_match','token_f1_ge_half']:
            s=m[name];v += [s['tp']+s['tn'],s['tp'],s['tn']]
        vals.append(v)
    vals=np.array(vals);rng=np.random.default_rng(20260928)
    draws=rng.integers(len(vals),size=(10000,len(vals)));sums=vals[draws].sum(axis=1)
    cis={};accuracies={}
    for i,name in enumerate(['judge','exact_match','token_f1_ge_half']):
        acc=sums[:,3+3*i]/sums[:,0];accuracies[name]=acc
        with np.errstate(divide='ignore',invalid='ignore'):
            ba=(sums[:,4+3*i]/sums[:,1]+sums[:,5+3*i]/sums[:,2])/2
        cis[name]=dict(accuracy=np.quantile(acc,[.025,.975]).tolist(),balanced_accuracy=np.nanquantile(ba,[.025,.975]).tolist())
    report=dict(status='AI judge agreement with external human ratings; not TPBench human annotation',
        model=json.loads(a.model_record.read_text()),overall=metrics(rs),ci95=cis,
        paired_accuracy_gain_ci95={k:np.quantile(accuracies['judge']-accuracies[k],[.025,.975]).tolist() for k in ['exact_match','token_f1_ge_half']},
        subsets={
            'non_exact':metrics([r for r in rs if not r['exact_match']]),
            'token_f1_below_half':metrics([r for r in rs if r['token_f1']<.5]),
            'human_disagreement':metrics([r for r in rs if r['disagreement']]),
            'human_unanimous':metrics([r for r in rs if not r['disagreement']])},
        bootstrap=dict(unit='question ID',clusters=len(groups),repetitions=10000,seed=20260928),
        individual_rating_agreement=sum((r['n_positive'] if r['judge'] is True else r['n_ratings']-r['n_positive'] if r['judge'] is False else 0) for r in rs)/sum(r['n_ratings'] for r in rs),
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.inputs,a.key,a.judgments,a.model_record]})
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(overall=report['overall'],ci95=cis,paired_gain_ci95=report['paired_accuracy_gain_ci95'])))

if __name__=='__main__':main()

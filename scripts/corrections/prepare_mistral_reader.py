"""Corrected Mistral requests; historical model revision was not recorded.
Unchanged saved scores may be reused; new outputs use the pinned current snapshot.
--artifact must be the original HF release at commit
11dcc75532f5d949a09b67f9c921eb254f3f1604. The corrected release already
ships the prepared requests for re-inference without downloading the old release.
"""
import argparse,json,hashlib,collections
from pathlib import Path
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--artifact',type=Path,required=True)
ap.add_argument('--contexts',type=Path)
ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--model-path',type=Path)
a=ap.parse_args();B=a.artifact.resolve();O=a.output;O.mkdir(parents=True,exist_ok=True)
context_path=a.contexts or Path(__file__).resolve().parents[2]/'data/llmlingua_correction/context_comparisons.jsonl'
MODEL='mistralai/Mistral-7B-Instruct-v0.3';REV='c170c708c41dac9275d15a8fff4eca08d52bab71';METHOD='llmlingua2_cache'
read=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s]
sha=lambda s:hashlib.sha256(s.encode()).hexdigest()
def request(p,probe):
 d=dict(prompt_system=p['prompt_system'],prompt_user=p['prompt_user'],max_new_tokens=128,attn_impl='eager' if probe=='P3' else 'sdpa',max_input_tokens=7168)
 return dict(id=sha(json.dumps(dict(d,model=MODEL,revision=REV,dtype='bfloat16',do_sample=False),sort_keys=True,ensure_ascii=False)),**d)
contexts={(x['dataset'],round(x['ratio']*100),x['dialogue_id']):x for x in read(context_path) if x['family']=='main'};assert len(contexts)==3374
cases=[]
for ds in ['sgd','multiwoz']:
 tag='sgd' if ds=='sgd' else 'mw'
 for seed in [42,43,44]:
  for pp,ratio in [('P2',10),('P2',30),('P3',30)]:
   probe_path=B/(f'data/probes/probes_{ds}_s{seed}_r{ratio}_p3_n200.jsonl' if pp=='P2' else f'data/probes/probes_{ds}_s{seed}_r30_n200_late_intent.jsonl')
   raw_path=B/(f'results/diagnostics/mistral_128/{tag}_r{ratio}_s{seed}_p3_reader_all.jsonl' if pp=='P2' else f'results/diagnostics/update_evidence/p3_{ds}_s{seed}_reader_mistral.jsonl')
   scored_path=B/(f'results/diagnostics/mistral_128/{tag}_r{ratio}_s{seed}_p3_scored.jsonl' if pp=='P2' else f'results/diagnostics/update_evidence/p3_{ds}_s{seed}_scored_mistral.jsonl')
   cases.append((ds,seed,pp,ratio,probe_path,raw_path,scored_path))
# Cross-method raw cache is intentionally NOT used: old model revision is unknown.
# Only unchanged original scored predictions are reused, transparently preserving baseline.
inputs={};expanded=[];reused=[];counts=collections.Counter()
for ds,seed,pp,ratio,probe_path,raw_path,scored_path in cases:
 scores={x['dialogue_id']:x for x in read(scored_path) if x['method']==METHOD}
 for p in read(probe_path):
  if p['method']!=METHOD:continue
  c=contexts[ds,ratio,p['dialogue_id']];old=p['compressed_text_used'];assert old==c['saved_context']
  prefix='Context:\n'+old;assert p['prompt_user'].startswith(prefix+'\n\nQuestion:')
  fixed=dict(p,compressed_text_used=c['corrected_context'],compressed_text_chars_used=len(c['corrected_context']),prompt_user='Context:\n'+c['corrected_context']+p['prompt_user'][len(prefix):])
  fixed['original_meta_k_turns']=fixed.get('meta',{}).get('k_turns')
  fixed['meta']=dict(fixed.get('meta',{}),k_turns=len(c['corrected_indices']))
  fixed['original_compressed_text_chars']=fixed.get('compressed_text_chars')
  fixed['compressed_text_chars']=len(c['corrected_context'])
  req=request(fixed,pp);expanded.append(dict(fixed,check_id=req['id'],paper_probe=pp,sample_seed=seed,original_probe_path=str(probe_path.relative_to(B)),original_score_path=str(scored_path.relative_to(B)),original_context_sha256=sha(old),corrected_context_sha256=sha(c['corrected_context']),corrected_kept_turn_indices=c['corrected_indices'],correction='word-label-only aggregation; original pool/prompt/recorded attention/token settings; new pinned snapshot because original revision unrecorded'))
  counts[f'{pp}_r{ratio}_expanded']+=1
  if old==c['corrected_context']:
   reused.append(dict(check_id=req['id'],dataset=ds,sample_seed=seed,paper_probe=pp,ratio=ratio/100,dialogue_id=p['dialogue_id'],score=scores[p['dialogue_id']],provenance='reused_unchanged_historical_Mistral_score_original_revision_unrecorded',source=str(scored_path.relative_to(B))));counts['unchanged_expanded']+=1
  else:inputs[req['id']]=req;counts['changed_expanded']+=1
for name,rows in [('inputs.jsonl',sorted(inputs.values(),key=lambda x:x['id'])),('key.jsonl',expanded),('reused_original_scores.jsonl',reused),('reused_raw_answers.jsonl',[])]:
 (O/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
s=dict(model=MODEL,revision=REV,original_revision=None,revision_note='current public snapshot pinned; original archived run has model ID but no revision',dtype='bfloat16',max_output_tokens=128,max_input_tokens=7168,counts=dict(counts),unique_new_requests=len(inputs),requests_by_attention=dict(collections.Counter(x['attn_impl'] for x in inputs.values())),reuse_policy='Unchanged original score only; no cross-method raw cache since historical immutable revision unavailable',no_paid_api=True)
(O/'preparation.json').write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))

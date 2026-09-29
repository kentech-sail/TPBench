"""Publish canonical model/runtime records without workstation paths."""
import argparse
import hashlib
import json
from pathlib import Path

FIELDS = ['model','revision','attention','dtype','greedy','batch_size','batch_tokens',
          'torch','transformers','python','gpu','runner_sha256','model_files','input_sha256']


def dump(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--staged',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--reader',choices=['llama','mistral'],required=True)
    a=p.parse_args()
    records={}
    if a.reader=='llama':
        phases=[('p1_eager_batch8','eager_p1_batch8','P1',96,'inputs_p1_batch8.jsonl'),
                ('p2_eager_serial','eager_p2_serial','P2',96,'inputs_p2_serial.jsonl'),
                ('p3_sdpa_serial','sdpa_serial','P3',128,'inputs.jsonl')]
    else:
        phases=[('p2_sdpa_serial','sdpa_serial','P2',128,'inputs.jsonl'),
                ('p3_eager_serial','eager_serial','P3',128,'inputs.jsonl')]
    for name, directory, probe, maxnew, requestfile in phases:
        original_record=json.loads((a.work/directory/'model_record.json').read_text())
        record={k:original_record[k] for k in FIELDS if k in original_record}
        record.update(primary_paper_probe=probe,max_input_tokens=7168,max_new_tokens=maxnew,
                      requests=f'data/llmlingua_correction/{a.reader}/{requestfile}',
                      answers=f'results/llmlingua_correction/{a.reader}/answers.jsonl')
        requestpath=a.staged/record['requests']
        assert hashlib.sha256(requestpath.read_bytes()).hexdigest()==record['input_sha256']
        relative=f'results/llmlingua_correction/{a.reader}/{name}_model_record.json'
        dump(a.staged/relative,record)
        records[probe]=relative
    summary={
        'reader':a.reader,'status':'complete_primary_local_rerun',
        'scope':'LLMLingua-derived main conditions only; unchanged methods retain their saved evidence',
        'primary_model_records':records,'paid_api_calls_performed':0,
        'logical_request_id':'SHA256 of exact system/user prompts, model ID/revision, dtype, greedy flag, attention, and input/output token limits; execution batch and library versions are separately recorded.',
        'reuse':'Original parsed scores are reused only for exactly unchanged contexts. Original raw answers are reused only for identical prompt/model/attention/token settings. Reused rows retain their original execution provenance.',
        'limitations':'Reader-score deltas combine corrected selections with recorded software/runtime changes and are not an experiment isolating the parser as the sole cause of every token difference.'}
    if a.reader=='llama':
        historic='results/historical/llmlingua_llama_legacy_20260929/results/reader_evaluation/eager_model_record.json'
        old=json.loads((a.original/'results/reader_evaluation/eager_model_record.json').read_text())
        dump(a.staged/historic,old)
        summary['historical_P1_record']=historic
        summary['runtime_comparison']={
            'model_revision':'Same recorded original Llama revision 0e9e39f249a16976918f6564b8830bc894c89659',
            'P1_batch':'Historical batch 8 / token budget 8000; corrected primary batch 8 / token budget 8000. Earlier batch 4 outputs are diagnostic only.',
            'P2_P3_batch':'Corrected primary uses serial batch 1, preserving the original main workers. Earlier batched outputs are diagnostic only.',
            'original_P1_python':old['python'],'original_P1_torch':old['torch'],
            'original_P1_transformers':old['transformers'],
            'corrected_python':original_record['python'],'corrected_torch':original_record['torch'],
            'corrected_transformers':original_record['transformers']}
        mixed={k:old[k] for k in ['model','revision','attention','dtype','greedy','batch_size','batch_tokens']}
        mixed.update(status='mixed_saved_executions',
            original_execution_record=historic,
            corrected_LLMLingua_P1_execution_record=records['P1'],
            corrected_fresh_request_ids='data/llmlingua_correction/llama/inputs_p1_batch8.jsonl',
            record_selection='Answer IDs in the corrected fresh request list use the corrected P1 record. Other unchanged/reused answers retain original execution provenance.',
            released_input_sha256=hashlib.sha256((a.staged/'data/reader_evaluation/inputs.jsonl').read_bytes()).hexdigest())
        dump(a.staged/'results/reader_evaluation/eager_model_record.json',mixed)
    else:
        summary['runtime_comparison']={
            'original_revision':'Not recorded in the original saved Mistral outputs',
            'corrected_revision':'c170c708c41dac9275d15a8fff4eca08d52bab71',
            'revision_comparability':'New explicitly pinned rerun; equality with the unrecorded original revision cannot be established',
            'original_runtime_versions':'Not recorded in original Mistral raw outputs',
            'corrected_python':original_record['python'],'corrected_torch':original_record['torch'],
            'corrected_transformers':original_record['transformers'],
            'preserved_settings':'bfloat16; greedy; P2 SDPA / P3 eager attention; serial batch 1; max input 7168 / max new tokens 128'}
    dump(a.staged/f'results/llmlingua_correction/{a.reader}/provenance.json',summary)
    overall=a.staged/'results/llmlingua_correction/provenance.json'
    current=json.loads(overall.read_text()) if overall.exists() else {'readers':{}}
    current['readers'][a.reader]=summary
    current['complete']=set(current['readers'])=={'llama','mistral'}
    dump(overall,current)
    print(f'OK: {a.reader} model/runtime records published without local model paths')


if __name__=='__main__':main()

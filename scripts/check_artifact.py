"""Validate distributed files, schemas, code syntax, and metadata without inference."""
import ast
import hashlib
import json
from pathlib import Path
from mlcroissant import Dataset

ROOT=Path(__file__).resolve().parents[1]

def main():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    files={str(p.relative_to(ROOT)) for p in ROOT.rglob('*') if p.is_file()
        and '__pycache__' not in p.parts and '.git' not in p.parts
        and p.relative_to(ROOT).parts[0] != 'outputs'}
    assert files==set(manifest['files'])|{'manifest.json'},'Missing or unlisted release files'
    for rel,digest in manifest['files'].items():
        p=ROOT/rel
        assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,rel
        if p.suffix=='.py':ast.parse(p.read_text(),filename=rel)
        elif p.suffix=='.json':json.loads(p.read_text())
        elif p.suffix=='.jsonl':
            for i,line in enumerate(p.open(),1):
                if line.strip():assert isinstance(json.loads(line),dict),(rel,i)
    for path in ['croissant.json','dataset_card/croissant.json']:
        dataset=Dataset(jsonld=str(ROOT/path))
        errors=list(dataset.metadata.issues.errors)
        assert not errors,(path,errors)
    assert (ROOT/'croissant.json').read_bytes()==(ROOT/'dataset_card/croissant.json').read_bytes()
    for p in (ROOT/'data/probes').glob('*.jsonl'):
        keys=set()
        for r in map(json.loads,p.open()):
            assert all(k in r for k in ['dialogue_id','dataset','method','probe_type','gold','prompt_system','prompt_user'])
            key=(r['dialogue_id'],r['method'],r['probe_type'])
            assert key not in keys,(p,key)
            keys.add(key)
    print(f'OK: {len(files)} release files, hashes, JSON, Python syntax, and Croissant metadata')

if __name__=='__main__':main()

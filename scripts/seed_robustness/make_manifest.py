#!/usr/bin/env python3
"""Write SHA-256 fingerprints for the released seed-robustness files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/seed_robustness/manifest.json"
FILES = [
    "data/probes/probes_risawoz_all207_r30_s42-46.jsonl",
    "data/probes/probes_lme_ku_all72_r30_s42-46.jsonl",
    "results/seed_robustness/experiment.json",
    "results/seed_robustness/reader/risawoz_all207_r30_llama.jsonl",
    "results/seed_robustness/reader/lme_ku_all72_r30_llama.jsonl",
    "results/seed_robustness/scored/risawoz_p2.jsonl",
    "results/seed_robustness/scored/lme_ku_p2.jsonl",
    "results/seed_robustness/aggregates/risawoz_p2.json",
    "results/seed_robustness/aggregates/lme_ku_p2.json",
    "results/seed_robustness/stats/summary.json",
    "results/seed_robustness/stats/summary.md",
    "scripts/seed_robustness/build_seed_robustness.py",
    "scripts/seed_robustness/bootstrap_stats.py",
    "scripts/seed_robustness/validate_seed_robustness.py",
    "scripts/scorer/scorer_p1_zh.py",
    "scripts/scorer/scorer_p3.py",
    "scripts/scorer/scorer_lme_ku.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    missing = [relative for relative in FILES if not (ROOT / relative).is_file()]
    if missing:
        raise SystemExit(f"missing files: {missing}")
    manifest = {
        "experiment_id": "tpbench_seed_robustness",
        "release": "1.0.0",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "model_revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "selector_seeds": [42, 43, 44, 45, 46],
        "bootstrap": {
            "unit": "dialogue_id",
            "replicates": 10000,
            "confidence_level": 0.95,
            "seed": 20260811,
        },
        "row_counts": {
            "risawoz_probes": 1863,
            "longmemeval_ku_probes": 648,
            "risawoz_reader": 1863,
            "longmemeval_ku_reader": 648,
        },
        "reader_format_errors": {
            "risawoz": 3,
            "longmemeval_ku": 0,
        },
        "files": {relative: sha256(ROOT / relative) for relative in FILES},
    }
    OUTPUT.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {OUTPUT.relative_to(ROOT)} with {len(FILES)} hashes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

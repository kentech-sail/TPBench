#!/usr/bin/env python3
"""Build the all-eligible RiSAWOZ and LongMemEval-KU robustness probes.

This script uses the released TPBench builders and changes only the random
turn selector.  Random selection is derived from the dialogue identifier and
an explicit selector seed, making each generated context deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RI_OUT = ROOT / "data/probes/probes_risawoz_all207_r30_s42-46.jsonl"
DEFAULT_LME_OUT = ROOT / "data/probes/probes_lme_ku_all72_r30_s42-46.jsonl"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def random_indices(dialogue_id: str, n: int, k: int, seed: int) -> list[int]:
    digest = hashlib.sha256(f"{seed}|{dialogue_id}".encode()).hexdigest()
    rng = random.Random(int(digest[:16], 16) % (2**32))
    return sorted(rng.sample(range(n), min(k, n)))


def risawoz_candidates(builder, source: Path) -> list[dict]:
    candidates = []
    for raw_record in builder.load_risawoz(source):
        record = builder.normalize_dialogue(raw_record)
        if record is None or len(record["turns"]) < 6:
            continue
        n_turns = len(record["turns"])
        transitions = [
            item for item in builder.find_transitions(record["turns"])
            if 3 <= item["transition_turn_id"] <= n_turns - 2
        ]
        if transitions:
            record["transition"] = max(
                transitions, key=lambda item: item["transition_turn_id"]
            )
            candidates.append(record)
    return sorted(candidates, key=lambda item: item["dialogue_id"])


def build_risawoz(source: Path, output: Path, ratio: float,
                   cap: int, seeds: list[int]) -> int:
    builder = load_module(
        "tpbench_risawoz_builder",
        ROOT / "scripts/builders/build_risawoz_probes.py",
    )
    candidates = risawoz_candidates(builder, source)
    if len(candidates) != 207:
        raise RuntimeError(f"expected 207 eligible RiSAWOZ dialogues, got {len(candidates)}")
    rows = []
    for record in candidates:
        n_turns = len(record["turns"])
        k = max(1, round(ratio * n_turns))
        methods = [
            ("full_context", builder.sel_full(n_turns)),
            ("recency", builder.sel_recency(n_turns, k)),
            ("first_n", builder.sel_first_n(n_turns, k)),
            ("uniform_stride", builder.sel_uniform(n_turns, k)),
        ]
        methods.extend(
            (f"random_seed{seed}", random_indices(record["dialogue_id"], n_turns, k, seed))
            for seed in seeds
        )
        for method, indices in methods:
            k_effective = n_turns if method == "full_context" else k
            context = builder.render(record["turns"], indices, cap=cap)
            p2 = builder.build_p3_row(
                record, record["transition"], method, k_effective, ratio, context, cap
            )
            if p2["gold"]:
                rows.append(p2)
            # This replication evaluates current-value recovery (P2).
    write_jsonl(output, rows)
    return len(rows)


def build_lme(source: Path, output: Path, ratio: float,
              cap: int, seeds: list[int]) -> int:
    transition_builder = ROOT / "scripts/builders/build_lme_transitions.py"
    probe_builder = load_module(
        "tpbench_lme_builder", ROOT / "scripts/builders/build_lme_probes.py"
    )
    with tempfile.TemporaryDirectory(prefix="tpbench-lme-transitions.") as tmp:
        transitions = Path(tmp) / "lme_ku.jsonl"
        subprocess.run(
            [sys.executable, str(transition_builder), "--input", str(source),
             "--output", str(transitions)],
            check=True,
        )
        records = sorted(load_jsonl(transitions), key=lambda item: item["dialogue_id"])
    if len(records) != 72:
        raise RuntimeError(f"expected 72 eligible LongMemEval-KU records, got {len(records)}")
    rows = []
    for record in records:
        turns = record["turns"]
        n_turns = len(turns)
        k = max(1, round(ratio * n_turns))
        methods = [
            ("full_context", probe_builder.sel_full(n_turns)),
            ("recency", probe_builder.sel_recency(n_turns, k)),
            ("first_n", probe_builder.sel_first_n(n_turns, k)),
            ("uniform_stride", probe_builder.sel_uniform_stride(n_turns, k)),
        ]
        methods.extend(
            (f"random_seed{seed}", random_indices(record["dialogue_id"], n_turns, k, seed))
            for seed in seeds
        )
        for method, indices in methods:
            k_effective = n_turns if method == "full_context" else k
            context = probe_builder.render(turns, indices, per_turn_char_cap=cap)
            row = probe_builder.build_one(record, method, context, ratio, k_effective)
            row["meta"]["per_turn_char_cap"] = cap
            if row["gold"]:
                rows.append(row)
    write_jsonl(output, rows)
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--risawoz-src", type=Path, required=True)
    parser.add_argument("--longmemeval-src", type=Path, required=True)
    parser.add_argument("--risawoz-out", type=Path, default=DEFAULT_RI_OUT)
    parser.add_argument("--longmemeval-out", type=Path, default=DEFAULT_LME_OUT)
    parser.add_argument("--ratio", type=float, default=0.30)
    parser.add_argument("--per-turn-char-cap", type=int, default=200)
    parser.add_argument("--selector-seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    args = parser.parse_args()
    ri_rows = build_risawoz(
        args.risawoz_src, args.risawoz_out, args.ratio,
        args.per_turn_char_cap, args.selector_seeds,
    )
    lme_rows = build_lme(
        args.longmemeval_src, args.longmemeval_out, args.ratio,
        args.per_turn_char_cap, args.selector_seeds,
    )
    print(json.dumps({
        "risawoz_rows": ri_rows,
        "longmemeval_ku_rows": lme_rows,
        "selector_seeds": args.selector_seeds,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

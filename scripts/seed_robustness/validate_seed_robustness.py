#!/usr/bin/env python3
"""Validate the released RiSAWOZ and LongMemEval-KU robustness files."""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/seed_robustness"
MODEL = (
    "meta-llama/Llama-3.1-8B-Instruct@"
    "0e9e39f249a16976918f6564b8830bc894c89659"
)
SEEDS = [42, 43, 44, 45, 46]
METHODS = {"full_context", "recency", "first_n", "uniform_stride"} | {
    f"random_seed{seed}" for seed in SEEDS
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate_probes(path: Path, rows_expected: int, dialogues_expected: int,
                    probe_types: set[str]) -> None:
    rows = load_jsonl(path)
    assert len(rows) == rows_expected, (path, len(rows), rows_expected)
    assert {row["method"] for row in rows} == METHODS
    assert {row["probe_type"] for row in rows} == probe_types
    assert len({row["dialogue_id"] for row in rows}) == dialogues_expected
    keys = {(row["dialogue_id"], row["method"], row["probe_type"]) for row in rows}
    assert len(keys) == rows_expected


def validate_reader(path: Path, rows_expected: int, errors_expected: int) -> None:
    rows = load_jsonl(path)
    assert len(rows) == rows_expected
    assert len({row["task_uid"] for row in rows}) == rows_expected
    assert {row["model"] for row in rows} == {MODEL}
    assert sum(bool(row.get("error")) for row in rows) == errors_expected


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, abs_tol=5e-5), (actual, expected)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest() -> None:
    manifest = json.loads((RESULTS / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["release"] == "1.0.0"
    assert manifest["selector_seeds"] == SEEDS
    assert manifest["reader_format_errors"] == {
        "risawoz": 3, "longmemeval_ku": 0
    }
    for relative, expected in manifest["files"].items():
        path = ROOT / relative
        assert path.is_file(), path
        assert sha256(path) == expected, path


def validate_summary() -> None:
    summary = json.loads((RESULTS / "stats/summary.json").read_text(encoding="utf-8"))
    assert summary["protocol"] == {
        "bootstrap_seed": 20260811,
        "bootstrap_replicates": 10000,
        "bootstrap_unit": "dialogue_id",
        "confidence_level": 0.95,
    }
    expected = {
        ("risawoz_p2", "p3_correct", "full_context"): 0.2318840580,
        ("risawoz_p2", "p3_correct", "random_selection"): 0.1497584541,
        ("lme_ku_p2", "p3_em_strict", "full_context"): 0.4166666667,
        ("lme_ku_p2", "p3_em_strict", "random_selection"): 0.2277777778,
    }
    for (section, metric, method), value in expected.items():
        close(summary[section][metric][method]["mean"], value)


def main() -> int:
    validate_probes(
        ROOT / "data/probes/probes_risawoz_all207_r30_s42-46.jsonl",
        1863, 207, {"P3"},
    )
    validate_probes(
        ROOT / "data/probes/probes_lme_ku_all72_r30_s42-46.jsonl",
        648, 72, {"P3"},
    )
    validate_reader(RESULTS / "reader/risawoz_all207_r30_llama.jsonl", 1863, 3)
    validate_reader(RESULTS / "reader/lme_ku_all72_r30_llama.jsonl", 648, 0)
    for filename, count in (
        ("risawoz_p2.jsonl", 1863),
        ("lme_ku_p2.jsonl", 648),
    ):
        assert len(load_jsonl(RESULTS / "scored" / filename)) == count
    validate_summary()
    validate_manifest()
    print("OK: seed robustness files validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

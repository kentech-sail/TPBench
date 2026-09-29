#!/usr/bin/env python3
"""Audit correction-cue firing rates for included vs. filtered-out P2 rows.

This script backs the filter-bias check discussed in the paper. It compares a
simple USER correction-cue regex on:

  1. rows included in the released seed-42 r=0.30 state probe, and
  2. rows in an upstream filtered-gold file that did not enter that probe.

The released artifact ships the probe JSONL files. The upstream filtered-gold
JSONL files are not duplicated in the artifact; pass them explicitly with
`--filtered-gold-sgd` and `--filtered-gold-multiwoz` when auditing from raw
preprocessed sources.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


CORRECTION_CUE_RE = re.compile(
    r"\b("
    r"actually|instead|rather|on second thought|scrap that|"
    r"wait,? (?:let me|i)|let me change|let's change|change (?:it|that|the)|"
    r"no,? (?:make it|change it|i (?:want|need|meant))|not (?:that|this)|"
    r"i meant|i'd rather|i would rather|i changed my mind|never mind"
    r")\b",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def fires_correction_cue(turns: list[str]) -> bool:
    """Return true if any USER turn has a correction cue.

    The probe builders render USER turns at even indices.
    """
    for i, turn in enumerate(turns):
        if i % 2 == 0 and CORRECTION_CUE_RE.search(turn or ""):
            return True
    return False


def slot_human_overlap(turns: list[str], slot_human: str | None) -> bool:
    if not slot_human:
        return False
    tokens = [t for t in re.split(r"\s+", slot_human.lower()) if len(t) >= 3]
    if not tokens:
        return False
    text = " ".join(turns).lower()
    return any(token in text for token in tokens)


def probe_keys(path: Path) -> set[tuple[str, str]]:
    keys: set[tuple[str, str]] = set()
    for row in load_jsonl(path):
        keys.add((str(row.get("dialogue_id")), str(row.get("slot") or "")))
    return keys


def audit_dataset(dataset: str, filtered_gold: Path, probe_path: Path) -> dict:
    post_filter_rows = load_jsonl(filtered_gold)
    included = probe_keys(probe_path)

    stats = {
        "dataset": dataset,
        "filtered_gold_total": len(post_filter_rows),
        "included_total": 0,
        "included_correction_cue": 0,
        "included_slot_human_overlap": 0,
        "excluded_total": 0,
        "excluded_correction_cue": 0,
        "excluded_slot_human_overlap": 0,
    }

    for row in post_filter_rows:
        key = (str(row.get("dialogue_id")), str(row.get("slot") or ""))
        turns = row.get("turns") or []
        if not isinstance(turns, list):
            turns = []
        cue = fires_correction_cue([str(t) for t in turns])
        slot_match = slot_human_overlap([str(t) for t in turns], row.get("slot_human"))
        if key in included:
            stats["included_total"] += 1
            stats["included_correction_cue"] += int(cue)
            stats["included_slot_human_overlap"] += int(slot_match)
        else:
            stats["excluded_total"] += 1
            stats["excluded_correction_cue"] += int(cue)
            stats["excluded_slot_human_overlap"] += int(slot_match)

    for prefix in ("included", "excluded"):
        total = max(1, int(stats[f"{prefix}_total"]))
        stats[f"{prefix}_correction_cue_rate"] = round(
            int(stats[f"{prefix}_correction_cue"]) / total, 6
        )
        stats[f"{prefix}_slot_human_overlap_rate"] = round(
            int(stats[f"{prefix}_slot_human_overlap"]) / total, 6
        )
    stats["correction_cue_rate_gap_pp"] = round(
        100
        * (
            stats["included_correction_cue_rate"]
            - stats["excluded_correction_cue_rate"]
        ),
        3,
    )
    return stats


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--filtered-gold-sgd", type=Path, required=True)
    parser.add_argument("--filtered-gold-multiwoz", type=Path, required=True)
    parser.add_argument(
        "--probe-sgd",
        type=Path,
        default=root / "data/probes/probes_sgd_s42_r30_p3_n200.jsonl",
    )
    parser.add_argument(
        "--probe-multiwoz",
        type=Path,
        default=root / "data/probes/probes_multiwoz_s42_r30_p3_n200.jsonl",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = [
        audit_dataset("sgd", args.filtered_gold_sgd, args.probe_sgd),
        audit_dataset("multiwoz", args.filtered_gold_multiwoz, args.probe_multiwoz),
    ]
    text = json.dumps(out, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

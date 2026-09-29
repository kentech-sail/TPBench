#!/usr/bin/env python3
"""Validate the synthetic FMTS support and matched-control deletions."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCRIPT = Path(__file__).resolve()
PACKAGE = SCRIPT.parents[2]
DEFAULT_DIR = PACKAGE / "results" / "diagnostics" / "update_evidence"
sys.path.insert(0, str(PACKAGE / "scripts" / "scorer"))
sys.path.insert(0, str(PACKAGE / "scripts"))
from normalize import normalize_value, normalized_phrase_occurs  # type: ignore  # noqa: E402
from scorer_p3 import extract_json, score_p3  # type: ignore  # noqa: E402


CONDITIONS = ("cf4_full", "cf4_minus_support", "cf4_minus_control")
TURN_RE = re.compile(r"^\[T(\d+)\]\s+([A-Za-z]+):", re.MULTILINE)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def context(prompt: str) -> str:
    body = prompt.split("\n\nQuestion:", 1)[0]
    return body.removeprefix("Context:\n")


def turn_speakers(prompt: str) -> dict[int, str]:
    return {int(turn_id): speaker.upper() for turn_id, speaker in TURN_RE.findall(context(prompt))}


def fmts_values_match_loose(prediction: str | None, reference: str | None) -> bool:
    """Apply the free-form FMTS loose rule in either phrase direction."""

    normalized_prediction = normalize_value(prediction, "fmts_indirect")
    normalized_reference = normalize_value(reference, "fmts_indirect")
    if not normalized_prediction or not normalized_reference:
        return False
    return normalized_phrase_occurs(
        normalized_prediction, normalized_reference
    ) or normalized_phrase_occurs(normalized_reference, normalized_prediction)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_DIR / "fmts_support_ablation_input.jsonl",
    )
    parser.add_argument(
        "--reader",
        type=Path,
        default=DEFAULT_DIR / "fmts_support_ablation_reader_llama.jsonl",
    )
    parser.add_argument(
        "--scored",
        type=Path,
        default=DEFAULT_DIR / "fmts_support_ablation_scored_llama.jsonl",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_DIR / "fmts_support_ablation_summary_llama.json",
    )
    args = parser.parse_args()

    probes = read_jsonl(args.input)
    readers = read_jsonl(args.reader)
    by_item: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in probes:
        if row.get("dataset") != "fmts_indirect" or row.get("probe_type") != "CF4":
            raise SystemExit(f"unexpected FMTS probe row: {row.get('dialogue_id')}")
        did, method = str(row["dialogue_id"]), str(row["method"])
        if method not in CONDITIONS or method in by_item[did]:
            raise SystemExit(f"bad or duplicate FMTS condition: {did}/{method}")
        by_item[did][method] = row

    if len(by_item) != 100 or any(set(rows) != set(CONDITIONS) for rows in by_item.values()):
        raise SystemExit("FMTS input must contain exactly 100 complete three-condition items")

    validation = Counter()
    for did, rows in by_item.items():
        full = rows["cf4_full"]
        minus_support = rows["cf4_minus_support"]
        minus_control = rows["cf4_minus_control"]
        meta = full.get("meta") or {}
        support = [int(x) for x in meta.get("cf4_support") or []]
        control = [int(x) for x in meta.get("cf4_control") or []]
        if len(support) != 3 or len(control) != len(support) or set(support) & set(control):
            raise SystemExit(f"invalid support/control set for {did}")
        for row in (minus_support, minus_control):
            row_meta = row.get("meta") or {}
            if row_meta.get("cf4_support") != support or row_meta.get("cf4_control") != control:
                raise SystemExit(f"condition metadata mismatch for {did}")

        full_turns = turn_speakers(str(full["prompt_user"]))
        support_turns = turn_speakers(str(minus_support["prompt_user"]))
        control_turns = turn_speakers(str(minus_control["prompt_user"]))
        if set(support_turns) != set(full_turns) - set(support):
            raise SystemExit(f"support deletion does not match metadata for {did}")
        if set(control_turns) != set(full_turns) - set(control):
            raise SystemExit(f"control deletion does not match metadata for {did}")
        if Counter(full_turns[x] for x in support) != Counter(full_turns[x] for x in control):
            raise SystemExit(f"control speaker composition mismatch for {did}")
        validation["exact_support_deletions"] += 1
        validation["exact_control_deletions"] += 1
        validation["speaker_matched_controls"] += 1
        gold = str(full["gold"])
        if normalize_value(gold, "fmts_indirect") in normalize_value(
            context(str(minus_support["prompt_user"])), "fmts_indirect"
        ):
            validation["gold_leaks_after_support_removal"] += 1

    reader_map: dict[tuple[str, str], dict[str, Any]] = {}
    models: Counter[str] = Counter()
    for row in readers:
        key = (str(row.get("dialogue_id")), str(row.get("method")))
        if key in reader_map:
            raise SystemExit(f"duplicate FMTS reader row: {key}")
        reader_map[key] = row
        models[str(row.get("model"))] += 1
    required = {(did, condition) for did in by_item for condition in CONDITIONS}
    if set(reader_map) != required:
        raise SystemExit(
            f"FMTS reader coverage mismatch: missing={len(required-set(reader_map))}, "
            f"extra={len(set(reader_map)-required)}"
        )

    scored: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    scored_rows: list[dict[str, Any]] = []
    reader_errors = 0
    for did in sorted(by_item):
        for condition in CONDITIONS:
            probe = by_item[did][condition]
            reader = reader_map[(did, condition)]
            error = reader.get("error")
            if error:
                reader_errors += 1
                score = {
                    "p3_correct": 0,
                    "p3_correct_loose": 0,
                    "abstain": 0,
                    "norm_pred": "",
                }
            else:
                score = score_p3(
                    reader.get("reader_output_text") or "",
                    probe["gold"],
                    context(str(probe["prompt_user"])),
                    "fmts_indirect",
                )
                parsed = extract_json(reader.get("reader_output_text") or "")
                predicted_value = parsed.get("value")
                prediction = predicted_value if isinstance(predicted_value, str) else None
                score["p3_correct_loose"] = int(
                    not score["abstain"]
                    and fmts_values_match_loose(prediction, probe["gold"])
                )
            norm_old = normalize_value(probe.get("old_value"), "fmts_indirect")
            norm_pred = str(score.get("norm_pred") or "")
            score["old_state_reversion"] = int(bool(norm_old) and norm_old in norm_pred)
            score["error"] = error
            scored[did][condition] = score
            scored_rows.append(
                {
                    "dialogue_id": did,
                    "dataset": "fmts_indirect",
                    "method": condition,
                    "probe_type": "CF4",
                    **score,
                }
            )

    summary: dict[str, Any] = {
        "status": "pass",
        "n_items": len(scored),
        "n_reader_errors": reader_errors,
        "reader_models": dict(models),
        "input_sha256": sha256_file(args.input),
        "reader_sha256": sha256_file(args.reader),
        "reader_configuration": {
            "dtype": "bfloat16",
            "attention_implementation": "sdpa",
            "decoding": "greedy",
            "max_new_tokens": 128,
            "max_input_tokens": 7168,
        },
        "scoring_rule": {
            "strict": "normalized_exact_match",
            "loose": "bounded_normalized_containment_either_direction",
            "scope": "synthetic_fmts_support_ablation",
        },
        "construction_validation": {
            **dict(validation),
            "gold_leaks_after_support_removal": validation[
                "gold_leaks_after_support_removal"
            ],
            "authored_support_size": 3,
        },
    }
    if validation["gold_leaks_after_support_removal"] != 0:
        raise SystemExit("normalized gold remains after FMTS support deletion")

    for metric in ("p3_correct", "p3_correct_loose"):
        metric_summary: dict[str, Any] = {
            condition: sum(scored[did][condition][metric] for did in scored) / len(scored)
            for condition in CONDITIONS
        }
        full_correct = [did for did in scored if scored[did]["cf4_full"][metric]]
        metric_summary.update(
            {
                "counterfactual_validity_rate": sum(
                    not scored[did]["cf4_minus_support"][metric]
                    and scored[did]["cf4_minus_control"][metric]
                    for did in full_correct
                )
                / len(scored),
                "n_full_correct": len(full_correct),
                "support_flip_given_full_correct": sum(
                    not scored[did]["cf4_minus_support"][metric] for did in full_correct
                )
                / max(1, len(full_correct)),
                "control_keep_given_full_correct": sum(
                    scored[did]["cf4_minus_control"][metric] for did in full_correct
                )
                / max(1, len(full_correct)),
            }
        )
        summary[metric] = metric_summary
    for metric in ("abstain", "old_state_reversion"):
        summary[f"{metric}_rate"] = {
            condition: sum(scored[did][condition][metric] for did in scored) / len(scored)
            for condition in CONDITIONS
        }

    args.scored.parent.mkdir(parents=True, exist_ok=True)
    with args.scored.open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

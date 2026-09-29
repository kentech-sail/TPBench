#!/usr/bin/env python3
"""Unicode-CJK P1 scorer for the TPBench RiSAWOZ experiment.

The scorer follows the P1 parsing, thresholds, token-F1 formula, and aggregate
schema. Each CJK codepoint is a content token; Latin and numeric runs remain
word tokens.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


LATIN_RE = re.compile(r"[a-z0-9]+")
CJK_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
)
LATIN_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "for",
    "of", "in", "on", "at", "by", "with", "and", "or", "i", "you",
    "we", "they", "this", "that", "my", "your", "please", "want", "need",
}


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in CJK_RANGES)


def content_tokens(text: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", text or "").lower()
    cjk = {ch for ch in normalized if is_cjk(ch)}
    latin = {
        tok for tok in LATIN_RE.findall(normalized)
        if tok not in LATIN_STOP and len(tok) >= 2
    }
    return cjk | latin


def parse_reader_text(text: str) -> dict | None:
    if not text:
        return None
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start, end = stripped.rfind("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(stripped[start:end + 1])
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def token_f1(gold: set[str], pred: set[str]) -> float:
    if not gold or not pred:
        return 0.0
    overlap = len(gold & pred)
    if not overlap:
        return 0.0
    precision = overlap / len(pred)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalized_contains(needle: str, haystack: str) -> int:
    needle_n = " ".join(unicodedata.normalize("NFKC", needle).lower().split())
    haystack_n = " ".join(unicodedata.normalize("NFKC", haystack).lower().split())
    return int(bool(needle_n) and needle_n in haystack_n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--reader-out", type=Path, required=True)
    ap.add_argument("--scored-out", type=Path, required=True)
    ap.add_argument("--aggregate-out", type=Path, required=True)
    args = ap.parse_args()

    probes = {
        (r["dialogue_id"], r["method"], r["probe_type"]): r
        for r in load_jsonl(args.probes) if r.get("probe_type") == "P1"
    }
    readers = load_jsonl(args.reader_out)
    args.scored_out.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_out.parent.mkdir(parents=True, exist_ok=True)
    totals = defaultdict(lambda: defaultdict(float))
    scored = []
    for row in readers:
        if row.get("probe_type") != "P1":
            continue
        key = (row.get("dialogue_id"), row.get("method"), "P1")
        probe = probes.get(key)
        if probe is None:
            continue
        parsed = parse_reader_text(row.get("reader_output_text") or "")
        abstain = parsed is None or parsed.get("abstain") is True
        pred = "" if abstain else str(parsed.get("value") or "")
        if not pred:
            abstain = True
        gold = str(probe.get("gold") or "")
        gold_tokens, pred_tokens = content_tokens(gold), content_tokens(pred)
        overlap = len(gold_tokens & pred_tokens)
        support = "" if parsed is None else parsed.get("support") or ""
        if isinstance(support, list):
            support = " ".join(str(x) for x in support if x)
        scored_row = {
            "dialogue_id": row.get("dialogue_id"),
            "dataset": row.get("dataset"),
            "method": row.get("method"),
            "probe_type": "P1",
            "meta": row.get("meta"),
            "error": row.get("error"),
            "p1_correct": int(not abstain and overlap >= 2),
            "p1_correct_loose": int(not abstain and overlap >= 1),
            "p1_token_f1": token_f1(gold_tokens, pred_tokens),
            "abstain": bool(abstain),
            "support_in_ctx": normalized_contains(
                str(support), str(probe.get("compressed_text_used") or "")
            ),
            "gold": gold,
            "pred_value": pred,
            "tokenization": "unicode_cjk_codepoint_plus_latin_runs_v1",
        }
        scored.append(scored_row)
        agg = totals[scored_row["method"]]
        agg["n"] += 1
        agg["strict"] += scored_row["p1_correct"]
        agg["loose"] += scored_row["p1_correct_loose"]
        agg["f1"] += scored_row["p1_token_f1"]
        agg["abstain"] += int(scored_row["abstain"])
        agg["support"] += scored_row["support_in_ctx"]
        agg["error"] += int(bool(scored_row["error"]))

    with args.scored_out.open("w", encoding="utf-8") as f:
        for row in scored:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    aggregate = {}
    for method, values in sorted(totals.items()):
        n = int(values["n"])
        aggregate[method] = {
            "n": n,
            "p1_em_strict": values["strict"] / n,
            "p1_em_loose": values["loose"] / n,
            "p1_token_f1": values["f1"] / n,
            "abstain_rate": values["abstain"] / n,
            "support_in_ctx_rate": values["support"] / n,
            "err_rate": values["error"] / n,
            "tokenization": "unicode_cjk_codepoint_plus_latin_runs_v1",
        }
    args.aggregate_out.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[scorer_p1_zh] scored {len(scored)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

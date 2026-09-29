#!/usr/bin/env python3
"""Score current-value answers (paper P2; serialized field prefix p3)."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

# Package-local frozen slot-value normalization.
#
# Import the package-local copy so the released scorer runs independently.
import sys

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PACKAGE_ROOT / "scripts"))
from normalize import (  # type: ignore  # noqa: E402
    normalize_value,
    values_match,
    values_match_loose,
)


P2_LOOSE_MATCHER = "reference_in_prediction_bounded_phrase_v2"


_STOP = {
    "the","a","an","of","to","in","on","for","at","by","is","are","was",
    "were","be","been","i","you","he","she","it","we","they","me","my",
    "your","his","her","our","their","this","that","these","those","and",
    "or","but","if","then","so","yes","no","not","do","does","did",
    "have","has","had","with","without","from","as","than","also","just",
    "about","please","thanks","thank","ok","okay",
}


def content_tokens(s: str | None) -> set[str]:
    if not s:
        return set()
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return {t for t in s.split() if t and t not in _STOP and len(t) > 1}


def extract_json(text: str | None) -> dict:
    if not text:
        return {}
    # take last balanced {...} block
    last_open = text.rfind("{")
    last_close = text.rfind("}")
    if last_open < 0 or last_close < 0 or last_close <= last_open:
        return {}
    snippet = text[last_open : last_close + 1]
    try:
        d = json.loads(snippet)
        return d if isinstance(d, dict) else {}
    except Exception:
        return {}


def score_p3(reader_text: str, gold_value: str | None,
             context: str | None, dataset: str = "sgd") -> dict:
    obj = extract_json(reader_text)
    val = obj.get("value")
    pred_str = val if isinstance(val, str) else None
    raw_sup = obj.get("support")
    if isinstance(raw_sup, list):
        support = " ".join(str(x) for x in raw_sup if x)
    elif isinstance(raw_sup, str):
        support = raw_sup
    else:
        support = ""
    abstain = bool(obj.get("abstain", False)) or (
        isinstance(val, str) and val.strip().lower() in {"", "unknown", "n/a", "none"}
    )

    # --- normalized scoring (primary) ---
    norm_pred = normalize_value(pred_str, dataset)
    norm_gold = normalize_value(gold_value, dataset)
    p3_correct_norm = int(
        (not abstain) and bool(norm_gold) and bool(norm_pred)
        and values_match(pred_str, gold_value, dataset)
    )
    # The loose answer scorer and answer-bearing-turn detector use the same
    # bounded matcher, preventing matches inside unrelated longer words.
    p3_correct_norm_loose = int(
        (not abstain) and values_match_loose(pred_str, gold_value, dataset)
    )

    # Auxiliary token-overlap diagnostics.
    pred_toks = content_tokens(pred_str)
    gold_toks = content_tokens(gold_value)
    overlap = len(pred_toks & gold_toks)
    p3_overlap_strict = int((not abstain) and overlap >= 2 and len(gold_toks) >= 2)
    p3_overlap_loose  = int((not abstain) and overlap >= 1 and len(gold_toks) >= 1)

    sup_in = 0
    if support and context:
        sn = normalize_value(support, dataset)
        cn = normalize_value(context, dataset)
        sup_in = int(bool(sn) and sn in cn)

    return {
        # primary metrics (normalized value match) - use these for paper claims
        "p3_correct": p3_correct_norm,
        "p3_correct_loose": p3_correct_norm_loose,
        # Auxiliary overlap metrics.
        "p3_overlap_strict": p3_overlap_strict,
        "p3_overlap_loose": p3_overlap_loose,
        "abstain": int(abstain),
        "support_in_ctx": sup_in,
        "pred_value": val,
        "gold_value": gold_value,
        "norm_pred": norm_pred,
        "norm_gold": norm_gold,
        "p2_loose_matcher": P2_LOOSE_MATCHER,
    }


def load_jsonl(p: Path) -> list[dict]:
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--reader-out", type=Path, required=True)
    ap.add_argument("--scored-out", type=Path, required=True)
    ap.add_argument("--aggregate-out", type=Path, required=True)
    args = ap.parse_args()

    probes = load_jsonl(args.probes)
    by_uid_p = {}
    for r in probes:
        # match the task_uid used by reader_worker_local.py, but we don't have
        # the prompt sha here. Match by (dialogue_id, method, probe_type) instead.
        key = (r.get("dialogue_id"), r.get("method"), r.get("probe_type"))
        by_uid_p[key] = r

    reader = load_jsonl(args.reader_out)
    args.scored_out.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_out.parent.mkdir(parents=True, exist_ok=True)

    agg: dict[tuple, dict] = defaultdict(lambda: defaultdict(int))
    with args.scored_out.open("w", encoding="utf-8") as fout:
        for o in reader:
            key = (o.get("dialogue_id"), o.get("method"), o.get("probe_type"))
            p = by_uid_p.get(key)
            if p is None or p.get("probe_type") != "P3":
                continue
            gold = ((p.get("meta") or {}).get("gold_value")
                    or p.get("gold_value")
                    or p.get("gold")
                    or p.get("new_value"))
            ctx = p.get("context") or p.get("prompt_user") or ""
            ds_for_norm = (p.get("dataset") or "sgd").lower()
            if "multi" in ds_for_norm or ds_for_norm.startswith("mw"):
                ds_for_norm = "multiwoz"
            err = o.get("error")
            if err:
                row = {
                    "dialogue_id": p["dialogue_id"], "method": p["method"],
                    "dataset": p["dataset"], "probe_type": "P3",
                    "p3_correct": 0, "p3_correct_loose": 0,
                    "p3_overlap_strict": 0, "p3_overlap_loose": 0,
                    "abstain": 0, "support_in_ctx": 0, "err": 1,
                    "p2_loose_matcher": P2_LOOSE_MATCHER,
                    "meta": p.get("meta"),
                }
            else:
                sc = score_p3(o.get("reader_output_text"), gold, ctx, ds_for_norm)
                row = {
                    "dialogue_id": p["dialogue_id"], "method": p["method"],
                    "dataset": p["dataset"], "probe_type": "P3",
                    **sc, "err": 0, "meta": p.get("meta"),
                }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            akey = (row["dataset"], row["method"])
            agg[akey]["n"] += 1
            agg[akey]["p3_em_strict"]  += row["p3_correct"]
            agg[akey]["p3_em_loose"]   += row["p3_correct_loose"]
            agg[akey]["p3_overlap_strict"] += row.get("p3_overlap_strict", 0)
            agg[akey]["p3_overlap_loose"]  += row.get("p3_overlap_loose",  0)
            agg[akey]["abstain"] += row["abstain"]
            agg[akey]["support_in_ctx"] += row["support_in_ctx"]
            agg[akey]["err"] += row["err"]

    out_agg = {}
    for (ds, method), d in agg.items():
        n = max(d["n"], 1)
        out_agg.setdefault(ds, {})[method] = {
            "n": d["n"],
            "p3_em_strict": round(d["p3_em_strict"] / n, 4),
            "p3_em_loose":  round(d["p3_em_loose"]  / n, 4),
            "p3_overlap_strict": round(d["p3_overlap_strict"] / n, 4),
            "p3_overlap_loose":  round(d["p3_overlap_loose"]  / n, 4),
            "abstain_rate": round(d["abstain"] / n, 4),
            "support_in_ctx_rate": round(d["support_in_ctx"] / n, 4),
            "err_rate": round(d["err"] / n, 4),
        }
    with args.aggregate_out.open("w", encoding="utf-8") as f:
        json.dump(out_agg, f, ensure_ascii=False, indent=2)
    print(f"[scorer_p3] wrote {sum(d['n'] for d in agg.values())} rows -> "
          f"{args.scored_out} + {args.aggregate_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

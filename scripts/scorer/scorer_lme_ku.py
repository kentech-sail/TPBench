#!/usr/bin/env python3
"""Deterministic LongMemEval-KU scorer for TPBench.

The scoring rules are:

* strict: an explicit answer variant matches after Unicode/case/punctuation
  normalization; a complete reference variant may occur at word boundaries in
  a longer prediction; and a numeric-only prediction may omit a trailing unit
  only when its complete numeric signature equals the reference signature;
* loose: strict, or at least half of the reference content tokens occur in the
  prediction.

JSON scalar numbers are accepted as candidate values, so ``{"value": 25}``
and ``{"value": "25"}`` are scored identically.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


SCORER_VERSION = "lme_ku_bounded_variants_v1"
_PUNCT_RE = re.compile(r"[^\w\s:/-]+")
_WS_RE = re.compile(r"\s+")
_TIME_AMPM_RE = re.compile(
    r"\b(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b",
    flags=re.IGNORECASE,
)
_TIME_HHMM_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_PAREN_OR_RE = re.compile(r"\((?:or\s+)?([^()]*)\)", flags=re.IGNORECASE)
_STOP = {
    "the", "a", "an", "of", "to", "in", "on", "for", "at", "by",
    "is", "are", "was", "were", "be", "been", "i", "you", "he",
    "she", "it", "we", "they", "me", "my", "your", "his", "her",
    "our", "their", "this", "that", "these", "those", "and", "or",
    "but", "if", "then", "so", "yes", "no", "not", "do", "does",
    "did", "have", "has", "had", "with", "without", "from", "as",
    "than", "also", "just", "about", "please", "thanks", "thank",
    "ok", "okay",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_json(text: str | None) -> dict:
    if not text:
        return {}
    stripped = text.strip()
    try:
        value = json.loads(stripped)
        return value if isinstance(value, dict) else {}
    except Exception:
        pass
    start, end = stripped.rfind("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        value = json.loads(stripped[start:end + 1])
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _normalize_time(match: re.Match[str]) -> str:
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = match.group(3).lower().replace(".", "")
    if ampm == "pm" and hour != 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    return f"{hour:02d}:{minute:02d}"


def normalize_value(value: object | None) -> str:
    if value is None or isinstance(value, bool):
        return ""
    text = unicodedata.normalize("NFKC", str(value)).lower().strip()
    text = _TIME_AMPM_RE.sub(_normalize_time, text)
    text = _TIME_HHMM_RE.sub(
        lambda match: f"{int(match.group(1)):02d}:{int(match.group(2)):02d}",
        text,
    )
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def answer_variants(gold: object | None) -> list[str]:
    """Return the displayed answer plus explicit parenthetical alternatives."""
    if gold is None:
        return []
    text = str(gold).strip()
    variants = [text]
    for match in _PAREN_OR_RE.finditer(text):
        candidate = match.group(1).strip(" .,;")
        if candidate:
            variants.append(candidate)
    base = _PAREN_OR_RE.sub(" ", text).strip(" .,;")
    if base:
        variants.append(base)
    return list(dict.fromkeys(variants))


def bounded_phrase_occurs(needle: str, haystack: str) -> bool:
    if not needle:
        return False
    return bool(re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack))


def _numeric_signature(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def numeric_only_equivalent(prediction: object, reference: str) -> bool:
    pred = normalize_value(prediction)
    ref = normalize_value(reference)
    if not pred or re.search(r"[a-z]", pred):
        return False
    pred_numbers = _numeric_signature(pred)
    ref_numbers = _numeric_signature(ref)
    return bool(pred_numbers) and pred_numbers == ref_numbers


def strict_match(prediction: object | None, gold: object | None) -> bool:
    pred = normalize_value(prediction)
    if not pred:
        return False
    for variant in answer_variants(gold):
        ref = normalize_value(variant)
        if not ref:
            continue
        if pred == ref or bounded_phrase_occurs(ref, pred):
            return True
        if numeric_only_equivalent(prediction, variant):
            return True
    return False


def content_tokens(value: object | None) -> set[str]:
    if value is None:
        return set()
    tokens = re.findall(r"[a-z0-9]+", normalize_value(value))
    return {token for token in tokens if len(token) > 1 and token not in _STOP}


def loose_match(prediction: object | None, gold: object | None) -> bool:
    if strict_match(prediction, gold):
        return True
    pred_tokens = content_tokens(prediction)
    gold_tokens = content_tokens(gold)
    if not pred_tokens or not gold_tokens:
        return False
    required = math.ceil(len(gold_tokens) / 2)
    return len(pred_tokens & gold_tokens) >= required


def value_in_context(gold: object | None, context: str) -> bool:
    normalized_context = normalize_value(context)
    return any(
        bounded_phrase_occurs(normalize_value(variant), normalized_context)
        for variant in answer_variants(gold)
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument("--reader-out", type=Path, required=True)
    parser.add_argument("--scored-out", type=Path, required=True)
    parser.add_argument("--aggregate-out", type=Path, required=True)
    args = parser.parse_args()

    probes = {
        (row.get("dialogue_id"), row.get("method"), row.get("probe_type")): row
        for row in load_jsonl(args.probes)
        if row.get("probe_type") == "P3"
    }
    readers = load_jsonl(args.reader_out)
    args.scored_out.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_out.parent.mkdir(parents=True, exist_ok=True)

    totals: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    scored_rows = []
    for reader in readers:
        key = (
            reader.get("dialogue_id"), reader.get("method"),
            reader.get("probe_type"),
        )
        probe = probes.get(key)
        if probe is None:
            continue
        gold = (
            (probe.get("meta") or {}).get("gold_value")
            or probe.get("gold_value") or probe.get("gold")
            or probe.get("new_value")
        )
        context = str(
            probe.get("compressed_text_used") or probe.get("context")
            or probe.get("prompt_user") or ""
        )
        parsed = extract_json(reader.get("reader_output_text"))
        prediction = parsed.get("value")
        abstain = bool(parsed.get("abstain", False)) or prediction is None
        if isinstance(prediction, str) and prediction.strip().lower() in {
            "", "unknown", "n/a", "none",
        }:
            abstain = True
        error = int(bool(reader.get("error")))
        strict = int(not error and not abstain and strict_match(prediction, gold))
        loose = int(not error and not abstain and loose_match(prediction, gold))
        support = parsed.get("support")
        if isinstance(support, list):
            support = " ".join(str(item) for item in support if item)
        support_in_context = int(
            bool(support) and bounded_phrase_occurs(
                normalize_value(support), normalize_value(context)
            )
        )
        row = {
            "dialogue_id": probe.get("dialogue_id"),
            "method": probe.get("method"),
            "dataset": probe.get("dataset"),
            "probe_type": "P3",
            "p3_em_strict": strict,
            "p3_em_loose": loose,
            "abstain": int(abstain),
            "support_in_ctx": support_in_context,
            "gold_in_ctx": int(value_in_context(gold, context)),
            "pred_value": prediction,
            "gold_value": gold,
            "err": error,
            "scorer_version": SCORER_VERSION,
            "meta": probe.get("meta"),
        }
        scored_rows.append(row)
        aggregate_key = (str(row["dataset"]), str(row["method"]))
        totals[aggregate_key]["n"] += 1
        totals[aggregate_key]["strict"] += strict
        totals[aggregate_key]["loose"] += loose
        totals[aggregate_key]["abstain"] += row["abstain"]
        totals[aggregate_key]["support"] += support_in_context
        totals[aggregate_key]["gold"] += row["gold_in_ctx"]
        totals[aggregate_key]["error"] += error

    with args.scored_out.open("w", encoding="utf-8") as handle:
        for row in scored_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    aggregate: dict[str, dict[str, dict]] = {}
    for (dataset, method), values in sorted(totals.items()):
        n = max(values["n"], 1)
        aggregate.setdefault(dataset, {})[method] = {
            "n": values["n"],
            "p3_em_strict": round(values["strict"] / n, 4),
            "p3_em_loose": round(values["loose"] / n, 4),
            "abstain_rate": round(values["abstain"] / n, 4),
            "support_in_ctx_rate": round(values["support"] / n, 4),
            "gold_in_ctx_rate": round(values["gold"] / n, 4),
            "err_rate": round(values["error"] / n, 4),
            "scorer_version": SCORER_VERSION,
        }
    args.aggregate_out.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[scorer_lme_ku] scored {len(scored_rows)} rows ({SCORER_VERSION})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

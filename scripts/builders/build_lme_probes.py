#!/usr/bin/env python3
"""Build LongMemEval-KU current-value reader probes."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LME_DIR = ROOT / "outputs"

SYSTEM_PROMPT = (
    "You are reading a conversation that may be partial or compressed. "
    "Answer ONLY using the provided context. Reply with a single JSON object "
    "and no other text. Do not add markdown fences."
)


def load_jsonl(p: Path) -> list[dict]:
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def speaker_tag(sp: str) -> str:
    sp = (sp or "").lower()
    if sp == "user":
        return "USER"
    if sp == "assistant":
        return "ASSISTANT"
    return "SYSTEM"


def _cap_text(text: str, cap: int) -> str:
    if cap <= 0 or len(text) <= cap:
        return text
    return text[:cap].rstrip() + " [...]"


def render(turns: list[dict], indices: list[int], per_turn_char_cap: int = 0) -> str:
    indices = sorted(set(i for i in indices if 0 <= i < len(turns)))
    parts = []
    for i in indices:
        t = turns[i]
        body = _cap_text(t.get("text", ""), per_turn_char_cap)
        parts.append(f"[T{t.get('turn_id', i)}] {speaker_tag(t.get('speaker'))}: {body}")
    return "\n".join(parts)


def sel_full(n: int) -> list[int]:
    return list(range(n))


def sel_recency(n: int, k: int) -> list[int]:
    return list(range(max(0, n - k), n))


def sel_random_seed42(dialogue_id: str, n: int, k: int) -> list[int]:
    h = hashlib.sha256(f"42|{dialogue_id}".encode("utf-8")).hexdigest()
    seed = int(h[:16], 16) % (2**32)
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), min(k, n)))


def sel_first_n(n: int, k: int) -> list[int]:
    return list(range(0, min(k, n)))


def sel_uniform_stride(n: int, k: int) -> list[int]:
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    step = n / k
    return sorted({min(n - 1, int(round(i * step))) for i in range(k)})


METHODS = ["full_context", "recency", "random_seed42", "first_n", "uniform_stride"]


def compress(method: str, dialogue_id: str, turns: list[dict], k: int,
             per_turn_char_cap: int = 0) -> str:
    n = len(turns)
    if method == "full_context":
        idx = sel_full(n)
    elif method == "recency":
        idx = sel_recency(n, k)
    elif method == "random_seed42":
        idx = sel_random_seed42(dialogue_id, n, k)
    elif method == "first_n":
        idx = sel_first_n(n, k)
    elif method == "uniform_stride":
        idx = sel_uniform_stride(n, k)
    else:
        raise ValueError(f"unknown method: {method}")
    return render(turns, idx, per_turn_char_cap=per_turn_char_cap)


def make_p3_user(context: str, question: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question} "
        f"Reply with one short value reflecting the user's CURRENT state "
        f"after the most recent change visible in the context.\n\n"
        "Required JSON schema:\n"
        "{\"value\": <string>, "
        "\"support\": <verbatim span from context, <= 30 words>, "
        "\"abstain\": <true if you cannot tell, else false>}\n"
        "Reply with only the JSON object."
    )


def template_sha(question: str) -> str:
    s = f"v26-lme-ku::P3::{question}"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def build_one(record: dict, method: str, ctx: str, ratio: float, k: int) -> dict:
    turns = record["turns"]
    n_turns = len(turns)
    probe = record["probe"]
    transition = record.get("transition") or {}
    question = (probe.get("question") or "").strip()
    gold = (probe.get("gold_answer") or "").strip()
    support_ids = transition.get("support_turn_ids") or []
    last_support = max(support_ids) if support_ids else -1
    return {
        "dialogue_id": record["dialogue_id"],
        "dataset": "lme_ku",
        "slot": transition.get("slot") or "",
        "slot_human": transition.get("slot") or "",
        "old_value": transition.get("old_state") or "",
        "new_value": transition.get("new_state") or gold,
        "method": method,
        "compressed_text_chars": len(ctx),
        "meta": {
            "n_turns": n_turns,
            "k_turns": k,
            "ratio": ratio,
            "last_support_turn_id": last_support,
            "support_turn_ids": support_ids,
            "source": record.get("source"),
            "answer_type": probe.get("answer_type"),
            "operation": transition.get("operation"),
        },
        "prompt_system": SYSTEM_PROMPT,
        "probe_type": "P3",
        "gold": gold,
        "prompt_user": make_p3_user(ctx, question),
        "compressed_text_chars_used": len(ctx),
        "compressed_text_used": ctx,
        "template_sha16": template_sha(question),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=72)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--src", type=str,
                    default=str(LME_DIR / "fmts_lme_ku.jsonl"))
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--per-turn-char-cap", type=int, default=200,
                    help="cap each turn's body text to this many chars; "
                         "0 disables. Required for LME because mean session "
                         "length exceeds 7k input tokens on a 24GB GPU. "
                         "Disclose this in the paper.")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"[err] missing input: {src}", file=sys.stderr)
        return 1

    recs = load_jsonl(src)
    rng = random.Random(args.seed)
    rng.shuffle(recs)
    pilot = recs[: args.n]

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = LME_DIR / (
            f"probes_lme_ku_s{args.seed}_r{int(args.ratio * 100):02d}"
            f"_n{args.n}.jsonl"
        )

    n_rows = 0
    method_counts: dict[str, int] = {}
    with out_path.open("w", encoding="utf-8") as f:
        for rec in pilot:
            n_turns = len(rec["turns"])
            k = max(1, round(args.ratio * n_turns))
            for method in args.methods:
                if method == "full_context":
                    k_eff = n_turns
                else:
                    k_eff = k
                ctx = compress(method, rec["dialogue_id"], rec["turns"], k_eff,
                               per_turn_char_cap=args.per_turn_char_cap)
                row = build_one(rec, method, ctx, args.ratio, k_eff)
                row["meta"]["per_turn_char_cap"] = args.per_turn_char_cap
                if not row["gold"]:
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_rows += 1
                method_counts[method] = method_counts.get(method, 0) + 1

    try:
        rel = out_path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        rel = str(out_path)
    print(f"[wrote] {rel}", file=sys.stderr)
    print(f"  rows={n_rows}  pilot_dialogues={len(pilot)}/{len(recs)}",
          file=sys.stderr)
    for m in sorted(method_counts):
        print(f"  method {m:18s}  n={method_counts[m]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

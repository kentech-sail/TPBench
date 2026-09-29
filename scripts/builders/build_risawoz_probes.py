#!/usr/bin/env python3
"""Build initial-goal and current-value probes from RiSAWOZ test dialogues."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path


SYSTEM_PROMPT_ZH = (
    "你正在阅读一段可能被部分压缩或截取的对话。"
    "只使用提供的上下文信息回答。"
    "你的输出必须是一个有效的 JSON 对象,不要包含其他文本或代码块。"
)


def load_risawoz(path: Path) -> list[dict]:
    """Load RiSAWOZ dialogues from a JSON file (or directory of JSON)."""
    if path.is_dir():
        rows = []
        for p in sorted(path.glob("*.json")):
            rows.extend(load_risawoz(p))
        return rows
    txt = path.read_text(encoding="utf-8")
    obj = json.loads(txt)
    if isinstance(obj, dict):
        # Some releases ship {"dial_id_1": {...}, "dial_id_2": {...}}
        if all(isinstance(v, dict) and ("log" in v or "dialogue" in v) for v in obj.values()):
            return [{"dial_id": k, **v} for k, v in obj.items()]
        # Or {"dialogues": [...]}
        for k in ("dialogues", "data", "instances"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        raise ValueError(f"Unrecognized RiSAWOZ JSON object structure: {list(obj.keys())[:5]}")
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unrecognized RiSAWOZ JSON top-level type: {type(obj)}")


def flatten_belief(belief) -> dict[str, str]:
    """Return a flat {slot: value} belief dict across RiSAWOZ variants."""
    if not belief:
        return {}
    if isinstance(belief, list):
        flat = {}
        for item in belief:
            if isinstance(item, dict) and "slot" in item:
                flat[str(item["slot"])] = str(item.get("value", ""))
        return flat
    if not isinstance(belief, dict):
        return {}
    slot_values = belief.get("inform slot-values")
    if isinstance(slot_values, dict):
        return {
            str(k): str(v)
            for k, v in slot_values.items()
            if isinstance(v, (str, int, float)) and str(v).strip()
        }
    return {
        str(k): str(v)
        for k, v in belief.items()
        if isinstance(v, (str, int, float)) and str(v).strip()
    }


def normalize_dialogue(rec: dict) -> dict | None:
    """Return {dialogue_id, turns: [{turn_id, speaker, text, belief}], ...}
    or None if the record is malformed."""
    dial_id = (
        rec.get("dial_id")
        or rec.get("dialogue_id")
        or rec.get("id")
        or rec.get("uuid")
    )
    log = rec.get("log") or rec.get("dialogue") or rec.get("turns")
    if not log or not dial_id:
        return None

    turns = []
    for i, t in enumerate(log):
        # GEM/RiSAWOZ stores a user/system exchange in one record.
        if "user_utterance" in t or "system_utterance" in t:
            belief = flatten_belief(t.get("belief_state") or t.get("metadata") or {})
            user_text = (t.get("user_utterance") or "").strip()
            if user_text:
                turns.append({
                    "turn_id": len(turns),
                    "speaker": "user",
                    "text": user_text,
                    "belief": belief,
                })
            system_text = (t.get("system_utterance") or "").strip()
            if system_text:
                turns.append({
                    "turn_id": len(turns),
                    "speaker": "system",
                    "text": system_text,
                    "belief": belief,
                })
            continue

        sp_raw = (t.get("speaker") or t.get("role") or "").strip()
        sp_lc = sp_raw.lower()
        if sp_lc in ("user", "用户", "客户", "顾客") or "用户" in sp_raw:
            speaker = "user"
        elif sp_lc in ("system", "agent", "客服", "assistant") or "客服" in sp_raw:
            speaker = "system"
        else:
            # fallback: alternate user/system starting from user
            speaker = "user" if i % 2 == 0 else "system"

        text = (
            t.get("utterance")
            or t.get("text")
            or t.get("content")
            or t.get("user_utterance")
            or t.get("system_utterance")
            or ""
        ).strip()
        belief = flatten_belief(t.get("belief_state") or t.get("metadata") or {})
        turns.append({
            "turn_id": len(turns),
            "speaker": speaker,
            "text": text,
            "belief": belief,
        })

    return {"dialogue_id": str(dial_id), "turns": turns}


def find_transitions(turns: list[dict]) -> list[dict]:
    """Identify replace-value transitions: slot whose belief value changes."""
    transitions = []
    seen: dict[str, str] = {}
    for t in turns:
        belief = flatten_belief(t.get("belief") or {})
        if not isinstance(belief, dict):
            continue
        for slot, val in belief.items():
            if not isinstance(val, str) or not val.strip():
                continue
            prev = seen.get(slot)
            if prev is not None and prev != val:
                transitions.append({
                    "slot": slot,
                    "old_value": prev,
                    "new_value": val,
                    "support_turn_ids": [t["turn_id"]],
                    "transition_turn_id": t["turn_id"],
                })
            seen[slot] = val
    return transitions


def speaker_tag(sp: str) -> str:
    return "USER" if sp == "user" else "SYSTEM"


def _cap(text: str, cap: int) -> str:
    if cap <= 0 or len(text) <= cap:
        return text
    return text[:cap].rstrip() + " [...]"


def render(turns: list[dict], indices: list[int], cap: int = 0) -> str:
    indices = sorted(set(i for i in indices if 0 <= i < len(turns)))
    parts = []
    for i in indices:
        t = turns[i]
        parts.append(f"[T{t['turn_id']}] {speaker_tag(t['speaker'])}: {_cap(t['text'], cap)}")
    return "\n".join(parts)


def sel_full(n): return list(range(n))
def sel_recency(n, k): return list(range(max(0, n - k), n))
def sel_first_n(n, k): return list(range(0, min(k, n)))

def sel_random(dial_id, n, k):
    h = hashlib.sha256(f"42|{dial_id}".encode()).hexdigest()
    seed = int(h[:16], 16) % (2**32)
    rng = random.Random(seed)
    return sorted(rng.sample(range(n), min(k, n)))

def sel_uniform(n, k):
    if n == 0 or k <= 0:
        return []
    if k >= n:
        return list(range(n))
    step = n / k
    return sorted({min(n - 1, int(round(i * step))) for i in range(k)})


METHODS = ["full_context", "recency", "random_seed42", "first_n", "uniform_stride"]


def compress(method, dial_id, turns, k, cap=0):
    n = len(turns)
    if method == "full_context": idx = sel_full(n)
    elif method == "recency":    idx = sel_recency(n, k)
    elif method == "random_seed42": idx = sel_random(dial_id, n, k)
    elif method == "first_n":    idx = sel_first_n(n, k)
    elif method == "uniform_stride": idx = sel_uniform(n, k)
    else: raise ValueError(method)
    return render(turns, idx, cap=cap)


def make_p3_user_zh(ctx: str, slot: str) -> str:
    return (
        f"上下文:\n{ctx}\n\n"
        f"问题:槽位 \"{slot}\" 的当前值是什么?"
        f"请返回用户在上下文中最近一次提及或确认的取值。\n\n"
        "请用以下 JSON 格式回答(不要添加任何代码块标记):\n"
        "{\"value\": <字符串>, "
        "\"support\": <上下文中不超过30字的原文片段>, "
        "\"abstain\": <如果无法判断则为 true,否则为 false>}\n"
        "只返回 JSON 对象。"
    )


def make_p1_user_zh(ctx: str) -> str:
    return (
        f"上下文:\n{ctx}\n\n"
        f"问题:用户在这段对话开始时表达的高层意图(目标)是什么?"
        f"请用一句话概括用户最初提出的请求。\n\n"
        "请用以下 JSON 格式回答(不要添加任何代码块标记):\n"
        "{\"value\": <一句话描述用户的高层目标>, "
        "\"support\": <上下文中不超过30字的原文片段>, "
        "\"abstain\": <如果无法判断则为 true,否则为 false>}\n"
        "只返回 JSON 对象。"
    )


def template_sha(text: str) -> str:
    return hashlib.sha256(f"v26-risawoz::{text}".encode()).hexdigest()[:16]


def build_p3_row(rec, transition, method, k_eff, ratio, ctx, cap):
    return {
        "dialogue_id": rec["dialogue_id"],
        "dataset": "risawoz",
        "language": "zh",
        "slot": transition["slot"],
        "old_value": transition["old_value"],
        "new_value": transition["new_value"],
        "method": method,
        "compressed_text_chars": len(ctx),
        "meta": {
            "n_turns": len(rec["turns"]),
            "k_turns": k_eff,
            "ratio": ratio,
            "last_support_turn_id": max(transition["support_turn_ids"]),
            "support_turn_ids": transition["support_turn_ids"],
            "per_turn_char_cap": cap,
        },
        "prompt_system": SYSTEM_PROMPT_ZH,
        "probe_type": "P3",
        "gold": transition["new_value"],
        "prompt_user": make_p3_user_zh(ctx, transition["slot"]),
        "compressed_text_chars_used": len(ctx),
        "compressed_text_used": ctx,
        "template_sha16": template_sha(f"P3::{transition['slot']}"),
    }


def build_p1_row(rec, method, k_eff, ratio, ctx, cap):
    # Gold = first user utterance text
    user_turns = [t for t in rec["turns"] if t["speaker"] == "user"]
    gold = user_turns[0]["text"] if user_turns else ""
    return {
        "dialogue_id": rec["dialogue_id"],
        "dataset": "risawoz",
        "language": "zh",
        "method": method,
        "compressed_text_chars": len(ctx),
        "meta": {
            "n_turns": len(rec["turns"]),
            "k_turns": k_eff,
            "ratio": ratio,
            "per_turn_char_cap": cap,
        },
        "prompt_system": SYSTEM_PROMPT_ZH,
        "probe_type": "P1",
        "gold": gold,
        "prompt_user": make_p1_user_zh(ctx),
        "compressed_text_chars_used": len(ctx),
        "compressed_text_used": ctx,
        "template_sha16": template_sha("P1"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path,
                    help="RiSAWOZ JSON file or directory")
    ap.add_argument("--ratio", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n", type=int, default=200,
                    help="Number of dialogues to sample (after filtering "
                         "for at least one transition).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe-type", choices=["P1", "P3", "both"], default="both")
    ap.add_argument("--per-turn-char-cap", type=int, default=200,
                    help="Cap each turn's body text in chars; 0 disables. "
                         "Chinese characters count as one each.")
    args = ap.parse_args()

    raw = load_risawoz(args.src)
    print(f"[info] loaded {len(raw)} raw RiSAWOZ records from {args.src}",
          file=sys.stderr)

    # Normalize and filter to dialogues with >= 1 transition.
    candidates = []
    for rec_raw in raw:
        rec = normalize_dialogue(rec_raw)
        if rec is None or len(rec["turns"]) < 6:
            continue
        transitions = find_transitions(rec["turns"])
        # Mid-position filter: transition must be in turns [3, n-2]
        n = len(rec["turns"])
        valid = [tr for tr in transitions if 3 <= tr["transition_turn_id"] <= n - 2]
        if not valid:
            continue
        # Pick the latest transition (last support turn) per dialogue
        rec["transition"] = max(valid, key=lambda t: t["transition_turn_id"])
        candidates.append(rec)
    print(f"[info] {len(candidates)} dialogues with >=1 mid-position transition",
          file=sys.stderr)

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    pilot = candidates[: args.n]

    n_rows = 0
    method_counts: dict[str, int] = {}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in pilot:
            n_turns = len(rec["turns"])
            k = max(1, round(args.ratio * n_turns))
            for method in METHODS:
                k_eff = n_turns if method == "full_context" else k
                ctx = compress(method, rec["dialogue_id"], rec["turns"], k_eff,
                               cap=args.per_turn_char_cap)
                if args.probe_type in ("P3", "both"):
                    row = build_p3_row(rec, rec["transition"], method, k_eff,
                                       args.ratio, ctx, args.per_turn_char_cap)
                    if row["gold"]:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_rows += 1
                        method_counts[f"P3::{method}"] = method_counts.get(f"P3::{method}", 0) + 1
                if args.probe_type in ("P1", "both"):
                    row = build_p1_row(rec, method, k_eff, args.ratio, ctx,
                                       args.per_turn_char_cap)
                    if row["gold"]:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        n_rows += 1
                        method_counts[f"P1::{method}"] = method_counts.get(f"P1::{method}", 0) + 1

    print(f"[wrote] {args.out}", file=sys.stderr)
    print(f"  rows={n_rows}  pilot_dialogues={len(pilot)}/{len(candidates)}",
          file=sys.stderr)
    for k_, v_ in sorted(method_counts.items()):
        print(f"  {k_:30s}  n={v_}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

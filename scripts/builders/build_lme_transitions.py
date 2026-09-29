"""Convert LongMemEval knowledge-update questions to dialogue transitions."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "data/sources/longmemeval_knowledge_updates.json"
DEFAULT_OUT = ROOT / "outputs/fmts_lme_ku.jsonl"

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _slot_label_from_question(question: str, max_words: int = 5) -> str:
    """Best-effort short noun phrase label for diagnostics. Fallback handled upstream."""
    q = question.strip().rstrip("?").lower()
    for prefix in (
        "what is my ",
        "what's my ",
        "what was my ",
        "what is the ",
        "what's the ",
        "what was the ",
        "what ",
    ):
        if q.startswith(prefix):
            q = q[len(prefix):]
            break
    tokens = _WORD_RE.findall(q)
    if not tokens:
        return ""
    return " ".join(tokens[:max_words])


def flatten_sessions(
    sessions: List[List[Dict[str, Any]]],
    session_ids: List[str],
    session_dates: List[str],
) -> Tuple[List[Dict[str, Any]], List[List[int]]]:
    """Produce flat turn list plus a session-to-flat-id index.

    Returns (turns, session_turn_ids). Each SESSION_BREAK marker is a single
    system turn inserted BEFORE each session (including the first). Marker
    ids are consumed in the flat sequence but NOT added to session_turn_ids.
    """
    turns: List[Dict[str, Any]] = []
    session_turn_ids: List[List[int]] = []

    for sess_idx, sess in enumerate(sessions):
        sid = session_ids[sess_idx] if sess_idx < len(session_ids) else f"sess_{sess_idx}"
        date = session_dates[sess_idx] if sess_idx < len(session_dates) else ""
        marker_text = f"[SESSION_BREAK: date={date} sid={sid}]"
        turns.append(
            {
                "turn_id": len(turns),
                "speaker": "system",
                "text": marker_text,
            }
        )
        ids_this_session: List[int] = []
        for t in sess:
            role = str(t.get("role", "user")).lower()
            if role not in {"user", "assistant", "system"}:
                role = "user"
            flat_id = len(turns)
            turns.append(
                {
                    "turn_id": flat_id,
                    "speaker": role,
                    "text": str(t.get("content", "")),
                }
            )
            ids_this_session.append(flat_id)
        session_turn_ids.append(ids_this_session)
    return turns, session_turn_ids


def collect_support_and_boundaries(
    sessions: List[List[Dict[str, Any]]],
    session_ids: List[str],
    session_turn_ids: List[List[int]],
    answer_session_ids: List[str],
) -> Tuple[List[int], List[int]]:
    """Return (support_ids, boundary_ids) using only has_answer flags.

    support_ids: every turn with has_answer=True, sorted.
    boundary_ids: first has_answer turn inside each session that appears
                  in answer_session_ids. At most one per answer session.
    """
    ans_set = set(answer_session_ids or [])
    support: List[int] = []
    boundaries: List[int] = []
    for sess_idx, sess in enumerate(sessions):
        sid = session_ids[sess_idx] if sess_idx < len(session_ids) else None
        flat_ids = session_turn_ids[sess_idx]
        first_evidence_flat_id: Optional[int] = None
        for local_idx, t in enumerate(sess):
            if not t.get("has_answer"):
                continue
            if local_idx >= len(flat_ids):
                continue
            fid = flat_ids[local_idx]
            support.append(fid)
            if first_evidence_flat_id is None:
                first_evidence_flat_id = fid
        if sid in ans_set and first_evidence_flat_id is not None:
            boundaries.append(first_evidence_flat_id)
    return sorted(set(support)), sorted(set(boundaries))


def make_record(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    qid = raw.get("question_id") or "unknown"
    qtype = raw.get("question_type")
    if qtype != "knowledge-update":
        return None
    sessions = raw.get("haystack_sessions") or []
    session_ids = raw.get("haystack_session_ids") or []
    session_dates = raw.get("haystack_dates") or []
    answer_session_ids = raw.get("answer_session_ids") or []
    question = str(raw.get("question") or "").strip()
    gold_answer = str(raw.get("answer") or "").strip()
    if not sessions or not question:
        return None

    turns, session_turn_ids = flatten_sessions(sessions, session_ids, session_dates)
    support_ids, boundary_ids = collect_support_and_boundaries(
        sessions, session_ids, session_turn_ids, answer_session_ids
    )
    if not support_ids or not boundary_ids:
        return None
    # support must contain every boundary
    support_ids = sorted(set(support_ids) | set(boundary_ids))

    slot_label = _slot_label_from_question(question) or f"lme_ku_{qid}"
    n_turns = len(turns)
    boundary_position = round(boundary_ids[0] / max(1, n_turns - 1), 4)
    record = {
        "dialogue_id": f"lme_ku_{qid}",
        "source": "LongMemEval",
        "split": "knowledge_update",
        "turns": turns,
        "transition": {
            "operation": "REPLACE_VALUE",
            "slot": slot_label,
            "old_state": "",
            "new_state": gold_answer,
            "discarded_state": "",
            "boundary_turn_ids": boundary_ids,
            "old_state_support_ids": [],
            "new_state_support_ids": list(boundary_ids),
            "support_turn_ids": support_ids,
        },
        "probe": {
            "question": question,
            "gold_answer": gold_answer,
            "old_state_answer": "",
            "answer_type": "knowledge_update",
        },
        "counterfactual": {},
        "diagnostics": {
            "lexical_shortcut_risk": "low",
            "recency_shortcut_risk": "low",
            "boundary_position": boundary_position,
            "n_turns": n_turns,
            "n_sessions": len(sessions),
            "n_support": len(support_ids),
            "n_boundaries": len(boundary_ids),
            "question_type": qtype,
            "notes": "LongMemEval knowledge-update; sessions flattened with SESSION_BREAK markers.",
        },
    }
    return record


def validate_record(rec: Dict[str, Any]) -> None:
    n = len(rec["turns"])
    for tid in rec["transition"]["boundary_turn_ids"]:
        assert 0 <= tid < n, f"boundary {tid} out of range [0,{n})"
    for tid in rec["transition"]["support_turn_ids"]:
        assert 0 <= tid < n, f"support {tid} out of range [0,{n})"
    b = set(rec["transition"]["boundary_turn_ids"])
    s = set(rec["transition"]["support_turn_ids"])
    assert b.issubset(s), "boundary must be subset of support"
    # SESSION_BREAK markers must never be in support/boundary
    for tid in b | s:
        txt = rec["turns"][tid]["text"]
        assert not txt.startswith("[SESSION_BREAK:"), (
            f"SESSION_BREAK leaked into evidence at turn {tid}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_IN)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[ERROR] input missing: {args.input}", file=sys.stderr)
        return 1

    raw_all = json.loads(args.input.read_text(encoding="utf-8"))
    ku_records = [r for r in raw_all if r.get("question_type") == "knowledge-update"]
    print(f"[build] {len(raw_all)} total records, {len(ku_records)} knowledge-update")

    n_built = 0
    n_skipped_no_evidence = 0
    n_skipped_other = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f_out:
        for raw in ku_records:
            if args.limit and n_built >= args.limit:
                break
            rec = make_record(raw)
            if rec is None:
                if not (raw.get("answer_session_ids") and any(
                    any(t.get("has_answer") for t in sess)
                    for sess in raw.get("haystack_sessions") or []
                )):
                    n_skipped_no_evidence += 1
                else:
                    n_skipped_other += 1
                continue
            try:
                validate_record(rec)
            except AssertionError as e:
                print(f"[skip] {rec['dialogue_id']}: {e}")
                n_skipped_other += 1
                continue
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_built += 1

    print(f"[build] wrote {n_built} records to {args.output}")
    print(
        f"[build] skipped: no_evidence={n_skipped_no_evidence} other={n_skipped_other}"
    )
    if n_built:
        sizes = []
        with args.output.open("r", encoding="utf-8") as f_in:
            for line in f_in:
                r = json.loads(line)
                sizes.append(
                    (
                        len(r["turns"]),
                        len(r["transition"]["support_turn_ids"]),
                        len(r["transition"]["boundary_turn_ids"]),
                        r["diagnostics"]["boundary_position"],
                    )
                )
        import statistics
        turns_med = int(statistics.median(s[0] for s in sizes))
        supp_med = statistics.median(s[1] for s in sizes)
        bnd_med = statistics.median(s[2] for s in sizes)
        pos_med = statistics.median(s[3] for s in sizes)
        print(
            f"[build] medians: turns={turns_med} support={supp_med} "
            f"boundaries={bnd_med} boundary_pos={pos_med:.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

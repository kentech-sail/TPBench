#!/usr/bin/env python3
"""Score joint initial-goal and late-update answers (paper P3)."""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Share the phrase-boundary primitive with the slot-value scorer while keeping
# this composite scorer's lightweight text normalization.
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))
sys.path.insert(0, str(_SCRIPT_DIR.parent))
from normalize import normalized_phrase_occurs  # type: ignore  # noqa: E402

CONTEXT_RE = re.compile(r"^Context:\n(.*?)\n\nQuestion:", re.DOTALL)
WORD_RE = re.compile(r"[a-z0-9]+")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "to", "for", "of",
    "in", "on", "at", "by", "with", "and", "or", "i", "you", "we", "they",
    "this", "that", "these", "those", "my", "your", "our", "user", "please",
    "want", "need", "would", "like", "find", "looking", "search", "me",
    "agent", "should", "next", "do", "action", "what", "is", "it",
    "have", "has", "had", "will", "shall", "can", "could", "may", "might",
    "as", "from", "up", "out", "if", "so", "no", "not", "but",
}


def load_jsonl(p: Path) -> list[dict]:
    out = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def content_tokens(text: str) -> set[str]:
    if not text:
        return set()
    toks = WORD_RE.findall(text.lower())
    return {t for t in toks if t not in STOPWORDS and len(t) >= 2}


def norm_text(t: str) -> str:
    if not t:
        return ""
    return re.sub(r"\s+", " ", t.strip().lower())


def parse_reader_text(text: str) -> dict | None:
    if not text:
        return None
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    last = s.rfind("{")
    if last == -1:
        return None
    end = s.rfind("}")
    if end <= last:
        return None
    try:
        return json.loads(s[last:end + 1])
    except Exception:
        return None


def support_in_ctx(parsed: dict | None, ctx: str) -> int:
    if parsed is None:
        return 0
    raw = parsed.get("support")
    if isinstance(raw, list):
        sup = " ".join(str(x) for x in raw if x).strip()
    elif isinstance(raw, str):
        sup = raw.strip()
    else:
        sup = ""
    if not sup or not ctx:
        return 0
    return 1 if norm_text(sup) in norm_text(ctx) else 0


def _make_embed_encoder(model_name: str):
    """Lazy load HF model for paraphrase-robust embedding cosine."""
    from transformers import AutoTokenizer, AutoModel
    import torch
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl = mdl.to(device)

    def encode(texts: list[str]):
        if not texts:
            return torch.zeros((0, mdl.config.hidden_size))
        with torch.no_grad():
            enc = tok(texts, padding=True, truncation=True,
                      return_tensors="pt", max_length=128)
            enc = {k: v.to(device) for k, v in enc.items()}
            out = mdl(**enc)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            return emb.cpu()
    return encode


def token_f1(gold_toks: set[str], pred_toks: set[str]) -> float:
    if not gold_toks or not pred_toks:
        return 0.0
    common = gold_toks & pred_toks
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(gold_toks)
    return 2 * p * r / (p + r)


def score_late(probe_row: dict, parsed: dict | None) -> dict:
    abstain = parsed is None or parsed.get("abstain") is True
    pred = "" if abstain else str(parsed.get("value") or "")
    if not pred:
        abstain = True
    gold_story_label = probe_row.get("gold_story_label") or ""
    gold_value = probe_row.get("gold_value") or ""
    pred_norm = norm_text(pred)
    pred_toks = content_tokens(pred)

    # Initial-goal half: content-token overlap (>=1 because the target is short).
    story_label_toks = content_tokens(gold_story_label)
    story_label_overlap = len(story_label_toks & pred_toks)
    story_label_correct = 1 if story_label_overlap >= 1 and story_label_toks else 0

    # value half: bounded phrase occurrence (strict) or token overlap (loose)
    value_norm = norm_text(gold_value)
    value_toks = content_tokens(gold_value)
    value_overlap = len(value_toks & pred_toks)
    value_strict = int(normalized_phrase_occurs(value_norm, pred_norm))
    # loose: bounded occurrence OR overlap covers >= half of value tokens (min 1)
    value_loose = value_strict
    if not value_loose and value_toks and value_overlap >= max(1, len(value_toks) // 2):
        value_loose = 1
    value_correct = value_loose

    combined = 1 if (story_label_correct and value_correct) else 0
    combined_strict = 1 if (story_label_correct and value_strict) else 0

    # paraphrase-robust auxiliary metrics over the full composite gold/pred.
    composite_gold = (gold_story_label + " " + gold_value).strip()
    composite_toks = story_label_toks | value_toks
    f1 = token_f1(composite_toks, pred_toks)

    return {
        "abstain": abstain,
        "p1late_story_label_correct": story_label_correct,
        "p1late_value_correct": value_correct,
        "p1late_value_correct_strict": value_strict,
        "p1late_combined": combined,
        "p1late_combined_strict": combined_strict,
        "p1late_token_f1": f1,
        "story_label_overlap": story_label_overlap,
        "value_overlap": value_overlap,
        "story_label_token_count": len(story_label_toks),
        "value_token_count": len(value_toks),
        "p3_value_matcher": "shared_bounded_normalized_phrase_v1",
        "pred": pred,
        "composite_gold": composite_gold,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--reader-out", type=Path, required=True)
    ap.add_argument("--scored-out", type=Path, required=True)
    ap.add_argument("--aggregate-out", type=Path, required=True)
    ap.add_argument("--embed-sim", action="store_true",
                    help="paraphrase-robust embedding cosine via "
                         "sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--embed-model", type=str,
                    default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    probes = load_jsonl(args.probes)
    probe_idx: dict[tuple[str, str, str], dict] = {}
    for p in probes:
        if p.get("probe_type") != "P1_LATE":
            continue
        key = (p["dialogue_id"], p["method"], p["probe_type"])
        probe_idx[key] = p

    reader_rows = load_jsonl(args.reader_out)

    args.scored_out.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_out.parent.mkdir(parents=True, exist_ok=True)

    per_method = defaultdict(lambda: {
        "n": 0, "n_abstain": 0,
        "n_story_label": 0, "n_value": 0, "n_value_strict": 0,
        "n_combined": 0, "n_combined_strict": 0,
        "n_support_in_ctx": 0, "n_err": 0,
        "sum_token_f1": 0.0, "sum_embed_sim": 0.0, "n_embed": 0,
    })

    embed_encoder = None
    if args.embed_sim:
        embed_encoder = _make_embed_encoder(args.embed_model)

    scored_rows: list[dict] = []
    embed_pairs: list[tuple[int, str, str]] = []
    for r in reader_rows:
        if r.get("probe_type") != "P1_LATE":
            continue
        key = (r["dialogue_id"], r["method"], r["probe_type"])
        probe = probe_idx.get(key)
        if probe is None:
            continue

        err = r.get("error")
        parsed = parse_reader_text(r.get("reader_output_text") or "")
        ctx_match = CONTEXT_RE.search(probe.get("prompt_user") or "")
        ctx = ctx_match.group(1) if ctx_match else ""

        sc = score_late(probe, parsed)
        sup = support_in_ctx(parsed, ctx)

        row_out = {
            "dialogue_id": r["dialogue_id"],
            "dataset": r["dataset"],
            "method": r["method"],
            "probe_type": r["probe_type"],
            "meta": r.get("meta"),
            "error": err,
            "abstain": sc["abstain"],
            "p1late_story_label_correct": sc["p1late_story_label_correct"],
            "p1late_value_correct": sc["p1late_value_correct"],
            "p1late_value_correct_strict": sc["p1late_value_correct_strict"],
            "p1late_combined": sc["p1late_combined"],
            "p1late_combined_strict": sc["p1late_combined_strict"],
            "p1late_token_f1": sc["p1late_token_f1"],
            "p1late_embed_sim": None,
            "support_in_ctx": sup,
            "gold_story_label": probe.get("gold_story_label"),
            "gold_value": probe.get("gold_value"),
            "pred_value": sc.get("pred", ""),
            "p3_value_matcher": sc["p3_value_matcher"],
        }
        scored_rows.append(row_out)
        if (embed_encoder is not None and sc.get("composite_gold")
                and sc.get("pred") and not sc["abstain"]):
            embed_pairs.append((len(scored_rows) - 1,
                                sc["composite_gold"], sc["pred"]))

    if embed_encoder is not None and embed_pairs:
        idxs = [p[0] for p in embed_pairs]
        golds = [p[1] for p in embed_pairs]
        preds = [p[2] for p in embed_pairs]
        BATCH = 256
        sims_all = []
        for i in range(0, len(idxs), BATCH):
            g_emb = embed_encoder(golds[i:i + BATCH])
            p_emb = embed_encoder(preds[i:i + BATCH])
            sims = (g_emb * p_emb).sum(dim=-1).tolist()
            sims_all.extend(sims)
            print(f"  [embed] {i + len(sims):5d}/{len(idxs):5d}",
                  file=sys.stderr)
        for ridx, sim in zip(idxs, sims_all):
            scored_rows[ridx]["p1late_embed_sim"] = float(sim)

    with args.scored_out.open("w", encoding="utf-8") as fout:
        for row_out in scored_rows:
            method = row_out["method"]
            agg = per_method[method]
            agg["n"] += 1
            if row_out["error"]:
                agg["n_err"] += 1
            if row_out["abstain"]:
                agg["n_abstain"] += 1
            agg["n_story_label"] += row_out["p1late_story_label_correct"]
            agg["n_value"] += row_out["p1late_value_correct"]
            agg["n_value_strict"] += row_out["p1late_value_correct_strict"]
            agg["n_combined"] += row_out["p1late_combined"]
            agg["n_combined_strict"] += row_out["p1late_combined_strict"]
            agg["sum_token_f1"] += row_out["p1late_token_f1"]
            if row_out["p1late_embed_sim"] is not None:
                agg["sum_embed_sim"] += row_out["p1late_embed_sim"]
                agg["n_embed"] += 1
            agg["n_support_in_ctx"] += row_out["support_in_ctx"]
            fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")

    aggregate = {}
    for method, a in per_method.items():
        n = max(1, a["n"])
        n_emb = a["n_embed"]
        aggregate[method] = {
            "n": a["n"],
            "p1late_story_label_rate": a["n_story_label"] / n,
            "p1late_value_rate": a["n_value"] / n,
            "p1late_value_rate_strict": a["n_value_strict"] / n,
            "p1late_combined_rate": a["n_combined"] / n,
            "p1late_combined_rate_strict": a["n_combined_strict"] / n,
            "p1late_token_f1": a["sum_token_f1"] / n,
            "p1late_embed_sim": (a["sum_embed_sim"] / n_emb) if n_emb else None,
            "p1late_embed_n": n_emb,
            "abstain_rate": a["n_abstain"] / n,
            "support_in_ctx_rate": a["n_support_in_ctx"] / n,
            "err_rate": a["n_err"] / n,
        }
    with args.aggregate_out.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"[scorer_p1_late] scored rows: {sum(a['n'] for a in per_method.values())}",
          file=sys.stderr)
    for m in sorted(aggregate, key=lambda x: -aggregate[x]["p1late_combined_rate"]):
        a = aggregate[m]
        emb_str = (f"emb={a['p1late_embed_sim']:.3f}"
                   if a['p1late_embed_sim'] is not None else "emb=  -- ")
        print(f"  {m:30s} n={a['n']:4d}  combined={a['p1late_combined_rate']:.3f} "
              f"story_label={a['p1late_story_label_rate']:.3f} value={a['p1late_value_rate']:.3f} "
              f"f1={a['p1late_token_f1']:.3f} {emb_str} "
              f"abst={a['abstain_rate']:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Score the paper's P1 initial-goal probe.

The scorer reads probe and reader-output JSON Lines files, then writes per-row
scores and per-method aggregates. Strict accuracy requires at least two shared
content words between the answer and the initial-goal sentence; loose accuracy
requires one. It also reports token F1, abstention, answer support found in the
compressed text, and reader errors. An optional embedding score is provided as
a separate sensitivity diagnostic.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

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
    else:
        sup = (raw or "").strip() if isinstance(raw, str) else str(raw or "").strip()
    if not sup or not ctx:
        return 0
    norm_sup = re.sub(r"\s+", " ", sup).strip().lower()
    norm_ctx = re.sub(r"\s+", " ", ctx).strip().lower()
    return 1 if norm_sup in norm_ctx else 0


def _make_embed_encoder(model_name: str):
    """Lazy load HF model for paraphrase-robust embedding cosine.
    Returns a callable encode(list[str]) -> torch tensor (n, d), L2-normalized.
    """
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
    """SQuAD-style token-set F1. Robust to surface-form drift."""
    if not gold_toks or not pred_toks:
        return 0.0
    common = gold_toks & pred_toks
    if not common:
        return 0.0
    p = len(common) / len(pred_toks)
    r = len(common) / len(gold_toks)
    return 2 * p * r / (p + r)


def score_p1(probe_row: dict, parsed: dict | None) -> dict:
    abstain = parsed is None or parsed.get("abstain") is True
    pred = "" if abstain else str(parsed.get("value") or "")
    if not pred:
        abstain = True
    gold = probe_row.get("gold") or ""
    gold_toks = content_tokens(gold)
    pred_toks = content_tokens(pred)
    overlap = len(gold_toks & pred_toks)
    f1 = token_f1(gold_toks, pred_toks)
    return {
        "abstain": abstain,
        "p1_correct": 1 if overlap >= 2 else 0,
        "p1_correct_loose": 1 if overlap >= 1 else 0,
        "p1_token_f1": f1,
        "overlap_count": overlap,
        "gold_token_count": len(gold_toks),
        "pred": pred,
        "gold": gold,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True,
                    help="P1 probe jsonl (input that reader saw)")
    ap.add_argument("--reader-out", type=Path, required=True,
                    help="reader_worker_local output jsonl (concat of all shards ok)")
    ap.add_argument("--scored-out", type=Path, required=True)
    ap.add_argument("--aggregate-out", type=Path, required=True)
    ap.add_argument("--embed-sim", action="store_true",
                    help="compute paraphrase-robust embedding cosine via "
                         "sentence-transformers/all-MiniLM-L6-v2 (CPU OK).")
    ap.add_argument("--embed-model", type=str,
                    default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    probes = load_jsonl(args.probes)
    probe_idx: dict[tuple[str, str, str], dict] = {}
    for p in probes:
        if p.get("probe_type") != "P1":
            continue
        key = (p["dialogue_id"], p["method"], p["probe_type"])
        probe_idx[key] = p

    reader_rows = load_jsonl(args.reader_out)

    args.scored_out.parent.mkdir(parents=True, exist_ok=True)
    args.aggregate_out.parent.mkdir(parents=True, exist_ok=True)

    per_method = defaultdict(lambda: {
        "n": 0, "n_abstain": 0, "n_correct": 0, "n_correct_loose": 0,
        "n_support_in_ctx": 0, "n_err": 0,
        "sum_token_f1": 0.0, "sum_embed_sim": 0.0, "n_embed": 0,
    })

    embed_encoder = None
    if args.embed_sim:
        embed_encoder = _make_embed_encoder(args.embed_model)

    # Pass 1: score every row (token-level metrics computed inline; embed
    #         sim deferred to a single batch encode for speed).
    scored_rows: list[dict] = []
    embed_pairs: list[tuple[int, str, str]] = []  # (row_idx, gold, pred)
    for r in reader_rows:
        if r.get("probe_type") != "P1":
            continue
        key = (r["dialogue_id"], r["method"], r["probe_type"])
        probe = probe_idx.get(key)
        if probe is None:
            continue

        err = r.get("error")
        parsed = parse_reader_text(r.get("reader_output_text") or "")
        ctx_match = CONTEXT_RE.search(probe.get("prompt_user") or "")
        ctx = ctx_match.group(1) if ctx_match else ""

        sc = score_p1(probe, parsed)
        sup = support_in_ctx(parsed, ctx)
        gold_text = probe.get("gold") or ""
        pred_text = sc.get("pred", "")

        row_out = {
            "dialogue_id": r["dialogue_id"],
            "dataset": r["dataset"],
            "method": r["method"],
            "probe_type": r["probe_type"],
            "meta": r.get("meta"),
            "error": err,
            "p1_correct": sc["p1_correct"],
            "p1_correct_loose": sc["p1_correct_loose"],
            "p1_token_f1": sc["p1_token_f1"],
            "p1_embed_sim": None,
            "abstain": sc["abstain"],
            "support_in_ctx": sup,
            "gold": gold_text,
            "pred_value": pred_text,
        }
        scored_rows.append(row_out)
        if embed_encoder is not None and gold_text and pred_text and not sc["abstain"]:
            embed_pairs.append((len(scored_rows) - 1, gold_text, pred_text))

    # Optional embedding cosine: single batched encode for all (gold, pred).
    if embed_encoder is not None and embed_pairs:
        idxs = [p[0] for p in embed_pairs]
        golds = [p[1] for p in embed_pairs]
        preds = [p[2] for p in embed_pairs]
        BATCH = 256
        sims_all = []
        import torch
        for i in range(0, len(idxs), BATCH):
            g_emb = embed_encoder(golds[i:i + BATCH])
            p_emb = embed_encoder(preds[i:i + BATCH])
            sims = (g_emb * p_emb).sum(dim=-1).tolist()
            sims_all.extend(sims)
            print(f"  [embed] {i + len(sims):5d}/{len(idxs):5d}",
                  file=sys.stderr)
        for ridx, sim in zip(idxs, sims_all):
            scored_rows[ridx]["p1_embed_sim"] = float(sim)

    # Pass 2: aggregate per method + write scored jsonl.
    with args.scored_out.open("w", encoding="utf-8") as fout:
        for row_out in scored_rows:
            method = row_out["method"]
            agg = per_method[method]
            agg["n"] += 1
            if row_out["error"]:
                agg["n_err"] += 1
            if row_out["abstain"]:
                agg["n_abstain"] += 1
            agg["n_correct"] += row_out["p1_correct"]
            agg["n_correct_loose"] += row_out["p1_correct_loose"]
            agg["sum_token_f1"] += row_out["p1_token_f1"]
            if row_out["p1_embed_sim"] is not None:
                agg["sum_embed_sim"] += row_out["p1_embed_sim"]
                agg["n_embed"] += 1
            agg["n_support_in_ctx"] += row_out["support_in_ctx"]
            fout.write(json.dumps(row_out, ensure_ascii=False) + "\n")

    aggregate = {}
    for method, a in per_method.items():
        n = max(1, a["n"])
        n_emb = a["n_embed"]
        aggregate[method] = {
            "n": a["n"],
            "p1_em_strict": a["n_correct"] / n,
            "p1_em_loose": a["n_correct_loose"] / n,
            "p1_token_f1": a["sum_token_f1"] / n,
            "p1_embed_sim": (a["sum_embed_sim"] / n_emb) if n_emb else None,
            "p1_embed_n": n_emb,
            "abstain_rate": a["n_abstain"] / n,
            "support_in_ctx_rate": a["n_support_in_ctx"] / n,
            "err_rate": a["n_err"] / n,
        }
    with args.aggregate_out.open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    print(f"[scorer_p1] scored rows: {sum(a['n'] for a in per_method.values())}",
          file=sys.stderr)
    print(f"[scorer_p1] methods: {len(per_method)}", file=sys.stderr)
    for m in sorted(aggregate, key=lambda x: -aggregate[x]["p1_em_strict"]):
        a = aggregate[m]
        emb_str = f"emb={a['p1_embed_sim']:.3f}" if a['p1_embed_sim'] is not None else "emb=  -- "
        print(f"  {m:30s} n={a['n']:4d}  EM={a['p1_em_strict']:.3f} "
              f"loose={a['p1_em_loose']:.3f} f1={a['p1_token_f1']:.3f} "
              f"{emb_str} abst={a['abstain_rate']:.2f} "
              f"sup={a['support_in_ctx_rate']:.2f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate compressed contexts for the FMTS free-form experiment.

The program reads FMTS JSON Lines records and writes one compressed context per
method and example for free-form answer evaluation.

Input:  data/fmts/fmts_<split>.jsonl
Output: outputs/fmts/<tag>.json

Usage:
    python run_compression.py \
        --fmts data/fmts/fmts_indirect_seed42.jsonl \
        --ratio 0.3 --model gpt2 --seed 42 --tag indirect_r30

Turn-selection conditions:
  full_context, recency, random_seed42, first_n, uniform_stride,
  attention_h2o_cache, embedding_mmr_cache, llmlingua2_cache.

ChunkKV is intentionally not emitted here: it is a reader/KV-cache pruning
method, not a static compressed-context method. Run the ChunkKV reader harness
for that row.
"""
from __future__ import annotations

import argparse
import json
import sys
import random as _random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent / "core"))
from turn_types import Turn
from compression_methods import ContextCompressor, CompressedContext  # noqa: E402
from evaluation import Evaluator  # noqa: E402


TPBENCH_TURN_METHODS = [
    "full_context",
    "recency",
    "random_seed42",
    "first_n",
    "uniform_stride",
    "attention_h2o_cache",
    "embedding_mmr_cache",
    "llmlingua2_cache",
]

STATIC_UNSUPPORTED_METHODS = {
    "chunkkv",
    "chunkkv_r03_kvpress",
    "chunkkv_r10_kvpress",
}

IMPLEMENTED_METHODS = set(TPBENCH_TURN_METHODS)


def load_fmts(path: Path):
    recs = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        recs.append(json.loads(raw))
    return recs


def record_to_turns(rec):
    return [
        Turn(speaker=t["speaker"], text=t["text"])
        for t in rec["turns"]
    ]


def adv_fp_rate(kept, adv):
    if not adv:
        return 0.0
    s = set(kept)
    return sum(1 for a in adv if a in s) / len(adv)


def support_recall(kept, support_ids):
    support_ids = list(support_ids or [])
    if not support_ids:
        return 1.0
    s = set(kept)
    return sum(1 for i in support_ids if i in s) / len(support_ids)


def relabel_context(ctx: CompressedContext, method: str) -> CompressedContext:
    """Reuse a computed context while recording the requested method key."""
    return CompressedContext(
        method,
        list(ctx.kept_turn_indices),
        ctx.compression_ratio,
        ctx.text,
    )


def validate_methods(methods: list[str]) -> None:
    unsupported = [m for m in methods if m in STATIC_UNSUPPORTED_METHODS]
    if unsupported:
        joined = ", ".join(sorted(unsupported))
        raise SystemExit(
            f"Unsupported in run_compression.py: {joined}. "
            "ChunkKV is a reader-side KV-cache method; use the ChunkKV reader "
            "harness and merge its scored row separately."
        )
    unknown = [m for m in methods if m not in IMPLEMENTED_METHODS]
    if unknown:
        joined = ", ".join(sorted(unknown))
        known = ", ".join(sorted(IMPLEMENTED_METHODS | STATIC_UNSUPPORTED_METHODS))
        raise SystemExit(f"Unknown method(s): {joined}\nKnown methods: {known}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fmts", required=True, type=Path)
    p.add_argument("--model", default="gpt2")
    p.add_argument("--ratio", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tag", required=True)
    p.add_argument("--threshold", type=float, default=1.5)
    p.add_argument("--window", type=int, default=3)
    p.add_argument("--top_k_ratio", type=float, default=0.2)
    p.add_argument("--out_dir", type=Path,
                   default=ROOT / "outputs" / "fmts")
    p.add_argument("--only", nargs="*", default=[],
                   help="If set, run only the listed turn-selection methods")
    p.add_argument("--limit", type=int, default=0,
                   help="If >0, process only the first N records (smoke use)")
    args = p.parse_args()

    _random.seed(args.seed)
    np.random.seed(args.seed)

    records = load_fmts(args.fmts)
    if not records:
        print(f"[fmts] no records in {args.fmts}")
        return
    if args.limit and args.limit > 0:
        records = records[: args.limit]

    if args.only:
        active_methods = list(args.only)
    else:
        active_methods = list(TPBENCH_TURN_METHODS)
    validate_methods(active_methods)

    print(f"[fmts] loaded {len(records)} records from {args.fmts.name}")
    print(f"[fmts] methods: {', '.join(active_methods)}")
    attn_extractor = None
    if set(active_methods) & {"attention_h2o_cache", "embedding_mmr_cache"}:
        from boundary_detector import EpisodicBoundaryDetector, load_shared_model
        attn_extractor = EpisodicBoundaryDetector(
            model_name=args.model, shared_model=load_shared_model(args.model),
        )

    compressor = ContextCompressor(compression_ratio=args.ratio)
    evaluator = Evaluator(use_bertscore=False)

    per_method = {m: [] for m in active_methods}
    per_record_compressed = []

    for idx, rec in enumerate(records):
        if idx % 10 == 0:
            print(f"  processing {idx + 1}/{len(records)}")
        turns = record_to_turns(rec)
        full_text = " ".join(t.text for t in turns)
        qa_pairs = [
            {
                "question": rec["probe"]["question"],
                "answer": rec["probe"]["gold_answer"],
                "type": rec["probe"].get("answer_type", "transition_qa"),
                "requires_boundary": True,
            }
        ]
        support_ids = rec["transition"].get("support_turn_ids", [])
        gt_boundaries = rec.get("gt_boundaries") or rec["transition"].get("boundary_turn_ids") or []
        adv_positions = rec.get("adv_positions", [])

        try:
            attn = attn_extractor.get_real_attention_scores(turns) if attn_extractor else None
        except Exception:
            attn = [1.0 / max(1, len(turns))] * len(turns)
        try:
            emb = attn_extractor.get_turn_embeddings(turns) if attn_extractor else None
        except Exception:
            emb = None

        comp = {}
        if "full_context" in active_methods:
            comp["full_context"] = compressor.full_context(turns)
        if "recency" in active_methods:
            comp["recency"] = compressor.recency(turns)
        if "attention_h2o_cache" in active_methods:
            h2o_ctx = compressor.attention_based(turns, attn)
            comp["attention_h2o_cache"] = relabel_context(h2o_ctx, "attention_h2o_cache")
        if "embedding_mmr_cache" in active_methods:
            mmr_ctx = compressor.embedding_mmr(turns, embeddings=emb)
            comp["embedding_mmr_cache"] = relabel_context(mmr_ctx, "embedding_mmr_cache")
        if "llmlingua2_cache" in active_methods:
            llm_ctx = compressor.llmlingua2(turns)
            comp["llmlingua2_cache"] = relabel_context(llm_ctx, "llmlingua2_cache")
        if "random_seed42" in active_methods:
            _random.seed(args.seed)
            rc = compressor.random_compression(turns)
            comp["random_seed42"] = relabel_context(rc, "random_seed42")
        if "first_n" in active_methods:
            comp["first_n"] = compressor.first_n(turns)
        if "uniform_stride" in active_methods:
            comp["uniform_stride"] = compressor.uniform_stride(turns)

        compressed_list = [comp[m] for m in active_methods if m in comp]
        session_eval = evaluator.evaluate_all(
            compressed_list, full_text, gt_boundaries, qa_pairs=qa_pairs,
        )

        record_dump = {"dialogue_id": rec["dialogue_id"], "methods": {}}
        for cobj, r in zip(compressed_list, session_eval):
            entry = {
                "dialogue_id": rec["dialogue_id"],
                "method": r.method,
                "ratio": args.ratio,
                "seed": args.seed,
                "qa_accuracy_exact": r.qa_accuracy,
                "b_qa_exact": r.boundary_qa_accuracy,
                "boundary_recall": r.boundary_recall,
                "support_recall": support_recall(cobj.kept_turn_indices, support_ids),
                "adv_fp": adv_fp_rate(cobj.kept_turn_indices, adv_positions),
                "compression_ratio": r.compression_ratio,
                "token_ratio": r.token_ratio,
                "kept_turn_ids": cobj.kept_turn_indices,
                "compressed_text": cobj.text,
                "question": qa_pairs[0]["question"],
                "gold_answer": qa_pairs[0]["answer"],
                "old_state_answer": rec["probe"]["old_state_answer"],
                "transition": rec["transition"],
            }
            per_method[r.method].append(entry)
            record_dump["methods"][r.method] = {
                "kept_turn_ids": cobj.kept_turn_indices,
                "compressed_text": cobj.text,
                "compression_ratio": r.compression_ratio,
                "support_recall": entry["support_recall"],
                "boundary_recall": entry["boundary_recall"],
            }
        per_record_compressed.append(record_dump)

    def mean(xs, key):
        vs = [x[key] for x in xs if x.get(key) is not None]
        return float(np.mean(vs)) if vs else None

    summary = {}
    for m in active_methods:
        rows = per_method[m]
        if not rows:
            continue
        summary[m] = {
            "n": len(rows),
            "qa_exact": mean(rows, "qa_accuracy_exact"),
            "b_qa_exact": mean(rows, "b_qa_exact"),
            "boundary_recall": mean(rows, "boundary_recall"),
            "support_recall": mean(rows, "support_recall"),
            "adv_fp": mean(rows, "adv_fp"),
            "compression_ratio": mean(rows, "compression_ratio"),
            "token_ratio": mean(rows, "token_ratio"),
        }

    print("\n=== FMTS compression summary ===")
    print(f"{'method':<20} {'qa':>6} {'b_qa':>6} {'bRec':>6} {'suppR':>6} {'advFP':>6}")
    for m, s in summary.items():
        print(f"{m:<20} "
              f"{(s.get('qa_exact') or 0):>6.3f} "
              f"{(s.get('b_qa_exact') or 0):>6.3f} "
              f"{(s.get('boundary_recall') or 0):>6.3f} "
              f"{(s.get('support_recall') or 0):>6.3f} "
              f"{(s.get('adv_fp') or 0):>6.3f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.tag}.json"
    out.write_text(
        json.dumps({
            "experiment": "fmts_reference_pool",
            "config": vars(args),
            "summary": summary,
            "per_record_compressed": per_record_compressed,
            "raw": per_method,
        }, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()

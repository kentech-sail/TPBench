#!/usr/bin/env python3
"""Run TPBench probes with ChunkKV cache compression.

The program uses the same input and output schema as reader_worker_local.py and
applies the vendored ChunkKVPress during context prefill.

Input probes are expected to have method="full_context" (uncompressed dialogue
prompts). Output rows are written with method=<--method-label-override>
(e.g. "chunkkv_r03_kvpress") so they sit alongside the other compression
methods inside the existing aggregate JSONs.

The run is resume-safe: it skips task_uid values already present in --out. The
identifier includes the overridden method, so different budgets or chunk
lengths do not collide.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make the vendored chunkkv package importable regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from chunkkv_vendored import ChunkKVPress, SnapKVPress  # noqa: E402

JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def task_uid(row: dict, full_prompt: str, method_override: str) -> str:
    template_sha = row.get("template_sha16") or ""
    prompt_sha = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()[:16]
    base = "|".join([
        row.get("dialogue_id") or "",
        method_override,
        row.get("probe_type") or "",
        str(row.get("meta", {}).get("ratio") or ""),
        template_sha,
        prompt_sha,
    ])
    return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def load_done_uids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                u = row.get("task_uid")
                if u:
                    done.add(u)
            except Exception:
                continue
    return done


def extract_json(text: str) -> Optional[str]:
    if not text:
        return None
    candidates = JSON_OBJ_RE.findall(text)
    for cand in reversed(candidates):
        try:
            json.loads(cand)
            return cand
        except Exception:
            continue
    return None


def build_chat_prompt(tokenizer, system: str, user: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return f"{system}\n\n{user}"


def build_press(args):
    """Build a ChunkKVPress (with SnapKV scorer) or plain SnapKVPress."""
    if args.press == "chunkkv":
        scorer = SnapKVPress(
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
            kernel_size=args.kernel_size,
        )
        return ChunkKVPress(press=scorer, chunk_length=args.chunk_length)
    if args.press == "snapkv":
        return SnapKVPress(
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
            kernel_size=args.kernel_size,
        )
    if args.press == "none":
        return None
    raise ValueError(f"unknown --press {args.press!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True,
                    help="probes JSONL (rows with method=full_context expected)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-i", type=int, default=0)
    ap.add_argument("--shard-n", type=int, default=1)
    ap.add_argument("--model", type=str,
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=7168)
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--attn-impl", type=str, default="eager",
                    choices=["eager", "sdpa", "flash_attention_2"])
    # press configuration
    ap.add_argument("--press", type=str, default="chunkkv",
                    choices=["chunkkv", "snapkv", "none"])
    ap.add_argument("--compression-ratio", type=float, default=0.7,
                    help="kvpress drop ratio. 0.7 = 70%% drop = 30%% retain.")
    ap.add_argument("--chunk-length", type=int, default=20,
                    help="ChunkKV chunk size (paper default 20)")
    ap.add_argument("--window-size", type=int, default=64,
                    help="SnapKV observation window (paper default 64)")
    ap.add_argument("--kernel-size", type=int, default=5,
                    help="SnapKV pooling kernel (paper default 5)")
    ap.add_argument("--method-label-override", type=str, required=True,
                    help="output `method` field, e.g. chunkkv_r03_kvpress")
    ap.add_argument("--input-method-filter", type=str, default="full_context",
                    help="only process probes with this method field")
    ap.add_argument("--compression-mode", type=str, default="context_only",
                    choices=["context_only", "full_prompt"],
                    help=("context_only (default): two-stage prefill — compress "
                          "system+context cache first, then prefill question on "
                          "top uncompressed (kvpress pipeline parity, fair vs "
                          "turn-level selectors which also compress without "
                          "seeing the probe). full_prompt: `with press: generate` "
                          "where question shares the compression window."))
    ap.add_argument("--question-marker", type=str, default="\n\nQuestion:",
                    help="boundary string in prompt_user that splits context "
                         "from question for context_only mode")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16,
                 "float16": torch.float16,
                 "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"[info] model={args.model}  device={args.device}  dtype={args.dtype}",
          file=sys.stderr)
    print(f"[info] press={args.press}  drop_ratio={args.compression_ratio}  "
          f"chunk={args.chunk_length}  window={args.window_size}  "
          f"kernel={args.kernel_size}", file=sys.stderr)
    print(f"[info] method_label_override={args.method_label_override}",
          file=sys.stderr)
    print(f"[info] shard {args.shard_i}/{args.shard_n}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device if args.device != "cpu" else None,
        attn_implementation=args.attn_impl,
    )
    if args.device == "cpu":
        model = model.to("cpu")
    model.eval()

    press = build_press(args)
    if press is not None:
        print(f"[info] press={press!r}", file=sys.stderr)

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_uids = load_done_uids(out_path)
    print(f"[info] resume: {len(done_uids)} already done", file=sys.stderr)

    rows = []
    with args.probes.resolve().open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i % args.shard_n != args.shard_i:
                continue
            r = json.loads(line)
            if r.get("method") != args.input_method_filter:
                continue
            full_prompt = f"{r['prompt_system']}\n\n{r['prompt_user']}"
            r["__task_uid"] = task_uid(r, full_prompt, args.method_label_override)
            if r["__task_uid"] in done_uids:
                continue
            rows.append(r)
    print(f"[info] worker {args.shard_i}/{args.shard_n}: {len(rows)} tasks",
          file=sys.stderr)

    t_start = time.time()
    n_ok = 0
    n_err = 0
    n_split_fail = 0
    with out_path.open("a", encoding="utf-8") as f:
        for idx, r in enumerate(rows):
            split_ok = False
            try:
                prompt_text = build_chat_prompt(
                    tokenizer, r["prompt_system"], r["prompt_user"]
                )

                use_two_stage = (
                    press is not None
                    and args.compression_mode == "context_only"
                )

                if use_two_stage:
                    q_pos = prompt_text.rfind(args.question_marker)
                    if q_pos > 0:
                        ctx_text = prompt_text[:q_pos]
                        qst_text = prompt_text[q_pos:]
                        ctx_ids = tokenizer(
                            ctx_text, return_tensors="pt",
                            add_special_tokens=False,
                            truncation=True, max_length=args.max_input_tokens,
                        ).input_ids.to(model.device)
                        qst_ids = tokenizer(
                            qst_text, return_tensors="pt",
                            add_special_tokens=False,
                        ).input_ids.to(model.device)
                        if ctx_ids.shape[1] > args.window_size + 4:
                            split_ok = True

                with torch.no_grad():
                    if split_ok:
                        # Stage 1: prefill+compress system+context only.
                        with press(model):
                            ctx_out = model(
                                input_ids=ctx_ids,
                                use_cache=True,
                            )
                        past_kv = ctx_out.past_key_values
                        # Stage 2: append question, generate without press.
                        cache_len = past_kv.get_seq_length()
                        attn_mask = torch.ones(
                            (1, cache_len + qst_ids.shape[1]),
                            dtype=torch.long, device=model.device,
                        )
                        out = model.generate(
                            input_ids=qst_ids,
                            past_key_values=past_kv,
                            attention_mask=attn_mask,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            temperature=1.0,
                            top_p=1.0,
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=True,
                        )
                        prompt_input_len = qst_ids.shape[1]
                    else:
                        if press is not None and args.compression_mode == "context_only":
                            n_split_fail += 1
                        # Fallback: full-prompt mode (no split or no press).
                        inputs = tokenizer(
                            prompt_text, return_tensors="pt",
                            truncation=True, max_length=args.max_input_tokens,
                        )
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        if press is not None:
                            with press(model):
                                out = model.generate(
                                    **inputs,
                                    max_new_tokens=args.max_new_tokens,
                                    do_sample=False,
                                    temperature=1.0, top_p=1.0,
                                    pad_token_id=tokenizer.pad_token_id,
                                )
                        else:
                            out = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=False,
                                temperature=1.0, top_p=1.0,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        prompt_input_len = inputs["input_ids"].shape[1]
                gen_ids = out[0, prompt_input_len:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                json_obj = extract_json(gen_text) or ""
                err = None if json_obj else "no-json"
                raw = json_obj if json_obj else gen_text.strip()
            except Exception as exc:
                raw = ""
                err = f"exception:{type(exc).__name__}:{exc}"
                split_ok = False

            row_out = {
                "task_uid": r["__task_uid"],
                "dialogue_id": r["dialogue_id"],
                "dataset": r["dataset"],
                "method": args.method_label_override,
                "probe_type": r["probe_type"],
                "meta": {
                    **(r.get("meta") or {}),
                    "press": args.press,
                    "compression_ratio_drop": args.compression_ratio,
                    "chunk_length": args.chunk_length,
                    "window_size": args.window_size,
                    "kernel_size": args.kernel_size,
                    "source_method": r.get("method"),
                    "compression_mode": (
                        "context_only" if split_ok else
                        ("full_prompt" if press is not None else "no_press")
                    ),
                },
                "reader_output_text": raw,
                "error": err,
                "model": args.model,
            }
            f.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            f.flush()
            if err is None and raw:
                n_ok += 1
            else:
                n_err += 1
            if (idx + 1) % args.progress_every == 0 or idx == len(rows) - 1:
                elapsed = time.time() - t_start
                rate = (idx + 1) / max(0.1, elapsed)
                eta = (len(rows) - idx - 1) / max(0.001, rate)
                print(f"  [{idx+1:5d}/{len(rows):5d}]  ok={n_ok}  err={n_err}  "
                      f"rate={rate:.2f}/s  eta={eta/60:.1f}min", file=sys.stderr)

    print(f"\n[done] worker {args.shard_i}/{args.shard_n}: ok={n_ok} err={n_err} "
          f"split_fallback={n_split_fail}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

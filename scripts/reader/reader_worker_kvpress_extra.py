#!/usr/bin/env python3
"""Run TPBench probes with KV-cache methods provided by kvpress.

Supported methods are:

  * SnapKVPress       — Li et al. NeurIPS 2024 (pooled attention, observation window)
  * PyramidKVPress    — Cai et al. NeurIPS 2024 (pyramid budget across layers)
  * StreamingLLMPress — Xiao et al. ICLR 2024 (sink tokens + recent window)
  * ExpectedAttentionPress (optional, --press expected_attention)

The output schema matches reader_worker_chunkkv.py, so result files can be
scored with:
  scorer_p1.py / scorer_p3.py / scorer_p1_late.py
and then included in the aggregate summaries.

Input probes must carry method=full_context (uncompressed dialogue prompts).
Output method field is set to --method-label-override (e.g. snapkv_r03,
pyramidkv_r03, streamingllm_r03).

The run is resume-safe and skips task_uid values already present in --out.

Required: pip install kvpress  (at least version with PyramidKV / Streaming).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from kvpress import (
        SnapKVPress,
        StreamingLLMPress,
    )
except ImportError as exc:
    print(f"[fatal] kvpress import failed: {exc}", file=sys.stderr)
    print("[fatal] install with: pip install kvpress", file=sys.stderr)
    sys.exit(2)

# PyramidKV and ExpectedAttention may live in slightly different locations
# across kvpress versions; import lazily so SnapKV/Streaming still work if
# one is missing.
try:
    from kvpress import PyramidKVPress  # newer API
except ImportError:
    try:
        from kvpress.presses.pyramidkv_press import PyramidKVPress
    except ImportError:
        PyramidKVPress = None

try:
    from kvpress import ExpectedAttentionPress
except ImportError:
    try:
        from kvpress.presses.expected_attention_press import ExpectedAttentionPress
    except ImportError:
        ExpectedAttentionPress = None


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
    p = args.press
    if p == "snapkv":
        return SnapKVPress(
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
            kernel_size=args.kernel_size,
        )
    if p == "pyramidkv":
        if PyramidKVPress is None:
            raise SystemExit(
                "PyramidKVPress not available in this kvpress install. "
                "Upgrade kvpress (pip install -U kvpress)."
            )
        return PyramidKVPress(
            compression_ratio=args.compression_ratio,
            window_size=args.window_size,
            kernel_size=args.kernel_size,
        )
    if p == "streamingllm":
        return StreamingLLMPress(
            compression_ratio=args.compression_ratio,
            n_sink=args.n_sink,
        )
    if p == "expected_attention":
        if ExpectedAttentionPress is None:
            raise SystemExit("ExpectedAttentionPress not available.")
        return ExpectedAttentionPress(
            compression_ratio=args.compression_ratio,
        )
    raise ValueError(f"unknown --press {p!r}")


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
    ap.add_argument("--attn-impl", type=str, default="sdpa",
                    choices=["eager", "sdpa", "flash_attention_2"])
    # press configuration
    ap.add_argument("--press", type=str, required=True,
                    choices=["snapkv", "pyramidkv", "streamingllm",
                             "expected_attention"])
    ap.add_argument("--compression-ratio", type=float, default=0.7,
                    help="kvpress drop ratio. 0.7 = 70%% drop = 30%% retain. "
                         "Use 0.9 for r=0.10 retain.")
    ap.add_argument("--window-size", type=int, default=64,
                    help="SnapKV / PyramidKV observation window")
    ap.add_argument("--kernel-size", type=int, default=5,
                    help="SnapKV / PyramidKV pooling kernel")
    ap.add_argument("--n-sink", type=int, default=4,
                    help="StreamingLLM sink tokens to retain at start")
    ap.add_argument("--method-label-override", type=str, required=True,
                    help="output `method` field, e.g. snapkv_r03_kvpress")
    ap.add_argument("--input-method-filter", type=str, default="full_context")
    ap.add_argument("--compression-mode", type=str, default="full_prompt",
                    choices=["context_only", "full_prompt"],
                    help=("full_prompt (default): one-pass with-press generate. "
                          "context_only: two-stage prefill (compress system+"
                          "context cache, then prefill question on top "
                          "uncompressed). NOTE: context_only is broken under "
                          "transformers 5.x because model.generate(past_key_values=...) "
                          "fails to compute cache_position; the headline ChunkKV "
                          "reader uses full_prompt for the same reason."))
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
          f"window={args.window_size}  kernel={args.kernel_size}  "
          f"n_sink={args.n_sink}", file=sys.stderr)
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
    err_log_path = out_path.with_suffix(".errors.log")
    with out_path.open("a", encoding="utf-8") as f, \
         err_log_path.open("a", encoding="utf-8") as ef:
        for idx, r in enumerate(rows):
            split_ok = False
            split_attempted = False
            stage = "init"
            try:
                prompt_text = build_chat_prompt(
                    tokenizer, r["prompt_system"], r["prompt_user"]
                )

                use_two_stage = (args.compression_mode == "context_only")

                if use_two_stage:
                    split_attempted = True
                    stage = "split"
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
                        if ctx_ids.shape[1] > max(args.window_size + 4,
                                                  args.n_sink + 4):
                            split_ok = True

                with torch.no_grad():
                    if split_ok:
                        stage = "ctx_prefill"
                        with press(model):
                            ctx_out = model(
                                input_ids=ctx_ids,
                                use_cache=True,
                            )
                        past_kv = ctx_out.past_key_values
                        cache_len = past_kv.get_seq_length()
                        attn_mask = torch.ones(
                            (1, cache_len + qst_ids.shape[1]),
                            dtype=torch.long, device=model.device,
                        )
                        stage = "qst_generate"
                        out = model.generate(
                            input_ids=qst_ids,
                            past_key_values=past_kv,
                            attention_mask=attn_mask,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                            temperature=1.0, top_p=1.0,
                            pad_token_id=tokenizer.pad_token_id,
                            use_cache=True,
                        )
                        prompt_input_len = qst_ids.shape[1]
                    else:
                        if args.compression_mode == "context_only":
                            n_split_fail += 1
                        stage = "full_prompt_generate"
                        inputs = tokenizer(
                            prompt_text, return_tensors="pt",
                            truncation=True, max_length=args.max_input_tokens,
                        )
                        inputs = {k: v.to(model.device) for k, v in inputs.items()}
                        with press(model):
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
                err = f"exception:{type(exc).__name__}:{exc}@stage={stage}"
                ef.write(
                    f"=== uid={r.get('__task_uid')} stage={stage} "
                    f"split_attempted={split_attempted} split_ok={split_ok} ===\n"
                )
                ef.write(traceback.format_exc())
                ef.write("\n")
                ef.flush()

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
                    "window_size": args.window_size,
                    "kernel_size": args.kernel_size,
                    "n_sink": args.n_sink,
                    "source_method": r.get("method"),
                    "compression_mode": (
                        "context_only" if split_ok else "full_prompt"
                    ),
                    "split_attempted": split_attempted,
                    "fail_stage": stage if err else None,
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

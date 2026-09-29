#!/usr/bin/env python3
"""Run a Hugging Face causal language model over TPBench probe files.

- Loads a single HuggingFace causal LM once per process.
- Greedy decoding (deterministic).
- Parses the trailing JSON object from model output.
- Sharded execution: --shard-i / --shard-n for multi-GPU/CPU parallelism.
- Resume-safe: skips task_uid already present in --out.

Each output row contains:
  {task_uid, dialogue_id, dataset, method, probe_type, meta,
   reader_output_text, error, model}
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


JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def task_uid(row: dict, full_prompt: str) -> str:
    template_sha = row.get("template_sha16") or ""
    prompt_sha = hashlib.sha256(full_prompt.encode("utf-8")).hexdigest()[:16]
    base = "|".join([
        row.get("dialogue_id") or "",
        row.get("method") or "",
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
    """Return the last well-formed JSON object substring, or None."""
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
    """Render via the tokenizer's chat template if available, else simple
    concatenation of system and user prompts."""
    if getattr(tokenizer, "chat_template", None):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    return f"{system}\n\n{user}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-i", type=int, default=0)
    ap.add_argument("--shard-n", type=int, default=1)
    ap.add_argument("--model", type=str,
                    default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--device", type=str, default=None,
                    help="cuda / cuda:0 / cpu (auto if omitted)")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--max-input-tokens", type=int, default=7168,
                    help="left-truncate user prompt if longer")
    ap.add_argument("--progress-every", type=int, default=10)
    ap.add_argument("--attn-impl", type=str, default="eager",
                    choices=["eager", "sdpa", "flash_attention_2"],
                    help="attention impl. 'eager' is the safest on Windows "
                         "(SDPA crashes with STATUS_STACK_BUFFER_OVERRUN on "
                         "some Win+torch+CUDA combos).")
    args = ap.parse_args()

    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype_map = {"bfloat16": torch.bfloat16,
                 "float16": torch.float16,
                 "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"[info] model={args.model}  device={args.device}  dtype={args.dtype}",
          file=sys.stderr)
    print(f"[info] shard {args.shard_i}/{args.shard_n}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # P3/P1 prompts put the question at the END of prompt_user. Left-truncate
    # so over-budget inputs drop the OLDEST context rather than the question.
    tokenizer.truncation_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=args.device if args.device != "cpu" else None,
        attn_implementation=args.attn_impl,
    )
    if args.device == "cpu":
        model = model.to("cpu")
    model.eval()

    out_path = args.out.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done_uids = load_done_uids(out_path)
    print(f"[info] resume: {len(done_uids)} already done", file=sys.stderr)

    # Pre-shard probe rows
    rows = []
    with args.probes.resolve().open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i % args.shard_n != args.shard_i:
                continue
            r = json.loads(line)
            full_prompt = f"{r['prompt_system']}\n\n{r['prompt_user']}"
            r["__task_uid"] = task_uid(r, full_prompt)
            if r["__task_uid"] in done_uids:
                continue
            rows.append(r)
    print(f"[info] worker {args.shard_i}/{args.shard_n}: {len(rows)} tasks",
          file=sys.stderr)

    t_start = time.time()
    n_ok = 0
    n_err = 0
    with out_path.open("a", encoding="utf-8") as f:
        for idx, r in enumerate(rows):
            try:
                prompt_text = build_chat_prompt(
                    tokenizer, r["prompt_system"], r["prompt_user"]
                )
                inputs = tokenizer(
                    prompt_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=args.max_input_tokens,
                )
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                        temperature=1.0,
                        top_p=1.0,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                gen_ids = out[0, inputs["input_ids"].shape[1]:]
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                json_obj = extract_json(gen_text) or ""
                err = None if json_obj else "no-json"
                raw = json_obj if json_obj else gen_text.strip()
            except Exception as exc:
                raw = ""
                err = f"exception:{type(exc).__name__}:{exc}"

            row_out = {
                "task_uid": r["__task_uid"],
                "dialogue_id": r["dialogue_id"],
                "dataset": r["dataset"],
                "method": r["method"],
                "probe_type": r["probe_type"],
                "meta": r.get("meta"),
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

    print(f"\n[done] worker {args.shard_i}/{args.shard_n}: ok={n_ok} err={n_err}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

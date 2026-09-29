#!/usr/bin/env python3
"""Dialogue-bootstrap uncertainty and random-selector seed summaries."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "results" / "seed_robustness"
CFG = json.loads((RESULT_ROOT / "experiment.json").read_text(encoding="utf-8"))
N_BOOT = int(CFG["protocol"]["bootstrap_replicates"])
BOOT_SEED = int(CFG["protocol"]["bootstrap_seed"])
ALPHA = 1.0 - float(CFG["protocol"]["confidence_level"])


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def stable_seed(label: str) -> int:
    digest = hashlib.sha256(f"{BOOT_SEED}|{label}".encode()).hexdigest()
    return int(digest[:16], 16)


def bootstrap(values: dict[str, float], label: str) -> dict:
    ids = sorted(values)
    observed = sum(values[x] for x in ids) / len(ids)
    rng = random.Random(stable_seed(label))
    reps = []
    for _ in range(N_BOOT):
        reps.append(sum(values[ids[rng.randrange(len(ids))]] for _ in ids) / len(ids))
    return {
        "n_dialogues": len(ids),
        "mean": observed,
        "ci_low": percentile(reps, ALPHA / 2),
        "ci_high": percentile(reps, 1 - ALPHA / 2),
        "bootstrap_replicates": N_BOOT,
        "bootstrap_unit": "dialogue_id",
    }


def score_file(path: Path, metrics: list[str], label: str) -> dict:
    rows = load_jsonl(path)
    by_metric_method: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in rows:
        did, method = str(row["dialogue_id"]), str(row["method"])
        for metric in metrics:
            if metric in row:
                by_metric_method[metric][method][did] = float(row[metric])

    output = {}
    for metric, methods in sorted(by_metric_method.items()):
        metric_out = {}
        for method, values in sorted(methods.items()):
            metric_out[method] = bootstrap(values, f"{label}|{metric}|{method}")

        random_methods = sorted(m for m in methods if m.startswith("random_seed"))
        if random_methods:
            common = set(methods[random_methods[0]])
            for method in random_methods[1:]:
                common &= set(methods[method])
            dialogue_means = {
                did: sum(methods[m][did] for m in random_methods) / len(random_methods)
                for did in sorted(common)
            }
            seed_means = [
                sum(methods[m].values()) / len(methods[m]) for m in random_methods
            ]
            consolidated = bootstrap(
                dialogue_means, f"{label}|{metric}|random_selection"
            )
            seed_mean = sum(seed_means) / len(seed_means)
            seed_sd = math.sqrt(
                sum((x - seed_mean) ** 2 for x in seed_means)
                / max(1, len(seed_means) - 1)
            )
            consolidated.update({
                "selector_seeds": [int(m.removeprefix("random_seed"))
                                   for m in random_methods],
                "selector_seed_means": seed_means,
                "selector_seed_mean": seed_mean,
                "selector_seed_sd": seed_sd,
                "estimand": "mean performance over the five random selectors",
            })
            metric_out["random_selection"] = consolidated
        output[metric] = metric_out
    return output


def average_ranks(scores: dict[str, float]) -> dict[str, float]:
    groups: dict[float, list[str]] = defaultdict(list)
    for method, score in scores.items():
        groups[score].append(method)
    ranks = {}
    position = 1
    for score in sorted(groups, reverse=True):
        methods = sorted(groups[score])
        average = (position + position + len(methods) - 1) / 2
        for method in methods:
            ranks[method] = average
        position += len(methods)
    return ranks


def spearman(left: dict[str, float], right: dict[str, float]) -> float:
    common = sorted(set(left) & set(right))
    lr, rr = average_ranks({m: left[m] for m in common}), average_ranks(
        {m: right[m] for m in common}
    )
    xs, ys = [lr[m] for m in common], [rr[m] for m in common]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x - mx) ** 2 for x in xs) *
                    sum((y - my) ** 2 for y in ys))
    return num / den if den else float("nan")


def consolidated_means(metric_summary: dict) -> dict[str, float]:
    allowed = {"full_context", "recency", "first_n", "uniform_stride",
               "random_selection"}
    return {
        method: values["mean"] for method, values in metric_summary.items()
        if method in allowed
    }


def compressed_means(metric_summary: dict) -> dict[str, float]:
    allowed = {"recency", "first_n", "uniform_stride", "random_selection"}
    return {
        method: values["mean"] for method, values in metric_summary.items()
        if method in allowed
    }


def selector_seed_rank_checks(p1_summary: dict, p2_summary: dict) -> list[dict]:
    checks = []
    deterministic = {"recency", "first_n", "uniform_stride"}
    random_methods = sorted(
        method for method in p1_summary if method.startswith("random_seed")
    )
    for random_method in random_methods:
        methods = deterministic | {random_method}
        p1 = {method: p1_summary[method]["mean"] for method in methods}
        p2 = {method: p2_summary[method]["mean"] for method in methods}
        p1_best = sorted(
            method for method, value in p1.items() if value == max(p1.values())
        )
        p2_best = sorted(
            method for method, value in p2.items() if value == max(p2.values())
        )
        checks.append({
            "selector_seed": int(random_method.removeprefix("random_seed")),
            "spearman": spearman(p1, p2),
            "p1_best_compressed": p1_best,
            "p2_best_compressed": p2_best,
            "same_best_method": bool(set(p1_best) & set(p2_best)),
        })
    return checks


def fmt(value: float) -> str:
    return f"{value:.3f}"


def markdown(summary: dict) -> str:
    lines = [
        "# TPBench seed extension results",
        "",
        f"Dialogue bootstrap: {N_BOOT:,} replicates; random selector seeds: "
        + ", ".join(map(str, CFG["protocol"]["random_selector_seeds"])),
        "",
    ]
    sections = [
        ("RiSAWOZ P2", "risawoz_p2", "p3_correct"),
        ("LongMemEval-KU P2", "lme_ku_p2", "p3_em_strict"),
    ]
    display = ["full_context", "recency", "first_n", "uniform_stride",
               "random_selection"]
    for title, key, metric in sections:
        lines.extend([f"## {title} strict", "",
                      "| Method | Mean | 95% CI | n |", "|---|---:|---:|---:|"])
        methods = summary[key][metric]
        for method in display:
            if method not in methods:
                continue
            s = methods[method]
            lines.append(
                f"| {method} | {fmt(s['mean'])} | "
                f"[{fmt(s['ci_low'])}, {fmt(s['ci_high'])}] | "
                f"{s['n_dialogues']} |"
            )
        lines.append("")
    rank = summary.get("risawoz_rank_comparison", {})
    if rank:
        lines.extend([
            "## RiSAWOZ P1--P2 method-rank comparison",
            "",
            "Compressed methods only; random selection is averaged over the "
            "five selector seeds.",
            "",
            f"Strict-rank Spearman rho: {rank['strict_spearman']:.3f}",
            "",
            "The best compressed P1 and P2 methods differ in all five "
            "selector-seed checks.",
            "",
        ])
    lines.extend([
        "LongMemEval-KU reports current-information retention (P2).",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    paths = {
        "risawoz_p2": (RESULT_ROOT / "scored" / "risawoz_p2.jsonl",
                        ["p3_correct"]),
        "lme_ku_p2": (RESULT_ROOT / "scored" / "lme_ku_p2.jsonl",
                       ["p3_em_strict", "p3_em_loose"]),
    }
    for path, _ in paths.values():
        if not path.exists():
            raise SystemExit(f"missing scored file: {path}")
    summary = {
        name: score_file(path, metrics, name)
        for name, (path, metrics) in paths.items()
    }
    summary["protocol"] = {
        "bootstrap_seed": BOOT_SEED,
        "bootstrap_replicates": N_BOOT,
        "bootstrap_unit": "dialogue_id",
        "confidence_level": CFG["protocol"]["confidence_level"],
    }
    out_dir = RESULT_ROOT / "stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (out_dir / "summary.md").write_text(markdown(summary), encoding="utf-8")
    print(f"[done] wrote {out_dir / 'summary.json'}")
    print(f"[done] wrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

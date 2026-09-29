"""Generate results/aggregates_flat.jsonl from per-cell aggregate JSON files.

The per-cell aggregates ship as nested JSON keyed by method (e.g. ``{method:
{n, p1_em_strict, ...}}``), with a few cells additionally wrapped in a dataset
key. This script flattens every cell-method pair to a single JSON line whose
top-level fields match the Croissant ``aggregates`` RecordSet jsonPaths.

Run from the package root after any aggregate is added or replaced:
    python scripts/build_aggregates_flat.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path

DATASET_TOKENS = {"sgd", "multiwoz", "mw", "schema", "lme_ku", "risawoz"}
ROOT = Path(__file__).resolve().parents[1]


def ds_norm(s: str) -> str:
    return {"mw": "multiwoz"}.get(s, s)


def emit(rows, *, cell_id, dataset, ratio, seed, probe, reader, method, m_data):
    if not isinstance(m_data, dict):
        return
    if probe == "P1":
        strict = m_data.get("p1_em_strict")
        loose = m_data.get("p1_em_loose")
    elif probe == "P3":
        strict = m_data.get("p3_em_strict")
        loose = m_data.get("p3_em_loose")
    else:
        # Paper P3 primary scoring is segment-local. The legacy p1late fields
        # remain whole-field sensitivity evidence. Preserve genuine zero rates.
        strict = next((m_data[k] for k in ("p3_joint_strict", "p1_late_em_strict",
            "combined_strict", "p1late_combined_rate_strict") if k in m_data), None)
        loose = next((m_data[k] for k in ("p3_joint_loose", "p1_late_em_loose",
            "combined_loose", "p1late_combined_rate") if k in m_data), None)
    rows.append({
        "cell_id": cell_id,
        "dataset": dataset,
        "compression_ratio": ratio,
        "seed": seed,
        "probe_type": probe,
        "reader": reader,
        "method": method,
        "n": m_data.get("n"),
        "strict_acc": strict,
        "loose_acc": loose,
    })


def add_methods(rows, agg_path, *, cell_id, dataset, ratio, seed, probe, reader):
    with open(agg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return
    top = list(data.keys())
    if len(top) == 1 and top[0] in DATASET_TOKENS:
        inner = data[top[0]]
    else:
        inner = data
    if not isinstance(inner, dict):
        return
    for method, m_data in inner.items():
        if method in DATASET_TOKENS:
            continue
        emit(rows, cell_id=cell_id, dataset=dataset, ratio=ratio, seed=seed,
             probe=probe, reader=reader, method=method, m_data=m_data)


def main() -> None:
    rows: list[dict] = []

    pilot_skip = ("seed_summary", "sweep", "valuespan", "tier1")

    for fp in sorted(glob.glob(str(ROOT / "results/main/*_aggregate.json"))):
        fname = os.path.basename(fp).replace(".json", "")
        if any(tok in fname for tok in pilot_skip):
            continue
        m = re.match(r"chunkkv_(?P<ds>sgd|mw)_r(?P<r>\d+)_(?P<probe>p1|p3|late)_aggregate$", fname)
        if m:
            ds = ds_norm(m.group("ds"))
            r = int(m.group("r")) / 100.0
            probe = {"p1": "P1", "p3": "P3", "late": "P1_LATE"}[m.group("probe")]
            if probe == "P1_LATE":
                # Joint results are read from update_evidence below.
                continue
            cell = f"{ds}_r{int(r*100):02d}_chunkkv_{probe}"
            add_methods(rows, fp, cell_id=cell, dataset=ds, ratio=r, seed=None,
                        probe=probe, reader="llama-3.1-8b-instruct")
            continue
        m = re.match(
            r"(?P<ds>sgd|mw)_r(?P<r>\d+)_s(?P<s>\d+)"
            r"(?:_(?P<sfx>p3|late_intent|late_intent_extras))?_aggregate$",
            fname,
        )
        if m:
            ds = ds_norm(m.group("ds"))
            r = int(m.group("r")) / 100.0
            s = int(m.group("s"))
            sfx = m.group("sfx")
            probe = {"p3": "P3", "late_intent": "P1_LATE",
                     "late_intent_extras": "P1_LATE", None: "P1"}[sfx]
            if probe == "P1_LATE":
                # Joint results are read from update_evidence below.
                continue
            cell = f"{ds}_r{int(r*100):02d}_s{s}_{probe}"
            add_methods(rows, fp, cell_id=cell, dataset=ds, ratio=r, seed=s,
                        probe=probe, reader="llama-3.1-8b-instruct")

    for fp in sorted(glob.glob(str(ROOT / "results/diagnostics/mistral_128/*_aggregate.json"))):
        fname = os.path.basename(fp).replace(".json", "")
        m = re.match(
            r"(?P<ds>sgd|mw)_r(?P<r>\d+)_s(?P<s>\d+)"
            r"_(?P<probe>p1|p3|late)_aggregate$",
            fname,
        )
        if m:
            ds = ds_norm(m.group("ds"))
            r = int(m.group("r")) / 100.0
            seed = int(m.group("s"))
            probe = {"p1": "P1", "p3": "P3", "late": "P1_LATE"}[m.group("probe")]
            if probe == "P1_LATE":
                continue
            cell = f"{ds}_r{int(r*100):02d}_s{seed}_{probe}_mistral128"
            add_methods(rows, fp, cell_id=cell, dataset=ds, ratio=r, seed=seed,
                        probe=probe, reader="mistral-7b-instruct-128")

    for seed in (42, 43, 44):
        for fp in sorted(glob.glob(str(ROOT / f"results/mistral_seed{seed}/*_aggregate.json"))):
            fname = os.path.basename(fp).replace(".json", "")
            m = re.match(
                r"(?P<ds>sgd|mw)_r(?P<r>\d+)(?:_(?P<probe>p1|p3|late))?_aggregate$", fname,
            )
            if not m:
                continue
            ds = ds_norm(m.group("ds"))
            r = int(m.group("r")) / 100.0
            probe = {"p1": "P1", "p3": "P3", "late": "P1_LATE", None: "P1"}[m.group("probe")]
            if probe == "P1_LATE":
                continue
            if probe == "P3":
                # Mistral P2 uses the 128-token configuration above.
                continue
            cell = f"{ds}_r{int(r*100):02d}_s{seed}_{probe}_mistral96"
            add_methods(rows, fp, cell_id=cell, dataset=ds, ratio=r, seed=seed,
                        probe=probe, reader="mistral-7b-instruct-96")

    # Cache-level P1/P2 comparisons.
    for fp in sorted(glob.glob(str(ROOT / "results/kv_methods/*_aggregate.json"))):
        fname = os.path.basename(fp).replace(".json", "")
        m = re.match(
            r"(?P<ds>sgd|multiwoz)_r30_s42_(?P<probe>p1|p3)_"
            r"(?P<kv>snapkv|pyramidkv|streamingllm)_aggregate$",
            fname,
        )
        if not m:
            continue
        ds = m.group("ds")
        probe = {"p1": "P1", "p3": "P3"}[m.group("probe")]
        kv = m.group("kv")
        add_methods(
            rows,
            fp,
            cell_id=f"{ds}_r30_s42_{probe}_{kv}",
            dataset=ds,
            ratio=0.3,
            seed=42,
            probe=probe,
            reader="llama-3.1-8b-instruct-kvpress",
        )

    # Joint P3 cells with final-update position filtering.
    origin_dir = ROOT / "results/diagnostics/update_evidence"
    for tag, reader in (
        ("llama", "llama-3.1-8b-instruct-sdpa128"),
        ("mistral", "mistral-7b-instruct-eager128"),
    ):
        for fp in sorted(origin_dir.glob(f"p3_*_s*_aggregate_{tag}.json")):
            m = re.match(
                r"p3_(?P<ds>sgd|multiwoz)_s(?P<s>\d+)_aggregate_"
                + re.escape(tag)
                + r"$",
                fp.stem,
            )
            if not m:
                continue
            ds = m.group("ds")
            seed = int(m.group("s"))
            cell = f"{ds}_r30_s{seed}_P1_LATE_{tag}"
            add_methods(
                rows,
                fp,
                cell_id=cell,
                dataset=ds,
                ratio=0.3,
                seed=seed,
                probe="P1_LATE",
                reader=reader,
            )

    for fp in sorted(origin_dir.glob("p3_*_s42_aggregate_chunkkv.json")):
        m = re.match(
            r"p3_(?P<ds>sgd|multiwoz)_s42_aggregate_chunkkv$",
            fp.stem,
        )
        if not m:
            continue
        ds = m.group("ds")
        add_methods(
            rows,
            fp,
            cell_id=f"{ds}_r30_s42_P1_LATE_chunkkv",
            dataset=ds,
            ratio=0.3,
            seed=42,
            probe="P1_LATE",
            reader="llama-3.1-8b-instruct-chunkkv-eager128",
        )

    for fp in sorted(glob.glob(str(ROOT / "results/lme_ku/*_aggregate.json"))):
        fname = os.path.basename(fp).replace(".json", "")
        m = re.match(r"lme_ku_s(?P<s>\d+)_r(?P<r>\d+)_n\d+(?:_sdpa)?_aggregate$", fname)
        if not m:
            continue
        r = int(m.group("r")) / 100.0
        s = int(m.group("s"))
        cell = f"lme_ku_r{int(r*100):02d}_s{s}_P3"
        add_methods(rows, fp, cell_id=cell, dataset="lme_ku", ratio=r, seed=s,
                    probe="P3", reader="llama-3.1-8b-instruct")

    for fp in sorted(glob.glob(str(ROOT / "results/risawoz/*_aggregate.json"))):
        fname = os.path.basename(fp).replace(".json", "")
        m = re.match(r"risawoz_s(?P<s>\d+)_r(?P<r>\d+)_n\d+_aggregate$", fname)
        if not m:
            continue
        r = int(m.group("r")) / 100.0
        s = int(m.group("s"))
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        for key, m_data in data.items():
            if "::" not in key or not isinstance(m_data, dict):
                continue
            method, probe = key.split("::", 1)
            cell = f"risawoz_r{int(r*100):02d}_s{s}_{probe}"
            rows.append({
                "cell_id": cell,
                "dataset": "risawoz",
                "compression_ratio": r,
                "seed": s,
                "probe_type": probe,
                "reader": "llama-3.1-8b-instruct",
                "method": method,
                "n": m_data.get("n"),
                "strict_acc": m_data.get("em_strict"),
                "loose_acc": m_data.get("em_loose"),
            })

    seen: dict[tuple, dict] = {}
    for r in rows:
        seen[(r["cell_id"], r["method"])] = r
    deduped = list(seen.values())

    out = ROOT / "results/aggregates_flat.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    # newline="" disables Windows CRLF translation so the file byte content
    # (and therefore its sha256) is identical on Linux, macOS, and Windows.
    with out.open("w", encoding="utf-8", newline="") as f:
        for r in deduped:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out} with {len(deduped)} rows ({len(rows) - len(deduped)} dupes dropped) "
          f"across {len({r['cell_id'] for r in deduped})} cells")


if __name__ == "__main__":
    main()

"""Reproduce the answer-retention tables of the paper.

Table "Retention and reading": P2-strict, answer retention, and accuracy on the
retained examples, for every method on both datasets at r=0.30.

Table "Retention by slot type": the same two quantities for SGD, split by what
the revised slot holds.

Retention means the complete current value occurs at phrase boundaries in the
compressed dialogue, using the matcher that also defines P2-loose. All paths are
relative to the repository root. The script reads saved answers only and calls
no model.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from normalize import value_occurs_in_text  # noqa: E402

SEEDS = (42, 43, 44)
FILE_TAG = {"sgd": "sgd", "multiwoz": "mw"}
METHODS = [
    "full_context", "recency", "random_seed42", "first_n", "uniform_stride",
    "attention_h2o_cache", "embedding_mmr_cache", "llmlingua2_cache",
]
LABEL = {
    "full_context": "full context", "recency": "recency",
    "random_seed42": "random selection", "first_n": "earliest turns",
    "uniform_stride": "uniform stride", "attention_h2o_cache": "attention/H2O proxy",
    "embedding_mmr_cache": "embedding MMR", "llmlingua2_cache": "LLMLingua-2",
}


def slot_group(slot):
    name = slot.lower()
    if "time" in name:
        return "Time"
    if "date" in name or "day" in name:
        return "Date"
    place = ("city", "location", "origin", "destination", "area", "place",
             "name", "theater", "hotel", "restaurant", "departure", "dest")
    if any(key in name for key in place):
        return "Place and name"
    return "Other"


def read_seed(dataset, seed):
    """Return scored rows joined to their probe record."""
    probes = {}
    path = ROOT / f"data/probes/probes_{dataset}_s{seed}_r30_p3_n200.jsonl"
    for line in path.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            probes[(record["dialogue_id"], record["method"])] = record
    joined = []
    path = ROOT / f"results/main/{FILE_TAG[dataset]}_r30_s{seed}_p3_scored.jsonl"
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        probe = probes.get((row["dialogue_id"], row["method"]))
        if probe is None:
            continue
        retained = value_occurs_in_text(
            probe["gold"], probe["compressed_text_used"], dataset)
        joined.append((row["method"], probe["slot"], bool(retained),
                       int(row["p3_correct"])))
    return joined


def summarize(rows):
    """Return accuracy, retention, and accuracy on the retained examples."""
    if not rows:
        return None
    accuracy = sum(correct for *_, correct in rows) / len(rows)
    kept = [correct for *_, retained, correct in rows if retained]
    retention = len(kept) / len(rows)
    return accuracy, retention, (sum(kept) / len(kept) if kept else float("nan"))


def main():
    per_seed = {ds: {s: read_seed(ds, s) for s in SEEDS} for ds in FILE_TAG}

    print("Retention and reading, r=0.30, mean over seeds", list(SEEDS))
    header = f"{'method':22}{'dataset':10}{'P2-strict':>11}{'retained':>11}{'correct|kept':>14}"
    print(header)
    for dataset in ("sgd", "multiwoz"):
        for method in METHODS:
            cells = [summarize([r for r in per_seed[dataset][s] if r[0] == method])
                     for s in SEEDS]
            cells = [c for c in cells if c]
            if not cells:
                continue
            accuracy, retention, given = (statistics.mean(v) for v in zip(*cells))
            print(f"{LABEL[method]:22}{dataset:10}{accuracy:11.3f}{retention:11.3f}{given:14.3f}")
        print()

    print("SGD retention by slot type, r=0.30, mean over seeds", list(SEEDS))
    groups = ("Time", "Date", "Place and name", "Other")
    print(f"{'method':22}" + "".join(f"{g:>22}" for g in groups))
    for method in METHODS:
        if method == "full_context":
            continue
        line = f"{LABEL[method]:22}"
        for group in groups:
            cells = [summarize([r for r in per_seed["sgd"][s]
                                if r[0] == method and slot_group(r[1]) == group])
                     for s in SEEDS]
            cells = [c for c in cells if c]
            accuracy, retention, _ = (statistics.mean(v) for v in zip(*cells))
            line += f"{accuracy:.3f} / {retention:.3f}".rjust(22)
        print(line)

    sizes = defaultdict(list)
    for seed in SEEDS:
        counts = defaultdict(int)
        for method, slot, _, _ in per_seed["sgd"][seed]:
            if method == "recency":
                counts[slot_group(slot)] += 1
        for group in groups:
            sizes[group].append(counts[group])
    print("\nSGD examples per seed:",
          {g: sizes[g] for g in groups},
          "total", [sum(sizes[g][i] for g in groups) for i in range(len(SEEDS))])


if __name__ == "__main__":
    main()

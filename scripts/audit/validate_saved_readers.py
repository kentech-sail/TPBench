#!/usr/bin/env python3
"""Verify saved FMTS execution metadata and recompute FMTS and P1 scores.

CPU-only; no credentials, network access, or model calls are used.
The FMTS rule is preserved exactly to reproduce the reported experiment.
It is a permissive string test, not a semantic correctness criterion.
"""
import json
import re
import runpy
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def fmts_correct(pred, gold, old_state_answer=None):
    pred_l = (pred or "").lower()
    gold_l = (gold or "").lower().strip()
    if not gold_l:
        return False
    if old_state_answer and old_state_answer.strip():
        old_l = old_state_answer.lower().strip()
        if old_l and old_l != gold_l and old_l in gold_l:
            return old_l in pred_l and any(
                tok in pred_l for tok in re.findall(r"\w+", gold_l) if len(tok) > 3
            )
    return gold_l in pred_l or pred_l in gold_l


def validate():
    protocol = json.loads((ROOT / "results/fmts_execution/protocol.json").read_text())
    assert protocol["n"] == 7000
    assert "llmlingua2_cache" in protocol["excluded_conditions"]
    execution = rows(ROOT / "results/fmts_execution/answers.jsonl")
    indexed = {(r["batch_tag"], r["item_id"]): r for r in execution}
    assert len(execution) == len(indexed) == protocol["n"]
    assert all(r["model"] == "gpt-5.4-mini" and not r["error"] for r in execution)
    assert all(isinstance(r["answer"], str) and r["answer"].strip() for r in execution)
    tasks = rows(ROOT / "results/fmts_execution/tasks.jsonl")
    assert len(tasks) == protocol["n"]
    assert all(not r["item_id"].endswith("::llmlingua2_cache") for r in tasks)
    assert {(r["batch_tag"], r["item_id"]) for r in tasks} == set(indexed)
    seen = set()
    batches = 0
    for path in sorted((ROOT / "batches/fmts_refpool").glob("responses_tqa_*.jsonl")):
        tag = path.stem.removeprefix("responses_tqa_")
        gold_rows = rows(path.with_name("gold_tqa_" + tag + ".jsonl"))
        gold = {r["id"]: r for r in gold_rows}
        responses = rows(path)
        assert len(gold) == len(gold_rows) == len(responses)
        assert {r["id"] for r in responses} == set(gold)
        groups = defaultdict(list)
        for row in responses:
            key = tag, row["id"]
            assert key not in seen
            seen.add(key)
            assert row["answer"] == indexed[key]["answer"]
            g = gold[row["id"]]
            groups[g["method"]].append(fmts_correct(row["answer"], g["gold_answer"], g.get("old_state_answer")))
        summary = {m: {"n": len(v), "correct": sum(v), "tqa": round(sum(v)/len(v), 4)} for m, v in groups.items()}
        expected = json.loads((ROOT / f"results/fmts_refpool/scored_tqa_{tag}.json").read_text())["summary"]
        assert summary == expected, tag
        batches += 1
    assert seen == set(indexed) and batches == 6
    print("OK: FMTS model/answer provenance: 7,000 matches; six seven-condition aggregates reproduced; LLMLingua reader condition excluded")

    scorer = runpy.run_path(str(ROOT / "scripts/scorer/scorer_p1.py"))
    count = 0
    for path in sorted((ROOT / "results/p1_saved").glob("*_scored.jsonl")):
        data = rows(path)
        assert len({(r["dialogue_id"], r["method"]) for r in data}) == len(data)
        ds, ratio, seed, _ = path.stem.split("_")
        ds = "multiwoz" if ds == "mw" else ds
        probes = rows(ROOT / f"data/probes/probes_{ds}_{seed}_{ratio}_p1_n200.jsonl")
        probe_index = {(r["dialogue_id"], r["method"]): r for r in probes}
        assert set(probe_index) == {(r["dialogue_id"], r["method"]) for r in data}
        groups = defaultdict(list)
        for row in data:
            assert row["gold"] == probe_index[row["dialogue_id"], row["method"]]["gold"]
            parsed = {"value": row["pred_value"], "abstain": row["abstain"]}
            score = scorer["score_p1"]({"gold": row["gold"]}, parsed)
            assert score["p1_correct"] == row["p1_correct"], (path.name, row["dialogue_id"], row["method"])
            assert score["p1_correct_loose"] == row["p1_correct_loose"]
            groups[row["method"]].append(score)
        aggregate = json.loads((ROOT / "results/main" / path.name.replace("_scored.jsonl", "_aggregate.json")).read_text())
        for method, values in groups.items():
            assert len(values) == aggregate[method]["n"] == 200
            for metric, field in [("p1_em_strict", "p1_correct"), ("p1_em_loose", "p1_correct_loose")]:
                actual = sum(r[field] for r in values) / len(values)
                assert abs(actual - aggregate[method][metric]) < 1e-9, (path.name, method, metric)
        count += len(data)
    assert count == 9600, count
    print("OK: P1 9,600 unique saved answers rescored; six per-seed cells reproduced")


if __name__ == "__main__":
    validate()

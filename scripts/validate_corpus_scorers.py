#!/usr/bin/env python3
"""Check the released LongMemEval-KU scores and RiSAWOZ tokenization."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_lme_scores() -> None:
    scorer_path = ROOT / "scripts/scorer/scorer_lme_ku.py"
    scorer = load_module("scorer_lme_ku", scorer_path)

    assert not scorer.strict_match("2", "32")
    assert not scorer.strict_match("0", "220")
    assert not scorer.strict_match("Kansas City Master", "Kansas City Masterpiece")
    assert scorer.strict_match(25, "25")
    assert scorer.strict_match(
        "25:50", "25 minutes and 50 seconds (or 25:50)"
    )

    with tempfile.TemporaryDirectory(prefix="tpbench_lme_score_") as tmp:
        tmp_path = Path(tmp)
        for seed in (42, 43):
            for ratio in (10, 30):
                stem = f"lme_ku_s{seed}_r{ratio}_n72_sdpa"
                generated_scored = tmp_path / f"{stem}_scored.jsonl"
                generated_aggregate = tmp_path / f"{stem}_aggregate.json"
                subprocess.run(
                    [
                        sys.executable,
                        str(scorer_path),
                        "--probes",
                        str(
                            ROOT
                            / f"data/probes/probes_lme_ku_s42_r{ratio}_n72.jsonl"
                        ),
                        "--reader-out",
                        str(ROOT / f"results/lme_ku/{stem}_reader_shard0.jsonl"),
                        "--scored-out",
                        str(generated_scored),
                        "--aggregate-out",
                        str(generated_aggregate),
                    ],
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
                assert generated_scored.read_bytes() == (
                    ROOT / f"results/lme_ku/{stem}_scored.jsonl"
                ).read_bytes()
                assert generated_aggregate.read_bytes() == (
                    ROOT / f"results/lme_ku/{stem}_aggregate.json"
                ).read_bytes()


def check_risawoz_tokenizer() -> None:
    scorer = load_module(
        "scorer_p1_zh", ROOT / "scripts/scorer/scorer_p1_zh.py"
    )
    assert scorer.content_tokens("我想预订酒店") == set("我想预订酒店")


def main() -> int:
    check_lme_scores()
    check_risawoz_tokenizer()
    print("OK: LongMemEval-KU scores and RiSAWOZ tokenization validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

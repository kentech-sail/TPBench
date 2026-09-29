# Paper results and reproducibility

All paths below are relative to this repository. `bash run_all.sh` checks
saved evidence without model inference.

| Paper comparison | Evidence |
|---|---|
| Main P1/P2 | `results/main/seed_summary.json` |
| P1 saved answers and token-F1 | `results/p1_saved/`, `results/reader_evaluation/scored/`, `results/main/seed_summary.json` |
| P3, Llama / Mistral (reported, segment-local) | `results/p3_segment_sensitivity/summary.json`, fields `separated_strict` and `separated_loose`; canonical scorer CLI: `scripts/scorer/scorer_paper_p3.py`; paired analysis: `scripts/semantic_evaluation/audit_p3_segments.py` |
| P3 whole-field sensitivity | `results/diagnostics/update_evidence/p3_crossseed_{llama,mistral}.json`, `results/aggregates_flat.jsonl` (`p1late_combined_rate*`), and per-cell outputs |
| P3 alias-exclusion sensitivity | `results/p3_alias_exclusion/summary.json`; regenerate with `scripts/audit/p3_alias_exclusion_sensitivity.py`; original saved generations, variable seed n, no resampling or new inference |
| P2 paired deletion | `results/diagnostics/update_evidence/p2_summary_llama.json`, `p2_scored_llama.jsonl` |
| Mistral P2 | `results/diagnostics/mistral_128/seed_summary_mistral128.json` |
| LongMemEval-KU / RiSAWOZ | `results/seed_robustness/stats/summary.json` and `scored/` |
| Initial-goal/current-value rank figure | `results/figures/tpbench_probe_ranks.pdf`, `.inputs.json`; reproduce with `python3 scripts/corrections/draw_probe_ranks.py --summary results/main/seed_summary.json --out outputs/tpbench_probe_ranks.pdf` (requires `reportlab`; create `outputs/` first) |
| Corrected LLMLingua contexts and reader provenance | `data/llmlingua_correction/`, `results/llmlingua_correction/provenance.json`, `scripts/corrections/` |
| FMTS free-form answers (full context + six selectors; LLMLingua excluded) | `results/fmts_refpool/scored_tqa_*.json`, `batches/fmts_refpool/` |
| FMTS model, prompts, and saved execution | `results/fmts_execution/` |
| FMTS text retention | Other JSON files under `results/fmts_refpool/` |
| KV methods | `results/aggregates_flat.jsonl`; ChunkKV P3 under `results/diagnostics/update_evidence/` |
| Wider budgets | `results/main/seed_summary_wider.json` |
| Update-turn annotations | `results/diagnostics/update_evidence/origin_index_*.jsonl`, `*_goal_labels.jsonl` |
| P1 semantic equivalence (full context + six selectors; LLMLingua excluded) | `results/semantic_evaluation/p1_summary.json` and `p1/` |
| First-user-plus-recent matched comparison | `data/reader_evaluation/`, `results/reader_evaluation/` |
| Answer retention and retention by slot type | `scripts/audit/retention_vs_reading.py` (joins `data/probes/` to `results/main/`) |

## Example and method identifiers

The README maps paper P1/P2/P3 to the file tags and scorer fields. Earliest
turns is `first_n`; random selection is `random_seed42` in the main pool;
H2O, MMR, and LLMLingua-2 use `attention_h2o_cache`, `embedding_mmr_cache`,
and `llmlingua2_cache`. These identifiers join contexts to saved answers.
Random selectors in the corpus/language comparison use five selector seeds.

`s_origin` identifies the final update turn. `s_evidence` contains normalized
phrase matches for the reference value. `s_transition_support` is their union.
The final-update candidate pools contain 791 SGD and 664 MultiWOZ examples;
seed-42 P2 has 128 and 161 examples. The all-evidence matched comparison has
128 and 159 examples. P3 samples 200 examples from each original late-update pool (452 SGD / 277
MultiWOZ). This original pool does not apply P2's old/new alias-equivalence
exclusion. Its prompt asks for the user's task "in this conversation", while
the scoring reference is the initial first-sentence anchor. Original prompts
and results are preserved. The heuristic alias-exclusion sensitivity retains
152/147/147 SGD and 181/183/180 MultiWOZ examples for seeds 42/43/44; this is
not independent semantic annotation or a corrected n=200 rerun.

## Statistical and scoring conventions

The deletion summary's `paired_discordance` conditions on a correct
full-context answer. The paper reports its four exact McNemar p-values:
2.38e-7, 1.08e-19, 1.03e-21, and 1.88e-37. The same summary also reports
all-matched-item comparisons under `paired_discordance_all_items`. The deletion
reader settings differ from the main P2 settings, as specified in the README.

`validate_saved_readers.py` rescores 9,600 saved P1 answers and
reproduces six per-seed aggregates. `test_reader_evaluation.py` verifies the
raw reader answers, token-F1, semantic scores, and matched positional baseline. The saved
answer identifiers select one answer per example and method; a source dialogue
can contribute multiple slot questions. P1 semantic evaluation separately uses
8,400 answers and 3,049 distinct judged pairs after excluding LLMLingua.
The full historical semantic records are preserved under
`results/historical/semantic_p1_legacy_20260929/`; they do not judge the corrected
LLMLingua outputs. Pending new pairs are released in
`data/llmlingua_correction/semantic_pending_inputs.jsonl`.

The canonical `scorer_paper_p3.py` reproduces the reported segment-local scores
on all 19,600 saved P3 answers (`scripts/tests/test_paper_p3.py`), including a
raw-reader CLI round trip. `scorer_p1_late.py` continues to implement the
whole-field sensitivity. `test_p3_alias_sensitivity.py` regenerates the saved
alias-exclusion subset with the original per-seed denominators.

The 7,000 current FMTS execution records identify `gpt-5.4-mini` and match the released
answer batches. The same validator reproduces all six answer-accuracy tables.
The previous 8,000-record set, including invalidated LLMLingua comparisons, is
archived under `results/historical/fmts_legacy_20260929/`. Corrected LLMLingua
compression diagnostics are independent of the retained reader comparisons.
The exact prompt and recorded backend are in `protocol.json`. FMTS uses a
permissive string score; its synthetic deletion experiment uses a separate
phrase-bounded rule. The saved API-backed run has no immutable model revision
or explicit temperature. Fresh execution therefore remains model-service dependent.

Files under `results/bench/` contain selector timing. They are not measured
reader prompt-token medians or end-to-end inference speedups. The paper's
linear/quadratic cost discussion is conditional token-length scaling.

## Reproduction levels

1. **Saved evidence:** CPU checks of scores, matched rows, schemas, and hashes.
2. **Construction:** regenerate probes and FMTS contexts from included inputs.
3. **Inference:** run the included reader workers with model access and hardware.

GPU and API inference can vary across hardware, software, and model-service
versions. The distributed saved outputs pin the reported experiment.

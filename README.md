---
pretty_name: 'TPBench: A Turning-Point Benchmark for Dialogue Compression'
license: other
language:
- en
- zh
tags:
- dialogue
- long-context
- evaluation
- croissant
task_categories:
- question-answering
configs:
- config_name: initial_goal
  data_files:
  - split: probes
    path: data/probes/probes_*_p1_n200.jsonl
- config_name: current_value
  data_files:
  - split: probes
    path: data/probes/probes_*_p3_n200.jsonl
  features:
  - name: dialogue_id
    dtype: string
  - name: dataset
    dtype: string
  - name: slot
    dtype: string
  - name: slot_human
    dtype: string
  - name: old_value
    dtype: string
  - name: new_value
    dtype: string
  - name: method
    dtype: string
  - name: compressed_text_chars
    dtype: int64
  - name: meta
    struct:
    - name: n_turns
      dtype: int64
    - name: k_turns
      dtype: int64
    - name: ratio
      dtype: float64
    - name: last_system_turn_id
      dtype: int64
    - name: last_boundary_turn_id
      dtype: int64
    - name: last_boundary_turn_id_oracle
      dtype: int64
    - name: last_boundary_turn_id_det
      dtype: int64
    - name: origin_turn_id
      dtype: int64
    - name: origin_source
      dtype: string
    - name: control_origin
      list: int64
    - name: control_transition_support
      list: int64
    - name: transition_support_control_status
      dtype: string
    - name: final_slot_update_turn_id
      dtype: int64
    - name: is_final_slot_update
      dtype: bool
    - name: s_origin
      list: int64
    - name: s_evidence
      list: int64
    - name: s_transition_support
      list: int64
    - name: control_transition_origin_component
      list: int64
    - name: control_transition_evidence_component
      list: int64
    - name: transition_support_control_role_status
      dtype: string
  - name: prompt_system
    dtype: string
  - name: probe_type
    dtype: string
  - name: gold
    dtype: string
  - name: prompt_user
    dtype: string
  - name: original_meta_k_turns
    dtype: int64
  - name: original_compressed_text_chars
    dtype: int64
  - name: compressed_text_chars_used
    dtype: int64
  - name: compressed_text_used
    dtype: string
  - name: template_sha16
    dtype: string
- config_name: joint_recovery
  data_files:
  - split: probes
    path: data/probes/probes_*_late_intent.jsonl
---

# TPBench: A Turning-Point Benchmark for Dialogue Compression

TPBench measures three retrieval targets after dialogue compression:

- **P1:** the user's initial goal;
- **P2:** the final value of a revised slot;
- **P3:** both answers when the final update occurs late.

This repository includes data, construction inputs, compression and reader
code, scorers, saved answers, and paper results. [RESULTS_MAP.md](RESULTS_MAP.md)
connects the paper comparisons to their evidence files.

## LLMLingua adapter correction

The previous turn selector extracted every number from LLMLingua's labeled
text, including numbers in the dialogue. The corrected adapter parses only
the binary word-retention labels and averages them within each turn.
Legacy replay exactly reproduces all 3,374 main and 1,000 FMTS saved contexts;
corrected selection changes 3,300 and 792 contexts, respectively. This establishes
the input-selection error independently of reader generation.
The saved labels and per-item comparisons are in `data/llmlingua_correction/`;
`scripts/corrections/validate_llmlingua_selections.py` reconstructs all 4,374
legacy/corrected selections without downloading model weights.

The correction retains all seven selectors in the local Llama/Mistral lexical
evaluations. [Runtime provenance](results/llmlingua_correction/provenance.json)
records fresh versus reused outputs, exact configurations, and software differences;
reader-score changes do not isolate the parser as their sole cause.
P1 semantic evaluation and FMTS reader-accuracy comparisons exclude
the LLMLingua condition because fresh judgments/answers are not available for
all corrected inputs. These comparisons retain full context and six selectors.
Do not reuse historical LLMLingua judgments or answers as corrected results.
FMTS compression diagnostics still include the corrected LLMLingua contexts;
they require no reader generation. Historical FMTS evidence is preserved under
`results/historical/fmts_legacy_20260929/` and is excluded from current comparisons.
See [the result map](RESULTS_MAP.md), [semantic evaluation](SEMANTIC_EVALUATION.md),
and [FMTS instructions](scripts/fmts_refpool/README.md) for the exact scopes.

## Check the release

Use Python 3.10 or newer. No model access is needed for these checks.

```bash
python3 -m pip install mlcroissant numpy rouge-score
bash run_all.sh
```

The checks validate file hashes and metadata, regenerate all 36 main probe
cells, rescore saved answers, and test FMTS context construction. Outputs from
tests are written to a temporary directory, not over the released results.

## Use a probe

```python
import json
from pathlib import Path
path = Path("data/probes/probes_sgd_s42_r30_p3_n200.jsonl")
rows = [json.loads(line) for line in path.open(encoding="utf-8")]
example = rows[0]
print(example["prompt_system"], example["prompt_user"])
print(example["gold"])
```

Each row fixes the example identifier, method, budget, reference answer, and
reader prompts. Identifiers remain stable across construction inputs and saved
answers. The following mapping specifies the machine-readable protocol:

| Paper probe | File tag / `probe_type` | Scorer |
|---|---|---|
| P1 initial goal | `p1` / `P1` | `scripts/scorer/scorer_p1.py` |
| P2 current value | `p3` / `P3` | `scripts/scorer/scorer_p3.py` |
| P3 joint recovery | `late_intent` / `P1_LATE` | `scripts/scorer/scorer_paper_p3.py` (canonical segment-local CLI) |

The paper's reported P3 numbers use the **segment-local** rule. It parses the
answer into a goal segment and a `slot: value` segment, applies the goal
criterion to the first and the value criterion to the second, and scores an
unparseable answer zero. `scripts/semantic_evaluation/audit_p3_segments.py`
reproduces the paired analysis and writes `results/p3_segment_sensitivity/summary.json`, whose
`separated_strict` and `separated_loose` fields are the reported values.
`scripts/scorer/scorer_p1_late.py` applies both criteria to the whole answer
field. The paper reports that rule as a sensitivity check, and the
`p1late_combined_rate*` fields in `results/diagnostics/update_evidence/` and in
`results/aggregates_flat.jsonl` are its output, not the reported numbers.

Use the canonical scorer for a new paper-P3 reader run:

```bash
python3 scripts/scorer/scorer_paper_p3.py \
  --probes data/probes/probes_sgd_s42_r30_n200_late_intent.jsonl \
  --reader-out results/diagnostics/update_evidence/p3_sgd_s42_reader_llama.jsonl \
  --scored-out outputs/paper_p3_scored.jsonl \
  --aggregate-out outputs/paper_p3_aggregate.json
```

It joins every input to one reader answer, rejects missing/duplicate answers,
and writes explicit `p3_joint_strict` / `p3_joint_loose` fields. It reuses the
reported parser and matching primitives; its regression check compares all
19,600 released P3 answers and a 1,600-answer raw-reader CLI run. To rescore
saved predictions directly, replace the first two arguments with
`--saved-scored results/diagnostics/update_evidence/p3_sgd_s42_scored_llama.jsonl`.
Keep reader conditions separate when combining saved files.

## Original P3 prompts and alias-exclusion sensitivity

The original saved P3 prompt asks for "the user's task in this conversation"
and the final slot value; it does **not** explicitly ask for the initial task.
The scoring reference is the normalized first sentence of the first USER turn.
We preserve that wording in the original contexts, prompts, generations, and
headline results. This limits how strongly P3 can be interpreted as an explicit
initial-goal question. See `scripts/builders/construction.py::p3_user_prompt`
and each probe's `prompt_user` for the exact original question.

The original P3 eligible pools have 452 SGD and 277 MultiWOZ examples. They
apply the final-update, dialogue-length, and late-position constraints, then
sample 200 examples per seed. They do not apply P2's old/new alias-equivalence
exclusion. The following saved-generation sensitivity applies that same P2
heuristic to the sampled P3 examples, without resampling or running a reader:

```bash
python3 scripts/audit/p3_alias_exclusion_sensitivity.py \
  --artifact . --output outputs/p3_alias_exclusion.json
```

`results/p3_alias_exclusion/summary.json` includes excluded example identifiers,
original and subset scores, source hashes, and all per-seed denominators. SGD
retains 152/147/147 examples and MultiWOZ 181/183/180 for seeds 42/43/44;
ChunkKV uses seed 42. Means average the per-seed accuracies with equal weight.
These are subsets of the original generations, not rebuilt 200-example runs.
The alias function uses normalized equality, comma-prefix equality, and
substring heuristics; it is not independent semantic annotation and can exclude
genuine distinctions. This sensitivity does not replace the original headline
tables or regenerate any content-aware selector outputs.

```bash
python3 scripts/tests/test_paper_p3.py
python3 scripts/tests/test_p3_alias_sensitivity.py \
  --expected results/p3_alias_exclusion/summary.json
```

P1 strict requires two shared content tokens; loose requires one. P2 strict
uses normalized equality; loose accepts the complete normalized reference at
phrase boundaries. P3 requires a shared initial-goal token in the goal segment
and current-value matching in the value segment, under its strict or loose
rule. These scores measure lexical recovery.

P1 uses the question “What was the user's initial goal at the beginning of this
conversation?” and is evaluated with Llama at r=0.30. P2/P3 retain their own
questions and scopes. The Mistral replication evaluates P2/P3; the corpus and
language replications evaluate P2. Saved prompts define each condition exactly.

The turn-selector budget is `max(1, round(n_turns * r))` in the main positional
pool. The LLMLingua selector and archived H2O/MMR implementations use
`max(1, floor(n_turns * r))`, so nominally equal ratios can differ by one turn.
KV methods retain fraction `r` of cache entries. Fixed compressed contexts
for every reported method are included. FMTS has its own floor-rounded budget,
documented in its [instructions](scripts/fmts_refpool/README.md). Token retention,
turn retention, and cache retention have different physical units.

Main full-context and positional inputs include turn/speaker markers, whereas
the content-aware proxy contexts concatenate the retained raw turn text.
Corrected LLMLingua preserves its original rendering. Together with the
floor/round distinction, this means the reported comparison concerns the
released pipelines under nominal budgets; it does not isolate selector logic
from input formatting. The fixed contexts make these differences inspectable.

## Construct probes and run a reader

All paths are relative to this repository; no author workspace is required.

```bash
python3 scripts/builders/build_probes.py --dataset sgd --probe p2 \
  --ratio 0.30 --seed 42 --out outputs/sgd_p2.jsonl

python3 -m pip install -r requirements.txt
python3 scripts/reader/reader_worker_local.py --help
```

The builder samples the included ordered candidate pools in `data/construction/`
and uses the supplied slot-update annotations. It reproduces the paper's
membership, reference answers, and prompts from the fixed compressor outputs.
`scripts/builders/construction.py` supplies prompt, alias, evidence-matching,
and positional-selector functions. `build_deletions.py` reconstructs the paired
P2 interventions from the released full contexts and evidence fields.

For new reader runs, use the saved `data/probes/` files to preserve all task
identifiers and metadata. The reader requires access to the selected model and
suitable hardware. Run `--help` on the local, ChunkKV, or additional KV worker
under `scripts/reader/` for its arguments. Install `kvpress` for the additional
KV methods. Model weights and account credentials are external dependencies.

Reader settings:

| Comparison | Output tokens | Attention |
|---|---:|---|
| Main Llama P1/P2 | 96 | eager |
| Llama P3 and paired deletion | 128 | SDPA |
| ChunkKV P3 | 128 | eager |
| Mistral P2 | 128 | SDPA |
| Mistral P3 | 128 | eager |

Greedy decoding is used for the local readers. The usual input cap is 7,168
tokens; LongMemEval-KU uses 32,768 with left truncation. Experiment-specific
configurations and model revisions accompany the saved outputs.

## Other corpora and free-form questions

The RiSAWOZ test input and LongMemEval knowledge-update source subset are under
`data/sources/`. To build the all-eligible replication sets:

```bash
python3 scripts/seed_robustness/build_seed_robustness.py \
  --risawoz-src data/sources/risawoz_test.json \
  --longmemeval-src data/sources/longmemeval_knowledge_updates.json \
  --risawoz-out outputs/risawoz.jsonl --longmemeval-out outputs/lme_ku.jsonl
```

FMTS includes its three question sets, the compression implementation, exact
answer prompts, an answer runner, and saved execution evidence. See
[the FMTS instructions](scripts/fmts_refpool/README.md).

## Layout

| Directory | Contents |
|---|---|
| `data/probes/` | Evaluation-ready reader inputs |
| `data/construction/` | Ordered SGD/MultiWOZ candidate context pools |
| `data/fmts/`, `data/sources/` | Included construction inputs for other experiments |
| `scripts/` | Builders, readers, scorers, and validation |
| `results/`, `batches/` | Saved outputs and paper result evidence |
| `LICENSES/` | Source-dataset terms |
| `croissant.json`, `manifest.json` | Record metadata and release file hashes |

## Scope and licensing

P1 uses the first sentence of the first user turn. P2/P3 use state annotations
from task-oriented dialogues. The main readers are Llama-3.1-8B-Instruct and
Mistral-7B-Instruct. Lexical scoring can reject semantic paraphrases and accept
incomplete answers; strict, loose, and token-F1 results document that behavior.

Source data retain their respective licenses in `LICENSES/`. Code and other
original release materials use the root `LICENSE`; vendored components retain
their own license headers. Cite TPBench and the source corpora when using the
derived probes. API-backed experiments require the caller's own model access;
no credentials are included.

## Auxiliary semantic and segment-local evaluation

See [SEMANTIC_EVALUATION.md](SEMANTIC_EVALUATION.md) for the fixed AI-judge protocol, external human-label validation, P1 semantic results, P3 segment-local scoring, and the matched first-user-plus-recent baseline.

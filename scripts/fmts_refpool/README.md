# Free-form dialogue questions (FMTS)

All three input sets are included in `data/fmts/`: 100 synthetic indirect
questions, 200 MultiWOZ questions, and 200 SGD questions. Paths below are
relative to the repository root. The compression implementation is in `core/`.

## Generate compressed contexts

```bash
python3 -m pip install -r requirements-fmts.txt
python3 scripts/fmts_refpool/run_compression.py \
  --fmts data/fmts/fmts_indirect_seed42.jsonl \
  --ratio 0.30 --seed 42 --tag indirect_seed42_refpool_r030 \
  --out_dir outputs/fmts
```

Repeat with `fmts_multiwoz_scope.jsonl` and `fmts_sgd_scope.jsonl`, and ratios
0.10 and 0.30. Use `--only full_context recency random_seed42 first_n
uniform_stride` for model-free positional selection. This needs only NumPy
and rouge-score. H2O/MMR use GPT-2 features; LLMLingua-2 uses
`microsoft/llmlingua-2-xlm-roberta-large-meetingbank`.

FMTS keeps `max(1, int(n_turns * r))` turns. The main SGD/MultiWOZ positional
selectors use rounding instead. The fixed contexts underlying each experiment
are distributed with its results. FMTS joins retained turn texts with spaces.

## Answer generation and scoring

`results/fmts_execution/protocol.json` supplies the exact question template,
instruction text, output schema, and model identifier. The saved reader is
`gpt-5.4-mini`, accessed through Codex CLI. The instruction text contains the
word "judge", but the model generates an answer from the compressed context;
it does not grade another model's answer. The saved run records no immutable
model revision or explicit temperature.

```bash
# Check inputs without a model call:
python3 scripts/fmts_refpool/run_answers.py --dry-run --limit 1
# Run after installing/authenticating Codex CLI and confirming model access:
python3 scripts/fmts_refpool/run_answers.py --limit 1 --out outputs/fmts_answers.jsonl
```

This command makes model calls only when `--dry-run` is absent. It uses the
included tasks and protocol. The executable must support the options listed
by `codex exec --help`; the model must be available to the caller's account.
Fresh model responses can differ from saved responses.

The main FMTS scorer applies the experiment's case-insensitive string rule.
The separate synthetic deletion test uses its documented phrase matcher.
To reproduce all six reported answer-accuracy aggregates without model calls:

```bash
python3 scripts/audit/validate_saved_readers.py
```

Saved contexts and aggregate results are in `results/fmts_refpool/`.
Matched references and answers are in `batches/fmts_refpool/`.
The current 7,000-task execution evidence is in `results/fmts_execution/`.
The LLMLingua reader condition is excluded because its historical contexts
were produced by a label-parsing defect. Its corrected deterministic selection
and retention diagnostics are included, but new reader answers were not generated.
All 8,000 original task/answer records and original contexts are preserved under
`results/historical/fmts_legacy_20260929/`, explicitly marked historical.
The current default answer-generation tasks and score summaries contain seven
conditions (full context plus six selectors); they cannot accidentally pair old
LLMLingua answers with corrected contexts.

The deterministic `qa_exact`/`b_qa_exact` metric is a lexical retention heuristic:
case-insensitive substring match or answer-token set overlap above 0.5.
It is not model-generated QA accuracy or exact answer-span retention.
`results/fmts_refpool/corrected_diagnostics.json` additionally reports exact
substring retention for every method. LLMLingua selection uses the mean binary
word-retention label, not a soft token importance score.

Verify corrected deterministic diagnostics and current/historical separation:

```bash
python3 scripts/audit/validate_fmts_correction.py
```

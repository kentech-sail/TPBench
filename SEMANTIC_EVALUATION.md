# Semantic P1 evaluation and segment-local P3 scoring

## Initial-goal semantic equivalence

P1 asks: “What was the user's initial goal at the beginning of this conversation?”
The current evaluation covers 8,400 Llama answers: SGD and MultiWOZ, retained turn
fraction 0.30, seeds 42/43/44, and seven conditions including full context.
The 3,049 distinct reference-answer pairs each have one cached judgment. The
analysis key expands judgments to the sampled examples used in lexical scoring.
Method identities, lexical scores, and expected rankings are absent from judge inputs.

The corrected LLMLingua-derived condition is excluded because new judgments
for its changed answers are unavailable. This omission is based on evaluation
coverage, not its scores. The primary lexical evaluation retains all seven
selectors. The 317 missing distinct reference-answer pairs are released as
pending inputs, not completed judgments; cached judgments are reused only for
exactly identical reference-answer pairs. Scope and counts are recorded in
`data/semantic_evaluation/p1_scope.json`.

The original 9,600-answer / 3,500-pair evidence is preserved under
`results/historical/semantic_p1_legacy_20260929/`, with its original key,
inputs, responses, labels, settings, and summary. It reproduces the historical
release and does not validate corrected LLMLingua answers.

The judge is `gpt-5.4-mini-2026-03-17`, temperature 0, reasoning effort `none`,
and a 512-token output limit. Full equivalence requires preservation of the
action, target, and explicit task-relevant constraints, allowing paraphrase.
Only `equivalent` contributes to the reported score; uncertain judgments receive
no credit. Other response annotations are not independently reported metrics.
Exact model requests, generated response content, labels, prompts, and settings
are included. Account and billing metadata are not part of the research artifact.

## External human-rating agreement

Answer Equivalence (Bulian et al., 2022) supplies human ratings for factual QA:
https://github.com/google-research-datasets/answer-equivalence-dataset
at commit `5032298f3fe69bebab4a013de5606231a948cebc` (Apache-2.0).
Fixed label-independent samples contain 200 development and 600 test pairs.
Majority voting leaves 199 and 595 resolved pairs. The same judge and
equivalence prompt obtain test agreement 0.837, balanced accuracy 0.841,
and a 0.067 paired accuracy gain over token-F1 >= 0.5.
AE supplies a question and passage; TPBench supplies its initial-goal reference.
This is external validation of an AI judge, not human annotation of TPBench.

## Recompute without API calls

```bash
python3 scripts/semantic_evaluation/aggregate_equivalence.py --output outputs/p1_semantic.json
python3 scripts/semantic_evaluation/aggregate_ae_judge.py --inputs data/semantic_evaluation/ae_test_inputs.jsonl --key data/semantic_evaluation/ae_test_key.jsonl --judgments results/semantic_evaluation/ae_test/judgments.jsonl --model-record results/semantic_evaluation/ae_test/model_record.json --output outputs/ae_test_metrics.json
python3 scripts/tests/test_reader_evaluation.py
```

P1 confidence intervals use 5,000 source-dialogue-cluster bootstrap resamples,
preserving repeated examples across seeds. AE uses 10,000 question-cluster
resamples. Intervals condition on saved judgments; systematic judge error is
not represented by sampling intervals.

## Segment-local P3 scoring

The analysis covers 19,600 saved answers across Llama, Mistral, and ChunkKV.
The parser separates `goal; slot: value` into goal and value, preserving colons
inside times, and applies lexical criteria within the relevant segment.
Parsing failures are reported separately from cross-segment matching losses.

```bash
python3 scripts/semantic_evaluation/audit_p3_segments.py --artifact . --output outputs/p3_segments
```

## Matched first-user-plus-recent comparison

`data/reader_evaluation/` contains reader requests and the example-to-request
key. `results/reader_evaluation/` contains complete saved answers, runtime
settings, and deterministic scores. The position-only baseline retains the
first USER turn and latest k-1 other turns, where k=max(1, round(n_turns*0.30)).
P1 includes eight reference conditions and this baseline. P2/P3 compare the
baseline with full context, earliest turns, recency, and uniform stride.
The matched P3 table uses segment-local strict scores.

```bash
python3 scripts/reader_evaluation/score_answers.py --artifact . --key data/reader_evaluation/key.jsonl --answers results/reader_evaluation/answers.jsonl --output outputs/reader_scores
python3 scripts/reader_evaluation/run_reader.py --help
```

New GPU inference uses a local copy of the recorded Llama snapshot. Model
weights and API access are external dependencies; saved-evidence checks need neither.

# LLMLingua correction workflow

Run from the release root. CPU checks use the released answers and word labels:

```bash
python3 scripts/corrections/validate_llmlingua_selections.py
python3 scripts/corrections/validate_reader_corrections.py
```

Primary requests, source-to-request keys, and explicitly marked original-response
reuse are under `data/llmlingua_correction/{llama,mistral}/`. Fresh primary answers,
expanded scores, and model/runtime records are under the matching
`results/llmlingua_correction/` directories. No model weights are distributed.

To rerun Llama inference, provision the recorded pinned checkpoint and software
versions, set `LLAMA_MODEL_PATH` to that local snapshot, and use separate outputs:

```bash
python3 scripts/reader_evaluation/run_reader.py --inputs data/llmlingua_correction/llama/inputs_p1_batch8.jsonl --output outputs/llmlingua_p1 --model-path "$LLAMA_MODEL_PATH" --attn eager --batch-size 8 --batch-tokens 8000
python3 scripts/reader_evaluation/run_reader.py --inputs data/llmlingua_correction/llama/inputs_p2_serial.jsonl --output outputs/llmlingua_p2 --model-path "$LLAMA_MODEL_PATH" --attn eager --batch-size 1 --batch-tokens 6000
python3 scripts/reader_evaluation/run_reader.py --inputs data/llmlingua_correction/llama/inputs.jsonl --output outputs/llmlingua_p3 --model-path "$LLAMA_MODEL_PATH" --attn sdpa --batch-size 1 --batch-tokens 6000
```

Mistral uses `run_mistral_reader.py`, its pinned `MISTRAL_MODEL_PATH`, and
`data/llmlingua_correction/mistral/inputs.jsonl`: P2 is SDPA and P3 eager, both
serial with 128 output tokens. The original Mistral revision was unrecorded;
the corrected revision is explicit and is not claimed to match it. Runtime
differences from historical evidence are detailed in
`results/llmlingua_correction/provenance.json`.

`score_main_reader.py` recomputes the lexical scores and identifies exact-pair
semantic cache reuse; `--help` lists all required paths. When `--artifact` is
this corrected release, it reads the archived complete original semantic cache.
Missing pairs remain pending. The whole corrected LLMLingua condition is omitted
from semantic P1 and paid FMTS reader headlines; absent coverage is not scored zero.

For request preparation from scratch, `prepare_main_reader.py` and
`prepare_mistral_reader.py` require the **original** artifact as `--artifact`,
from Hugging Face revision `11dcc75532f5d949a09b67f9c921eb254f3f1604`.
Pass this release's `data/llmlingua_correction/context_comparisons.jsonl` as
`--contexts`. This explicit original dependency preserves the original questions,
membership, and cache lineage without duplicating the complete historical release.

After integrating completed reader corrections, `refresh_p3_outputs.py` refreshes
the canonical segment-local fields, whole-field sensitivity, alias-exclusion
subset, and flattened aggregate table. It does not change prompts or run models.

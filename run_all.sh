#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONDONTWRITEBYTECODE=1
PYTHON_CMD="${PYTHON:-python3}"
"$PYTHON_CMD" scripts/check_artifact.py
"$PYTHON_CMD" scripts/audit/validate_saved_readers.py
"$PYTHON_CMD" scripts/audit/validate_fmts_correction.py
"$PYTHON_CMD" scripts/validate_corpus_scorers.py
"$PYTHON_CMD" scripts/seed_robustness/validate_seed_robustness.py
"$PYTHON_CMD" scripts/tests/test_construction.py
"$PYTHON_CMD" scripts/tests/test_corpus_construction.py
"$PYTHON_CMD" scripts/tests/test_fmts.py
"$PYTHON_CMD" scripts/tests/test_reader_evaluation.py
TMP_P3="$(mktemp -d)"
"$PYTHON_CMD" scripts/semantic_evaluation/audit_p3_segments.py --artifact . --output "$TMP_P3" > /dev/null
"$PYTHON_CMD" -c "import json,sys; a=json.load(open(sys.argv[1])); b=json.load(open(sys.argv[2])); assert a==b, 'P3 segment-local summary does not regenerate'; print('OK: P3 segment-local scores regenerate')" "$TMP_P3/summary.json" results/p3_segment_sensitivity/summary.json
rm -rf "$TMP_P3"
"$PYTHON_CMD" scripts/audit/retention_vs_reading.py > /dev/null
"$PYTHON_CMD" scripts/tests/test_paper_p3.py
"$PYTHON_CMD" scripts/tests/test_p3_alias_sensitivity.py --expected results/p3_alias_exclusion/summary.json
"$PYTHON_CMD" scripts/tests/test_llmlingua_labels.py
"$PYTHON_CMD" scripts/corrections/validate_llmlingua_selections.py
"$PYTHON_CMD" scripts/corrections/validate_reader_corrections.py
echo "OK: run_all complete"

# Experiment stages

This package owns dataset preparation, measurement caches, prediction seals and evaluation. It imports `myproject`; the detector library does not depend on experiments. Run `python -m experiments --help` or `token-experiment --help` for all stages. `r4-experiment` is an alias. Imports from the former `myproject_experiment` package must change to `experiments`.

## Text workflow

```bash
token-experiment prepare-dataset --name ragtruth --data /path/to/RAGTruth/dataset \
  --split test --task QA --output outputs/qa/inputs
token-experiment predict-dataset --input outputs/qa/inputs \
  --model /path/to/Meta-Llama-3.1-8B-Instruct --task QA \
  --method shared --output outputs/qa/scores
token-experiment evaluate-dataset --scores outputs/qa/scores \
  --label-granularity token --output outputs/qa/evaluation
```

Preparation separates inputs, annotations and their manifest. Prediction validates text evidence without opening annotations, checkpoints responses, and seals completion. Evaluation reads labels after checking prediction artifacts. Resumption requires identical inputs, model, method and source identities. Use a new output directory when changing them.

`--method r4` or explicit R4 references select the historical calibrated detector. Shared methods do not consume those distributions. RAGTruth character spans and HalluRAG sentence-derived token labels retain different meanings. ClaimDB preparation does not establish database-evidence token detection support.

## Cached observations and diagnostics

`transfer collect`, `transfer score` and `transfer evaluate` separate native observation, frozen scoring and label joins. Other transfer stages expose recovered research diagnostics, including full-gap observation, native controls and span resolution. Check each stage's help before use; model execution requires local weights and explicit protocols where requested.

Reusable candidates live in `experiments/research/`, exposed by `r4-research`. They are research implementations, not verified improvements over R4. Keep generated files under `outputs/<run-id>/` and preserve completed runs.

## Frozen verification

```bash
r4-experiment verify --packs /path/to/historical/packs \
  --baseline /path/to/r04_tempered_continuity_v1 --output outputs/verification.json
```

This CPU command needs external historical artifacts and checks numerical identity, not a new benchmark result. `tools/verify_frozen.py` delegates to this implementation. The [ledger](EXPERIMENTS.md) preserves remote historical run claims; restoration has not rerun the remote GPU experiments or recovered every referenced output.

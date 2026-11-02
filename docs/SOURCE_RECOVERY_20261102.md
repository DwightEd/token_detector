# Remote source integration - 2026-11-02

This update integrates the remote v7 installable source snapshot into the maintained Git history. It was built on 2026-10-31 and recovered through the owner's authenticated storage portal. The original backup remains at commit `6b8a6d4e98c2b27aed8f63b749450d99d88d4790` on `backup/remote-20261031-core`.

This page describes integration baseline `1210120`. The subsequent [theory/source audit](source-reading/README.md) adds post-seal evaluation diagnostics in `experiments/local_evaluation.py` and `experiments/fullgap_dataset.py`; their current hashes are explicitly recorded in `SOURCE_RECOVERY_MANIFEST.json.integration_changes`. Original recovered hashes remain unchanged as provenance.

## Source identity

The wheel SHA256 is `d31e3c27d8d6b7a0f3d3e5d40679da79daa198750214d8d8ab264f20391cdb74`, matching the remote release receipt. All wheel RECORD entries and 88 receipt-bound source/resource files were checked before integration. Three R4 reference NPZs retain their original bytes.

`SOURCE_RECOVERY_MANIFEST.json` records the recovered source hashes. Documentation, retained regression tests and packaging integration changes are separate from that snapshot. Any portability corrections to recovered files are recorded below rather than represented as unchanged remote bytes.

## Integration

- Migrate `myproject_experiment` imports and entry points to `experiments`.
- Restore shared inference, explicit R4 and research kernels without claiming factual calibration or a new detector breakthrough.
- Retain existing regressions and select `method="r4"` explicitly in legacy R4 API fixtures.
- Keep Python 3.11+ and the remote dependency pins; check an installed wheel outside the checkout.
- Update current documentation while retaining the dated historical validation report.

The remote receipt records 724 repository tests passed, one skipped, nine subtests passed and 147 installed-package tests passed. Those are historical outcomes, not tests rerun by this integration. The full remote test suite and raw outputs were not recovered. Current checks and review findings are recorded below.

## Portability correction

`experiments.data.read_json` now explicitly uses UTF-8, matching the existing writer. On Windows with a GBK default locale, reading metadata containing Chinese text and a check mark raised UnicodeDecodeError. A regression reproduced the failure before the one-line fix. This changes metadata decoding, not detector arithmetic, references or source-token identities. The original source hash and corrected integration hash remain separately recorded in the manifest.

## Current integration validation

Run on Windows with Python 3.13.9, NumPy 2.4.6, SciPy 1.17.1, Torch 2.8.0+cpu, Transformers 4.57.1, huggingface-hub 0.36.2, scikit-learn 1.9.0 and pytest 9.1.1. CUDA was disabled and model downloads were disabled. Tiny random CPU Llama tests exercise actual forwards without loading the research model.

- Source checkout: **122 passed, 1 skipped, 9 subtests passed**.
- Newly built wheel installed and tested from outside the checkout: **122 passed, 1 skipped, 9 subtests passed**.
- Seven installed CLI/module help invocations passed, including the retained R4 aliases and the research entry point.
- All 88 installed source/resource files match the integrated checkout; no stale extra modules were found in the three installed packages.
- All 87 unmodified recovered files match their original source hashes. The UTF-8 fix has a separately recorded hash; all three frozen reference arrays remain unchanged.
- Independent finite-state enumeration checks shared Beta-mixture log odds; existing tests retain tiny-model observation checks, token offsets, annotation isolation, sealing and resumption coverage.

The skipped test requires optional external frozen HalluRAG TEST artifacts, which were not installed in this environment.

The integration wheel SHA256 is `02c3af300ee0454823f5e56e064461bd7213195a2db44eb7e7a4aa51134d4169`. The complete remote 724-test suite, historical GPU experiments and original dataset-wide score parity were not rerun.

## Integration review

A fresh review pass by the implementing agent checked the default-method change, package/CLI migration, dependency direction, installed-file identities, reference preservation and UTF-8 metadata behavior. It identified and fixed the reproduced Windows decoding failure. The migrated R4 tests explicitly select R4; the new shared default has its own no-FIT text-API regression. This was an engineering integration review, not an independent scientific review or manuscript re-evaluation.

## Subsequent performance work, recorded 2026-11-02

The maintained comparison now contains **271 recovered historical result records**. Its six primary tasks are RAGTruth QA, Summary and Data2txt, FRED FinQA and TATQA, and the released HalluRAG answerable/target-available 891-response group. R4 meets AUROC 0.8 at 2/6 point estimates and V3 at 3/6; their descriptive per-task envelope meets it at 4/6. The envelope is not a tested detector. HalluRAG pooled and unavailable strata remain separately reported, and the recovered complete-TEST R4 report is kept separate from the older 132-response VAL/DEV result. See [performance review](source-reading/PERFORMANCE_REVIEW.md) and [machine-readable task results](results/r4_v3_task_summary.json).

The current goal allows modular unsupervised hybrids and task-specific configurations. Methods, parameters, score populations and complete predictions must be frozen before final ground-truth evaluation; previously inspected populations remain exploratory. Existing untouched annotated evaluation data are valid confirmation sources.

The post-recovery CPU utilities now verify native cache identities, export unchanged R4/V3 scores, seal a fixed rank baseline, and evaluate released labels afterward. All five paired tasks (2,992 responses / 448,202 tokens) were evaluated. The fixed equal-rank control was rejected: Summary AUROC was 0.7694, FinQA 0.7515, TATQA 0.7929; the all-task target remains unmet. No detector default was changed. Current work is [source-grounded FN/FP analysis](source-reading/FNFP_MECHANISM.md).

Four added experimental runtime modules, their central CLI wiring and three test files are outside the original 88-file receipt. The separate final wheel contains exactly **92 modules/resources** matching the checkout; **57 targeted synthetic tests passed outside the checkout**, and all three added CLI help checks passed. The final wheel SHA256 is `6bcfd5ac2960ae648736e8e9247636406fb04d297971be0202de187a78daab58`. This is a separate scoped verification, not a rerun of the old full suite. Receipt: `outputs/rank-fusion-wheel-final-20261102/receipt.json`.

Recovered cache seals and response/token axes were verified independently. Full V3 HalluRAG scores and prepared data are restored; historical R4 HalluRAG arrays were deleted upstream, while its historical aggregate report survives. Source facts, character labels, score propagation and alarm-budget effects are separated in the posthoc audit. All original recovery receipts and reference arrays remain unchanged. New experimental code still requires its own line explanation, tracked in SOURCE_INVENTORY.json.

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

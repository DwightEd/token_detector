# Code organization

## Confirmed research constraint (2026-11-02)

- The user requires unsupervised detection. Ground-truth support/error labels
  may be opened only for final evaluation after freezing methods, parameters,
  thresholds, partitions and scores. Do not use them for training, tuning,
  method selection or label-based risk calibration.
- Keep previously inspected evaluation populations exploratory; they cannot
  become fresh confirmation data by renaming or resplitting them.
- Follow refine-logs/theory-grounding-20261102/USER_CONSTRAINTS.md. Preserve
  historical proposals and the immutable anchor; this confirmed user choice
  supersedes their previously pending supervised/calibration alternatives.

## Confirmed performance priorities (2026-11-02)

- Target AUROC >= 0.8 on each declared primary task. Modular hybrids and
  task-specific configurations are allowed; do not impose a single probability
  model or restrict improvement to V3. The unsupervised/final-truth rule above
  remains active.
- Use released HalluRAG answerable/target-available metadata for the primary
  comparison requested by the user; retain pooled and unavailable results.
  Existing annotated datasets are valid evaluation sources.
- Keep actual runs, recovered results, proposed modules and unresolved resource
  needs explicit. Do not call a per-task best-result envelope a tested detector.
- The user rejected blind score weighting as a method-design direction. Prioritize
  concrete source-grounded FN/FP explanations and module-level diagnostics before
  proposing new combinations. The equal-rank CPU control is a rejected experiment,
  not the mainline. Error labels in posthoc evaluation must not become tuning inputs.

## Implementation

- This repository is the canonical token-detector workspace. Develop from its root;
  test installed packages in a clean directory to avoid unrelated same-name packages.
- Keep the callable detector library in myproject/. Put all experiment code in
  experiments/, with reusable candidate methods in experiments/research/.
- Route dataset workflows through r4-experiment and its shared stage functions;
  extend the integrated entry point instead of creating parallel one-off runners.
- Experiments may import myproject; the detector must not depend on experiments.
- Use existing shared functions instead of copying them into scripts or result folders.
- Mainline readouts and their diagnostic stages have one implementation in myproject/.
  Experiment runners call SharedDetector, Detector or their library kernels directly;
  independent numerical oracles belong in tests/audits and never produce method scores.
  Keep library calls silent, in memory, and separate from CLI/file orchestration.
- Put generated run files in outputs/<run-id>/ and keep them out of Git. Do not
  recreate deleted directories or add old-package aliases without an actual need.
- Keep dependencies in pyproject.toml. Use explicit imports and only abstractions
  with real callers; do not hardcode machine paths in algorithm source.
- Preserve public API and numerical contracts during refactors. Run relevant tests;
  verify an installed wheel outside the checkout for packaging/resource changes.
- Build wheels from a clean generated build cache. Check that installed module and
  resource hashes match the checkout exactly, including no stale extra modules.
- Request a fresh code review for substantial changes and fix verified important issues.
- Keep run commands, input/source identities, logs and actual outcomes alongside outputs;
  record experiment status in experiments/EXPERIMENTS.md without duplicating the code.

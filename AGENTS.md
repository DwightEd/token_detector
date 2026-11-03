# Code organization

## Active persistent research goal

- At the start of every research continuation, read RESEARCH_BRIEF.md and
  research-state/RESUME.md, then run `python tools/research_memory.py context`.
  Load only the relevant fact/prior-experiment records for the next question.
- The current question, facts and prior experiments live in research-state/;
  older refinement snapshots are history, not automatically active instructions.
- Do not rerun a known phenomenon, a rejected readout, prefix reset or repair
  interaction under a new name. Require a new discriminating intervention with
  different predictions under competing explanations before scientific admission.
- Exact run fingerprints prevent duplicate execution; they do not establish
  scientific novelty. A parameter/seed/code change alone does not reopen a closed
  hypothesis. Reserve jobs before launch; reuse verified results and retain failures.
- Keep scoring and evaluation as separate run stages. Changes only to bootstrap
  analysis must not trigger another model-observation/scoring run.
- At each conclusion, completed run or pivot, persist evidence, certainty scope,
  the decision and the next discriminating question before continuing.
- New model observations are authorized by the user's active research goal.
  Use isolated environments and explicit resource admission; preserve the current
  CPU environment and other user processes. No background watchdog is installed.

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
- Before pushing code, run the complete CPU suite against the installed package
  from outside the checkout, including repository-tool tests. Targeted checks
  alone do not establish that the full CI suite passes. After pushing, inspect
  the GitHub Actions run for that exact commit and fix failures before reporting
  the push as validated. Repository-tool tests must load their explicit script
  path without adding the checkout root to sys.path or masking installed imports.
  If a local full run exceeds an explicit resource/time limit, record it as
  incomplete, preserve the failure, and require the complete Linux CI run to pass
  before claiming validation; never substitute selected passing tests for it.
- Batch related source and checkpoint commits into one push when possible. A
  local frozen source commit can establish provenance before a model run;
  creating that commit does not require immediately triggering another CI run.
- Build wheels from a clean generated build cache. Check that installed module and
  resource hashes match the checkout exactly, including no stale extra modules.
- Request a fresh code review for substantial changes and fix verified important issues.
- Keep run commands, input/source identities, logs and actual outcomes alongside outputs;
  record experiment status in experiments/EXPERIMENTS.md without duplicating the code.
- On continuation inspect research-state/RESUME.md and its live local-idle job
  before starting anything. A waiting unclaimed job is still a real armed job;
  verify PID/start time/command and never duplicate it. STOP_AUTORUN cancels this
  task when the user requests stopping. Its six-hour wait is bounded, with no
  automatic retry or parameter change. Evaluate only after observation seals pass.
- H002 completed18 proposal-only calls, failed its frontend gate, and exited;
  its frozen95-file target under outputs/h002-context-20261102/installed is the
  archived compatible reader. Preserve it. H003 checkout/backend development is
  a separate release; do not rerun H002 when an evaluator/helper changes.

# Code organization

## Confirmed relation-misbinding direction (2026-11-03)

- Read research-state/RELATION_BINDING_DIRECTION.md and the current experiment plan.
  Investigate internal computations that expose source relations incorrectly
  expressed in the existing response, with localization/detection/cost value beyond
  direct text verification. Do not restrict proposals to V3 probability/smoothing.
- Separate observer probability, value attention, source fact ownership and temporal
  persistence. High attention, confident output or stable hidden association does
  not establish correct binding. Preserve duplicate values, conditions, legitimate
  computation and unknown; do not impose a global bijection.
- Before generating a new candidate module, record its scientific admission and
  independent review: new information, closed-prior differences, competing
  predictions, observation contract, stop criteria and mainline call sites.
  research_memory check/claim/finish supervises identity and evidence, not novelty.
  A proposed plan or a new fingerprint alone does not admit an experiment.
- Put experiment orchestration in experiments/ and candidates in experiments/research/.
  Reuse mainline axes, baselines, seals and evaluation. Do not treat existing pooled
  post-token attention as physical-head pre-choice relation evidence. Keep cached
  observer mechanisms distinct from original-generator causality.

## Active persistent research goal

- Read research-state/DETECTION_OBJECTIVE.md: the primary endpoint is hallucination
  risk on the EXISTING response and its original evaluation axis. Auxiliary verifier
  QA accuracy is not detector accuracy or task AUROC improvement. H015/H016 are
  closed auxiliary records; do not extend or rerun their role-QA branch as mainline.
  QA/NLI may be internal components only with an explicit original-response risk
  interface and direct natural-data detection evaluation under frozen rules.

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
- Keep the V3 library in myproject/: HallucinationDetector.extract returns
  Observations, callers explicitly save/load them, and score/predict return original
  token risks. Read myproject/README.md before changing this execution path.
- Keep dataset workflows, diagnostics, evaluation and reports in experiments/;
  historical R4 and candidate kernels/resources belong in experiments/research/.
  Read experiments/README.md when changing a research method or its imports.
- token-detect owns extract/score/predict. The multi-method API is
  experiments.inference.HallucinationDetector; r4-detect uses experiments.detect.
  This API/module migration is intentional; update callers and documentation
  directly instead of restoring old-module compatibility aliases.
- Route dataset workflows through r4-experiment and its shared stage functions;
  extend the integrated entry point instead of creating parallel one-off runners.
- Experiments may import myproject; the detector must not depend on experiments.
- Use existing shared functions instead of copying them into scripts or result folders.
- Keep V3 state inference canonical in myproject/witness.py. Experiments call its
  kernels; their R4 and other readouts have one implementation in experiments/research/.
  Independent numerical oracles belong in tests/audits and never produce method scores.
  Library calls remain silent; file persistence is explicit through Observations,
  while CLI/file orchestration stays outside the numerical kernels.
- Put generated run files in outputs/<run-id>/ and keep them out of Git. Do not
  recreate deleted directories or add old-package aliases without an actual need.
- Keep dependencies in pyproject.toml. Use explicit imports and only abstractions
  with real callers; do not hardcode machine paths in algorithm source.
- Preserve original token axes, V3 numerical contracts and frozen R4 reference bytes
  during refactors. Document deliberate API migrations and run relevant tests;
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

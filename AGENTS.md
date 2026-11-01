# Code organization

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

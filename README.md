# Recovered remote core snapshot - 2026-10-31

This branch backs up the validated v7 installable source recovered from the user's shared storage. The original upstream main branch remains the separately maintained release history.

The snapshot contains 85 Python source files and three calibrated reference arrays from `myproject`, `experiments`, and `experiments.research`, plus the remote dependency declarations. Every source/resource byte matches the recorded v7 validation receipt; the downloaded wheel's RECORD hashes were also verified. No algorithm was rewritten during recovery.

The v7 receipt records 724 remote repository tests passed, one skipped, nine subtests passed, and 147 installed-package tests passed. These are historical recorded checks. Recovery verified hashes and Python syntax; it did not rerun those tests or model inference.

## Scope

This is a core source backup, not a complete remote Git working tree. The current remote tests, all research documents, raw datasets, model weights, run outputs, complete manuscript sources, and Codex conversation history are not included. The dependency declaration's `tests` location therefore does not supply a test suite in this snapshot. The source README is preserved as `README_REMOTE_20261031.md`; its references to omitted remote artifacts may not resolve here.

Candidate and diagnostic modules are preserved as research code. Manuscript completion and internal numerical checks do not establish a new detector breakthrough or conference acceptance.

## Install

Use Python 3.11+ and an isolated environment:

```bash
python -m pip install -e .
python -m myproject --help
python -m experiments --help
```

Pinned dependencies are retained in `pyproject.toml`. Model weights and external data must be supplied separately. See `RECOVERY_MANIFEST.json` for recovered file hashes and validation-package identity.

# Experiment code

Follow the root AGENTS.md. Keep dataset splits, annotations, caches, manifests and evaluation here, with reusable candidates in research/. Import the detector library instead of duplicating its numerical methods. The library must not import experiments.

Preserve original token identities and frozen R4 arithmetic. Read labels only after freezing scores. Keep runs under outputs/<run-id>/ and preserve completed evidence. Recovering code or documentation does not authorize rerunning GPU experiments.

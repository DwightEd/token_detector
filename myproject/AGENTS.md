# V3 mainline

Read README.md before editing; follow the repository-root AGENTS.md.
Keep the visible flow: encode text, observe both source conditions, align the
original tokens, return Observations. score runs on those observations without
loading a model; predict composes extraction and scoring.

Keep model execution in model.py, alignment in measure.py, explicit NPZ persistence
in observations.py and canonical V3 state inference in witness.py. Library calls
are silent and in memory; callers opt into save/load. run.py only parses external
parameters, constructs the detector, invokes a stage and presents its result.

Preserve original token/character positions and historical V3 floating-point
contracts. Validate external text/files at their boundary, then keep the numerical
path direct. Experiments import these kernels instead of duplicating V3.

Put method selection, R4 references, research candidates, dataset orchestration,
diagnostics, evaluation and reports in experiments/. This package never imports
experiments. Keep the three public exports HallucinationDetector, Observations and
WitnessDetector; deliberate module migration uses direct caller updates rather
than old-path aliases. Engineering cleanup alone establishes no new detection result.

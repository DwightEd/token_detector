# Maintenance rules

Keep `myproject` as the directly callable, silent, in-memory R4 detector.
Keep dataset preparation, annotations, caching, exports and evaluation in
`myproject_experiment`; imports may go from experiments to the detector only.

Preserve frozen recipes, reference arrays, original token identities and scoring
numerics. Research candidates and raw historical artifacts stay outside this
repository until explicitly selected for inclusion. Do not replace the stable
baseline with an unvalidated candidate.

Use small functions and explicit imports. Avoid unused abstractions and duplicate
dependency lists; `pyproject.toml` is the installation source of truth. Test from
an installed wheel outside the checkout when changing packaging or resources.
Keep datasets, model weights, run outputs and secrets out of Git.

Before merging significant changes, request a fresh code review, check each
finding against the actual code, and fix verified critical or important issues.

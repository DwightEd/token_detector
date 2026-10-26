# Mainline code

Read README.md first. Keep only the directly callable R4 detector here.
Research plans and new methods belong in the shared research handoff; raw
artifacts and historical experiments remain at their recorded locations.

Follow the repository-root AGENTS.md engineering and code-review rules.
Prefer small functions with explicit imports. Do not add registries, managers,
compatibility exports, fallbacks or generalized schemas without an actual caller.

Library inference must remain silent and operate in memory. Return final token
scores without writing intermediate measurements, metadata or diagnostics.
Keep dataset preparation, experiment exports, evaluation and verification in
the sibling myproject_experiment package. The main package never imports it.
Refactors must preserve the original token axis, calibrated reference arrays
and R4 numerics. Keep research candidates separate from the maintained method.

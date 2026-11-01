# Mainline code

Read README.md first. Keep the callable shared detector, explicit frozen R4,
and their numerical kernels here. Research orchestration belongs in experiments;
raw artifacts and historical execution outputs stay at their recorded locations.

Follow the repository-root AGENTS.md engineering and code-review rules.
Prefer small functions with explicit imports. Do not add registries, managers,
compatibility exports, fallbacks or generalized schemas without an actual caller.

Library inference must remain silent and operate in memory. Return final token
scores without writing intermediate measurements, metadata or diagnostics.
Keep dataset preparation, experiment exports, evaluation and verification in
the sibling experiments package. The main package never imports it.
Refactors must preserve the original token axis, calibrated reference arrays
and R4 numerics. Method selection must remain explicit; adding a candidate
does not establish that it improves factual detection.

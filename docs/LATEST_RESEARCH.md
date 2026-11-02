# Research status through 2026-10-31

2026-11-02 reconciliation: **V3 / witness remains the retained historical finance candidate**, while the API defaults to V1 / shared. See the [complete recovered comparison](RESULTS_CURRENT.md), [V3 derivation](source-reading/METHOD_V3.md), and [proposed integrated design](source-reading/INTEGRATED_DESIGN.md). Later implemented candidates did not establish a replacement. The integrated proposal is unrun and does not imply promotion or universal superiority.

2026-11-02 local follow-up: [theory and source walkthrough](source-reading/README.md) now documents all 21 detector-library Python files. The new post-seal AUC pair decomposition and finite-state checks validate computation, not new detector performance. [The refinement record](../refine-logs/theory-grounding-20261102/REVIEW_SUMMARY.md) retains unresolved natural-data and novelty gates. The remote research results below remain historical.

The latest remote manuscript is **Contextual Borrowing in Token-Level Hallucination Detection: Regional Ranking and Local Resolution**: eight main-text pages and 33 pages including references and appendices. It is an unsubmitted working manuscript.

Its reported finding concerns pooled token ranking versus local error resolution. Averaging within fixed spans improved V3 AP in five of six populations while reducing within-span mixed-label AUROC to 0.5. This is a controlled diagnostic, not a detector breakthrough. A finite counterexample also shows that stronger evidence at one token can lower another token's posterior under shared inference.

The report retains 4,774 original responses across three human-span RAGTruth tasks, two financial silver-label tasks and sentence-derived HalluRAG labels. It records 14,322 backbone forwards for the pinned Qwen observer and reuse of sealed scores for later deletion experiments. These label granularities must not be treated as interchangeable independent confirmation sets.

Internal review rated scientific strength 5/10 and writing 7/10, leaning reject. Subsequent editorial revisions were not rescored. Unexposed natural human-span confirmation, partition/scale robustness and stronger independent comparisons remain gaps. Numerical consistency and manuscript completeness do not establish acceptance or fulfillment of the original detection goals.

Recovered code includes shared inference, explicit frozen R4, persistence/witness candidates and span-resolution diagnostics. The [experiment ledger](../experiments/EXPERIMENTS.md) preserves the development record. Full raw outputs and complete manuscript compilation inputs are not restored into this repository.

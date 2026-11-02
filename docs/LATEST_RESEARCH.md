# Research status through 2026-10-31

2026-11-02 最新：已直接下载并校验五任务 2,992 答/448,202 token 的配对缓存，以及 HalluRAG V3 1,782 答/69,483 token。当前工作转向[完整 FN/FP 机制统计](source-reading/FNFP_MECHANISM.md)及[来源逐例核验](source-reading/FNFP_CASES.md)。固定等权实验失败，未采用为主线；目标 AUROC≥0.8 仍未完成。


2026-11-02 recovered-result update: the complete historical R4 HalluRAG TEST report is recovered, including answerable AUROC/AP **0.807066/0.518830** versus later V3 **0.7669/0.4561**. Old R4 **0.472617/0.287484** was VAL DEV132, not this TEST. See [task comparison, actual FN/FP and modular optimization target](source-reading/PERFORMANCE_REVIEW.md). The user permits multiple modules and no longer requires one probabilistic architecture; unsupervised/final-truth-only constraints remain.

2026-11-02 user correction: [natural semantic relation qualification cannot be inferred from unlabeled structural checks](source-reading/UNLABELLED_LIMITS.md). Relation perturbations are deferred from the default research line; V3 remains the empirical anchor. The next proposed study measures operational source/history prediction effects, without promoting them to factuality scores or asserting a new detector.

2026-11-02 reconciliation: **V3 / witness remains the retained historical finance candidate**, while the API defaults to V1 / shared. See the [complete recovered comparison](RESULTS_CURRENT.md), [V3 derivation](source-reading/METHOD_V3.md), and [proposed integrated design](source-reading/INTEGRATED_DESIGN.md). Later implemented candidates did not establish a replacement. The integrated proposal is unrun and does not imply promotion or universal superiority.

2026-11-02 local follow-up: [theory and source walkthrough](source-reading/README.md) now documents all 21 detector-library Python files. The new post-seal AUC pair decomposition and finite-state checks validate computation, not new detector performance. [The refinement record](../refine-logs/theory-grounding-20261102/REVIEW_SUMMARY.md) retains unresolved natural-data and novelty gates. The remote research results below remain historical.

The latest remote manuscript is **Contextual Borrowing in Token-Level Hallucination Detection: Regional Ranking and Local Resolution**: eight main-text pages and 33 pages including references and appendices. It is an unsubmitted working manuscript.

Its reported finding concerns pooled token ranking versus local error resolution. Averaging within fixed spans improved V3 AP in five of six populations while reducing within-span mixed-label AUROC to 0.5. This is a controlled diagnostic, not a detector breakthrough. A finite counterexample also shows that stronger evidence at one token can lower another token's posterior under shared inference.

The report retains 4,774 original responses across three human-span RAGTruth tasks, two financial silver-label tasks and sentence-derived HalluRAG labels. It records 14,322 backbone forwards for the pinned Qwen observer and reuse of sealed scores for later deletion experiments. These label granularities must not be treated as interchangeable independent confirmation sets.

Internal review rated scientific strength 5/10 and writing 7/10, leaning reject. Subsequent editorial revisions were not rescored. Unexposed natural human-span confirmation, partition/scale robustness and stronger independent comparisons remain gaps. Numerical consistency and manuscript completeness do not establish acceptance or fulfillment of the original detection goals.

Recovered code includes shared inference, explicit frozen R4, persistence/witness candidates and span-resolution diagnostics. The [experiment ledger](../experiments/EXPERIMENTS.md) preserves the development record. Full raw outputs and complete manuscript compilation inputs are not restored into this repository.

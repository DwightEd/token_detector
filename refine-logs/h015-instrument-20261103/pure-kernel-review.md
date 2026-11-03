# Independent pure-kernel review

Verdict: PASS for the stated pure mathematical/structural contract. No backend, real data, H013 raw observation, model or GPU was used in this review.

Reviewed experiments/research/completion_trie.py SHA256: 8b07e3da8476883dc062e49fd2df87485076976792f9443ba42f6bef76fd213b. The author reports 27 targeted synthetic tests passed in 4.40 seconds; this independent review did not repeat them.

The eighteen-field completion framing is checked structurally, including full literal candidate tokens plus terminal EOS, prefix-only causal positions, exact counts and common prefix/EOS identity. Candidate token aliases are rejected. Complete context IDs identify one node; shared outgoing token edges reuse that node. A short candidate ending with EOS and a longer continuation use two different edges of the same node. EOS cannot occur within candidate tokens, so terminal candidate codes are prefix-free.

The scorer rebuilds the canonical trie before accepting observations. It requires exactly one complete finite full-vocabulary row per expected node and checks complete population, unique contexts, identical vocabulary size and all prefix/edge token bounds before scoring. Stable edge log probabilities use shifted logits and shifted log-sum-exp directly. The separately reported absolute log-normalizer is not subtracted to recover edge probabilities, avoiding cancellation under a large common offset.

The selected-child and candidate-pool mass bounds follow from a single normalized distribution per node in exact arithmetic and prefix-free path induction. The implementation additionally enforces its predeclared strict finite-precision gates. It rejects small positive numerical mass violations rather than clipping them; valid mathematical distributions can therefore be rejected by this engineering policy. Absolute probability underflow is retained alongside finite log values and separately normalized conditional candidate mass.

Returned data retain each needed edge's raw logit, stable log probability, node normalizer decomposition and complete candidate path. Full vectors are not returned. A later CPU reader of only those retained values can verify internal arithmetic and backend provenance, but cannot independently recompute an unsaved full-vocabulary normalizer. The design states this attestation limit and does not claim exact replay of an ideal LM or the old variable-length forwards.

This pass is not backend admission, measured resource readiness, a speed result, a rerun of the failed H013 attempt, or evidence of QA/factuality performance. A future backend must freeze the model/runtime identity and guarantee that each unique causal context supplies exactly one shared row; padding alone does not establish that guarantee.

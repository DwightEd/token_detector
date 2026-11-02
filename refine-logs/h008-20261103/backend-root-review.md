# H008 completion backend independent review

Root independently read answer_completion.py and the existing parent HFGenerator after the author completed 22 CPU/fake-model checks. No real model or GPU forward was used for this review.

The causal indexing is correct: with P prefix tokens and K literal candidate tokens, the model receives P+K input tokens and returns the last K+1 full-vocabulary rows. Those positions P−1 through P+K−1 predict every candidate token, beginning with the first token, then one fixed tokenizer EOS. The terminal completion EOS is a target rather than appended input; EOS tokens closing earlier chat messages remain valid.

The complete encoded prefix is checked against the chat-template tokenization; the candidate suffix is taken from the full prefix+candidate encoding, round-trips exactly, and never undergoes trimming. Unstable prefix boundaries, candidate-internal EOS, missing EOS, excessive input, wrong logits shape, nonfinite logits, or out-of-vocabulary targets reject explicitly.

One admitted forward uses no KV cache or generated text. Full-vocabulary log-sum-exp is evaluated in float64 for each required row, raw selected logits and normalizers are retained, and the joint score includes every target log probability. No candidate-length correction, A/B/C verbalizer, temperature fitting, or target-response value enters this scorer. Candidate normalization belongs to the separately frozen source-only kernel.

The old 3B/NF4 runtime, asset hashes, 3 GiB allocator policy and model-load resource guards are inherited. The pure completion_metadata function can run with a CPU fast tokenizer and never calls the parent's GPU preparation. The two-candidate probabilities remain conditional on the restricted complete-name-plus-EOS domain, not calibrated truth probabilities. The future actual32 run is still needed; fake-model checks are not task efficacy.

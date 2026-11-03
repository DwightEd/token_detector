# H014 implementation and targeted validation

Status: READY for root's installed-release admission and CPU-only execution. No real H011 observations, evaluation packet, model, GPU, or natural task labels were opened by this implementation/test work.

API: experiments.masked_assignment.analyze(inputs, observations, model_path, reconstruction_review, output, protocol). The protocol is the exact DEFAULT_PROTOCOL mapping in that module. The CLI remains root-owned.

Frozen implementation byte identities:

- experiments/masked_assignment.py: ae9f8bc5a9f48d6d09becb5653cc26ae92de6582d584f98385aa4971f6ba5fa3
- tests/test_masked_assignment.py: f368d136c9d2ad3ab27476aed193722ac71996786d46265c3b7138ec9bf066b6
- DESIGN.md: 760329ed422f3bb76bac9c4134d37e05756431cb8c8220a80a09cafd866a5c0e

The independent test/review agent ran .venv/Scripts/python.exe -m pytest tests/test_masked_assignment.py -q: 13 passed in 252.44 seconds. Only this new suite ran; its observations, references, model assets and tokenizer are synthetic fixtures. No older suite or model observation was repeated.

The tests verify all 24 source/burden matrices, 48 queries, 96 cached completions and 48 original cases; full score/plan/protocol/source freezing before any H011 reference open; whole donor namespace and seal-key admission before payload hashing; rejection of resealed changes to old raw observations, query references and case readouts; late reference-seal and output mutation revocation; exact source and original-response suffix grouping; direct reuse of the existing assignment kernel; retention of independent same-name collapse when joint is correct, wrong or tied; and explicit unknown risk one for a non-bijective response mapping.

The independent static review found no material blocker. Its compatibility claim is deliberate reconstruction: pin the unchanged H011 collector/compiler/numerical/helper set plus its masked evaluator, rebuild the complete donor through the current shared CPU validator, and reproduce all four sealed H011 outputs exactly. H013's addition of keyword forwarding changed the shared CPU file bytes, so this code does not claim that whole file remains byte-identical to its H011 version. H014's own complete source identity pins and continuously guards the current validator bytes.

Only after all raw-score matrices and both fixed readouts are frozen does the analyzer admit, pin and read the twelve-member H011 CPU packet. It preserves the initial donor and reference seals, all nine model assets, all old summaries and costs, both readouts and every burden. Completion and output seal are revoked on a late integrity failure.

There are zero new language observations, offsets, parameter choices, temperature fitting or winner selection. The assignment masses are conditional energy readouts under an explicitly admitted toy bijection, not calibrated truth probabilities. The existing variable-length separate-forward instrument's numerical probability-tree coherence is a separate engineering limitation; this cached diagnostic does not certify it.

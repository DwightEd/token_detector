# H016 fresh static code review

Decision: no verified important blocker in the reviewed core qualification, evaluation, pure four-arm kernel, CLI integration, or synthetic tests. One runner provenance blocker needs correction before the admitted model attempt.

Scope: frozen 16 b0 cases / eight sources; compound versus atomic question bundles crossed with independent-row maximum-slot risk versus admitted-bijection joint assignment mass. This is an exposed toy instrument diagnosis. No natural-task performance claim is supported or made.

## Verified important runner issue

`outputs/h016-paired-scope-20261103/run_once.py` records `manifest['model_identity']['hardware']` without checking the live CUDA device against it. Its call path through HFPrefixCompletion/HFGenerator also lacks this check. The pinned H015 helper explicitly verifies `torch.cuda.get_device_name(0)`, `list(torch.cuda.get_device_capability(0))`, and `torch.version.cuda` against the hardware identity at lines 198-200, before scoring. A different visible CUDA device could therefore produce observations labeled with the donor hardware identity. Add the same live hardware gate before the first H016 model call. This is a missing enforcement check; this review has not observed a live hardware mismatch.

## Core findings

- The fixed full48 input hash and exact source/burden grid select all 16 b0 originals, with no unknown-slot filtering. Both scope compilers, candidate order, event order, two response directions, 32 unique queries, and 64 candidate paths are retained.
- All planned tokenizer tries register before the first score call. Global context-node collisions and partial donor trees are rejected; qualification rejects any cached-tree import it cannot implement without a forward. The current supported path admits disjoint new contexts only.
- Each successful query requires exactly one forward for every canonical context, complete original finite FP16 full-vocabulary rows, exact NPZ members, context and raw-byte digests, and same-kernel replay. Actual calls and input tokens must equal the frozen plan.
- Four readouts are derived from complete name-plus-EOS raw joint log probabilities. Candidate-pool mass is not substituted for those scores. The row normalizer and joint assignment kernel retain their declared conditional meanings and explicit tie/unknown behavior.
- Observation and preparation inventories, root and child seals, input/protocol identities, installed project identities, and the qualified donor snapshot are guarded. The initial observation inventory now explicitly permits freeze.json to be absent before its creation.
- Evaluation first validates and replays every query, reconstructs every score, writes all four arms and the CPU freeze, and checks that freeze before reading either original reference seal or reference payload. Fixed-arm comparisons retain all denominators and do not choose a winning method.

## Validation boundary

This was a fresh static source review. No numerical oracle or test was executed by this reviewer. No tokenizer, model, CUDA call, real inference input, reference seal, design payload, construction truth, or gold data was opened. The parent separately reported an immutable-wheel 70-check pass with 129 members (126 Python modules and three resources); this reviewer did not reproduce that run. No core package file was modified. This report is the only review artifact written.

## Exact reviewed bytes

The fingerprint below is SHA256 over the listed lines, in listed order, UTF-8, with two spaces between digest and path, LF line endings, and one final LF. It identifies this review scope rather than an unbuilt future package or unreviewed runner revision.

Reviewed-source fingerprint: e5331f75304cb289c715082161120edbbf9b53f94f2e28000c189fda929e0b77

```text
899f77b7fa13b7e42616bb365f43d9230017725bee0c012a2ca5225b85acea22  experiments/paired_scope_qualification.py
ee2fa28c3eb780d7763da23c6cf0ffa8273ba7b37962906dc79cfa67df67a29d  experiments/paired_scope_evaluation.py
beb619f48c6735adbb617db5a96e4f35ca14062a6c886ba43f0bc33acf8b4939  experiments/research/paired_scope.py
c981eb91687f44dc433ad2f818141e1fd836c0cd70afbbcd24d1e3d2b05a5668  experiments/run.py
faaa31b95c17dc3e0e8ea278b113cf4b1e05b51fd5af119e9316c44f20a74ba3  tests/test_paired_scope_pipeline.py
bdc07641cd97865553a063024fb4ffde79a9ef226cf75c20855e4346cae30824  tests/test_paired_scope.py
f2dcd6aad9cb3ce9122d3db6c2f0ceb4a2ab993ac3f357eec749585facab5b85  experiments/prefix_completion.py
8b07e3da8476883dc062e49fd2df87485076976792f9443ba42f6bef76fd213b  experiments/research/completion_trie.py
6567479da83c0c6f1453387431f8f99effbea375d57641402fd5f31a8b851f65  experiments/answer_completion.py
68955f6d11a83f7196e375eee5d0b6b89dab2d5028f7d1db933a5f4136bde95c  experiments/semantic_qualification.py
196e7d1f5c8e218b6b6217a793df6eef7512029fc887f9edef83d1f648470515  experiments/provenance.py
d7bb93724bfd1c6fb5d2bbe327c792fa487e2f90977ce6e36ab0dfe2290154f5  outputs/h016-paired-scope-20261103/run_once.py
fd241fa9e9d9a9f3d36e01f041e0dd3ee9c8ccc0d051d8692dca7732e6c4a14e  outputs/h015-prefix-trie-20261103/run_engineering_probe.py
```

# H012 outer inventory: independent static review

Verdict: PASS for the artifact-only contract. This review opened producer code and synthetic tests only; it did not run H010/H012 data, models, tokenizers, NER, QA, predictions, or evaluation references.

Reviewed byte identities:

- experiments/natural_masked_inventory.py: d325d8ba894a7d7f846159a9889dedecb57ec3b55f09e0c8ce8d2c3b625498dd
- tests/test_natural_masked_inventory.py: 6a117856c8ed7940683acc4262be2b4d2f658f77ad7c49a3030df9942eb287fd
- DESIGN.md: 0a0a4005c502010eb5aae97df79fcb9242db8f0af541491e01d1e6c5f03bae7f

The reader admits all four named H010 payload paths, bounded sizes, seal schema, exact seal keys and aliases before hashing payloads. Adjacent H010 engineering resources are explicitly outside the admitted namespace. Its initial pilot seal bytes and all four hashes remain guarded through final output sealing. Inputs, external protocol, producer/helper code, original ordered eighteen cases, all 719 source and 136 response occurrences, exact Unicode slices, frozen frontend identity and evidence hash links are checked.

The output plan, protocol, producer identity and their freeze are written and guarded before calling the deterministic compiler. Source pools retain every source occurrence provenance. Each response occurrence is checked against its original owner, order, identity, label and span. The artifact admission rejects executable QA fields, assessed semantic states, and any finite or zero risk. Every unproposed region remains unknown and unscored; no candidate-pool membership becomes a truth decision. Source/frontend asset loading is absent.

Retained output members, readback equality and original hashes remain checked through final seal creation. Late input, H010 seal, protocol, code, output-member, result or final-seal changes prevent completion; failure during the completion/seal gate removes both success markers. Existing output directories are never overwritten. The eighteen synthetic tests cover these boundaries, including model-before-freeze prevention, missing/escaped/oversized members, altered population/offsets, late mutation and false QA/zero-risk claims. Their already reported pass is author evidence; this review did not rerun them.

This establishes inventory and information-flow compliance only. It does not establish natural answerability, source-pool domain completeness, entity/coreference correctness, factuality performance, or a remedy for the H011 direct-cloze failure. External NER supervision remains disclosed.

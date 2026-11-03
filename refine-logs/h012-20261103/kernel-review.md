# H012 pure kernel independent static review

2026-11-03. PASS, no material blocker found. Reviewed experiments/research/natural_masked_questions.py SHA256 21fc3098a1cb63c0268f2dab3fe93f9c300ad921dbc110cfd350decc01f532eb. The author reports 21 small synthetic tests passed; this reviewer did not rerun tests, NLP, tokenizers, QA or models, and did not inspect real H010 values, evaluation labels or predictions for this review.

All source candidate pools are built before response target values are inspected. Response native labels may request empty typed pools; exact candidate members/order still depend only on source entities. Exact raw strings are sorted without normalization or alias merging; every source occurrence provenance is retained. Empty/single/multiple counts do not become factuality or confidence judgments.

Every existing response entity index yields a retained proposal and original-case link. Full target-literal occurrences are enumerated including overlaps, same-label NER masks are added, same-coordinate provenance is combined with target priority, and other overlapping masks/invalid slices/bounds/reserved markers become explicit unknown. Mask construction preserves unmasked original characters. The integrity check binds every original target occurrence and the rendered artifact to its span map. Fixed placeholder substring coincidences are separately recorded; they are not mistaken for value-dependent copies from R.

Original values and source-pool membership remain readout-only. Membership is computed after all pools and proposals and cannot change them. Cloze artifacts contain only masked_response/target_placeholder/native_label, no question/system/messages/prompt. Anchor/domain completeness/alias/arithmetic/assertion/factuality/answerability remain unknown_not_assessed, qa_admitted=False, risk=None. No original occurrence is treated as semantically supported, and no unproposed token receives zero risk.

The module is generic over small populations. Its outer production caller must independently bind H010 seal and identities and require all 18 originals/136 response occurrences/719 source occurrences; malformed/missing inventories cannot be made complete by a pure helper. The fields remain artifact validation, not semantic coverage or a qualified natural QA interface. H011's direct-cloze failure is retained explicitly.

Only the new kernel/tests were authored for this task; no old observation helper, state, runtime or model asset was changed by this review.

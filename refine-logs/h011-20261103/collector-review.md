# H011 collector and pure compiler independent static review

Reviewed 2026-11-03, without running models, tokenizers, old/new tests, or opening actual observation scores/construction truth. The reviewer read the pure masked-query compiler, adapter, common collector diff and CLI diff. The compiler author separately reports 19 new synthetic tests passed; root owns collector tests and installed-release checks.

**PASS for the code contracts inspected, conditional on pinning the final literal question template before observation.** No unresolved numerical/information-flow/seal blocker was found. The prior NATURAL_CONTRACT_NEXT proposal quotes root's concept sentence; the implemented QUESTION_TEMPLATE has slightly different articles/punctuation and uses “denotes” plus an explicit masked-statement suffix. This is a pre-observation wording choice, not a tested variant. The observation protocol/spec must identify the actual constant bytes as final; neither wording may later be called a rescued or selected template. This discrepancy was sent to root before freezing.

Reviewed SHA256:

- experiments/research/masked_reconstruction.py: c78f49ceeff182f6f0432a4e67c13b7ad99cf07bf65f04927a1407a42457cf60
- experiments/masked_reconstruction_qualification.py: 9d6475d6f9546d2fbe3f2eb37d52597452fe50dc824541f5ed3d0b29fe226ef1
- experiments/reconstruction_qualification.py: b3b3e68cce0956c7dca6e85ded7ad7a71560b4529724adafdf660420b012cd7a
- experiments/run.py: abe9c7381def669b82f06aa4c94f9d665a1da64a3a03982f00779a70c4b82923

The reviewed fixed template is the single QUESTION_TEMPLATE in the compiler:

    Using only the source, give the full person name for [TARGET] in the masked statement. [OTHER] denotes a different source person.
    Masked statement:
    {masked_response}

## Information and scientific contract

The compiler first parses S with the unchanged strict two-person/two-event source grammar. Both candidates and their ordering derive only from S. It aligns the original two response slots only to erase and later map their exact original values; relation bindings from source parsing are not exported as preferred answers. Backend messages contain the fixed evidence-only system instruction, source S, and the compiled masked statement/question. Original target values, candidate indices, offsets, and mask maps stay in originals/readout metadata, outside those messages.

Every literal occurrence of the target person and the other source person is enumerated across the complete original R. The compiler rejects placeholder collisions, overlapping literals, missing values, residual original values, same-person responses, unsupported suffixes, and unknown source/response grammar. Non-person bytes—including event conditions, punctuation, whitespace, and 0/4/12 fixed filler suffix—are carried through by exact slicing. The second person is explicitly constrained to be different: this is admitted toy bijection knowledge, not a general natural-language assumption or a recovered edit-cause label.

The pure compiler preserves malformed originals with two explicit unknown occurrences and emits no partial request set for them. The controlled adapter subsequently requires the entire admitted cohort, refusing qualification if an original is unknown; it does not quietly drop it. This differs from a production natural detector's unknown-score policy and is correctly limited to a complete engineering qualification.

For the fixed complete cohort, the adapter enforces 48 originals, eight sources, 48 unique masked queries and 96 unique completions. Every original retains two exact slots and each unique query is referenced exactly twice, representing the two original response directions. Thus 96 original slot-query instances × two source candidates = 192 logical completions map to 96 unique executions. Response-direction collapse is required, not interpreted as independent evidence twice.

The question now comes from a value-erased R with retained event/suffix conditions; this is a different observable from H008 source-defined Q, H007 A/B/C judgment, and P021 original-R continuation. It is a conditional known-person-role bridge. It cannot establish natural NER/alias/candidate completeness, uniquely identify the originally edited token, or uniquely attribute previous verifier failure to label bias.

## Freeze, runtime observations and evidence integrity

The adapter supplies fixed protocol/prepare/plan/completion profiles to common _collect. All source/input/model/protocol identity and every complete-name-plus-EOS tokenizer metadata item for all 96 executions are checked and written before the first score_completion call. Per-question prefixes agree, one global EOS is fixed, complete candidate+EOS targets preserve their first-token causal shift, and untruncated limits are enforced.

The common collector still passes only copied messages and a source candidate to the backend. Message mutation, metadata/logit/logsumexp/logp/joint changes, full-vocabulary shapes, resource admission and allocator peaks are checked by the unchanged numerical validators. No A/B/C head semantics, fitted length/temperature normalization, or target-label selection is introduced.

_inventory now receives expected_requests from the strict profile (default 32). It admits exact root/child trees and every alias before hashes. Every sealed child retains request/raw/response/seal hashes, disk values are compared with validated memory, and all completed child snapshots persist across subsequent calls and finalization. The final root seal is now itself hash-bound before post-seal rechecks. Raw failures remain retained, existing output directories reject implicit retries, and no complete sealed root may survive a final guarded failure.

## Shared-refactor compatibility

H008 qualify now delegates to _collect with its original fixed profile. Its default request count remains 32; _completion derives the same enforced 48/8/16/32/96 coverage from prepared data rather than hardcoding those numbers. The default protocol, numerical _metadata/_response/reduction and outputs are otherwise unchanged. The added root-seal-hash check strengthens integrity without changing scores. The CLI's old branch retains the same admitted backend, while the new branch selects only the new fixed adapter.

This is source-level/numerical compatibility, not a claim that helper SHA bytes are unchanged. Archived H008/H009 observers/readers remain immutable; their old archived helpers must be read with their existing compatible installed targets or an explicitly reviewed compatible reader. A new checkout helper hash is not permission to rerun old model observations.

No current code was edited by this reviewer. This review is a static gate, not proof of semantic success; root's fake complete-population/manifest/mutation tests, final template identity, and installed checks remain separate admission evidence.

# H015 prefix backend: independent static review

Decision: **PASS for engineering freeze of the reviewed source bytes**. No remaining material blocker was found in this backend's stated contract. This is a static source review, not GPU qualification, launch approval, factual evaluation, or proof of exact replay by the language model.

## Reviewed identity and scope

| File | SHA256 | Lines |
| --- | --- | ---: |
| `experiments/prefix_completion.py` | `f2dcd6aad9cb3ce9122d3db6c2f0ceb4a2ab993ac3f357eec749585facab5b85` | 228 |
| `tests/test_prefix_completion.py` | `0f553fb6189408dd603df181f47efa884cb0e7a00796e4ba3b816a16adbd075d` | 224 |

Supporting pure/runtime source inspected: `experiments/research/completion_trie.py` (`8b07e3da8476883dc062e49fd2df87485076976792f9443ba42f6bef76fd213b`), `experiments/answer_completion.py` (`6567479da83c0c6f1453387431f8f99effbea375d57641402fd5f31a8b851f65`), and the `HFGenerator` runtime methods in `experiments/semantic_qualification.py` (`68955f6d11a83f7196e375eee5d0b6b89dab2d5028f7d1db933a5f4136bde95c`).

The reviewer read source and synthetic tests only. No tests, model imports, CUDA calls, installations, actual H013 requests/raw arrays, target labels, or model assets were executed/opened by the reviewer. Only this review document was written.

## Contracts checked

- Before `_ensure_model`, scoring rebuilds the entire canonical trie and requires its exact identity to match a prior tokenizer-only registration on this backend. Original caller messages and candidate order, copied completion metadata, tokenizer framing/object identity, and frozen model identity are rechecked. Candidate+EOS metadata retains complete literal suffixes, causal alignment, and EOS; token aliases and incomplete or altered trees are rejected.
- Initial asset verification hashes the complete nine-file manifest and checks its exact file population. Subsequent guards recheck population, regular paths, size/mtime stamps, and non-weight hashes; weight shards use the recorded stamps after the initial full hash. The declared positive vocabulary is checked against every context token and child token. All nodes are checked for the fixed 32-node, 4096-token, and 160-MiB temporary Python-vector admission bounds before weights are loaded. No truncation path is present.
- The backend inherits the existing admitted local-only Qwen2.5-3B loader: pinned runtime, NF4 with double quantization, FP16 compute, SDPA, CUDA device 0, evaluation mode, at least 2 GiB available RAM and 3 GiB free CUDA memory before model load, and a 3-GiB PyTorch allocator cap. This allocator cap is not a system-wide GPU reservation.
- Every unique context in one admitted trie has exactly one attempted prefix-only forward. Its input is the chat prefix plus only the candidate tokens preceding the current edge; future target tokens are absent. The input is `long[1,L]`, the mask is all ones, `logits_to_keep=1`, and `use_cache=False`.
- Successful captures require finite `float16[1,1,V]` logits. A native FP16 NumPy row `[V]` is copied, marked read-only, and retained with exact byte SHA256, dtype, shape, context identity, admission metadata, timing, and CUDA peaks. The pure CPU scorer receives lists made directly from those retained rows; its result does not return those Python vectors, and the temporary list is cleared in `finally`. The numerical kernel retains full-vocabulary normalization and strict probability gates without clamping or epsilon admission.
- Backend model-call totals remain distinct from the nested pure kernel's zero calls. Attempted calls and input tokens are counted before each invocation; successful-return forwards are counted separately. Generated output tokens remain zero. Failures retain completed rows, a bounded partial failing-node record, costs, stage/reason, and `automatic_retry=False`; oversized invalid outputs are not copied without bound. The method contains no retry loop.

## Finding closed before this PASS

The initial revision swallowed CUDA timing/peak measurement exceptions and could still return `scored` without required resource fields. The reviewed revision now rejects such a node at `resource_measurement`, retains its captured row/costs, and stops without retry. Successful measurements must be integer peaks satisfying `0 <= allocated <= reserved <= 3 GiB`. An earlier forward/capture exception still propagates with its original stage/reason while measurement failure is attached as secondary evidence.

The same revision preserves unsupported NumPy dtypes such as BF16 as a bounded read-only `uint8` bit view for failure forensics, while retaining the strict FP16 success gate. It does not cast those observations into FP16 or score them.

The author reported **23 CPU fake tests passed in 6.05 seconds**, including measurement-exception, allocator-overlimit, and BF16 failure-retention cases. Those test changes were inspected; the reviewer did not rerun them or verify a real GPU outcome. The retained-row recomputation test checks agreement with the same pure kernel, not an independent implementation of the numerical formulas.

## Boundary

This review covers one registered trie call and this backend API. It does not audit a future collector's cross-request deduplication, archive seals, wheel contents, or resource liveness at launch. `scored` establishes the observer-defined shared-context probability calculation; the kernel explicitly leaves exact language-model replay and factual correctness unestablished. No natural-dataset effectiveness or detector improvement follows from this engineering review.

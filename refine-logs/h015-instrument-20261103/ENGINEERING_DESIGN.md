# H015 bounded synthetic prefix-tree engineering qualification

This is a new observation-contract qualification, not a detector or QA experiment. It does not reuse H013 messages/candidates, read its raw records or references, repair its failure, or compare rankings. Existing model assets and the pinned Qwen2.5-3B NF4/FP16/SDPA runtime are reused without installation or download.

## Fixed inputs and budget

Exactly two synthetic tries:

- short_shared_branch: system = “This is a synthetic string-completion engineering probe.”; user = “Return a synthetic name.”
- bounded_512_shared_branch: the same system; user = the short user text plus “ Padding:” and n repetitions of “ neutral”. Before any model load, use the pinned offline fast tokenizer to find the smallest n in 0..512 for which the longest node context is exactly 512 tokens. No logits or semantic outcomes enter this construction. If no such n exists, stop; do not change strings.

Both use exactly these three complete candidate literals, without stripping: Rina Vale; Rina Vale Hart; Milo Reed. The first must be a strict token prefix of the second before terminal EOS, and the root must have at least two different child tokens. The shared node after Rina Vale must expose both EOS and continuation from the very same row. All candidate tokens and the fixed tokenizer EOS are retained; prefix-internal chat EOS is legal.

Each trie must have at most 16 unique context nodes; the whole job has at most 32. The short tree's maximum context must be at most 128 tokens; the long tree's maximum is exactly 512. The manifest records actual node/edge/target counts and summed input tokens. There are no extra LM warmups, retries, repeated-prefix “agreement” runs, generation or natural QA. One unique context means one forward. The 4096 backend bound is retained as a rejection bound, not claimed as tested context capacity. This engineering receipt will certify only its actual contexts up to 512.

With full model vocabulary V, exact raw FP16 storage is 2*V*N bytes; the Python-vector preflight bound is 32*V*max(N_per_trie) bytes. Require each to fit the existing 160 MiB temporary-vector policy. For the existing config, the tokenizer-prepared manifest supplies concrete numbers. Process total input-token upper bound is 32*512=16384, with actual sum frozen before loading weights.

## Freeze before observation

The preparation step is offline tokenizer-only. It binds this design, its preparation script, the unchanged completion framing, trie kernel, prefix backend and resource parent bytes, all nine model-file SHA256 values, config vocabulary, seven nonweight model-file checks and exact runtime pins. It writes a preparation freeze before decoding the synthetic strings, then retains both complete messages, all eighteen-field metadata, canonical tries, every context ID and edge, exact padding count, and complete call/storage budget in engineering-manifest.json. No CUDA admission or model construction occurs in preparation.

The actual runner additionally binds the new immutable installed wheel/target, runtime/lock identity, engineering manifest and producer script before any model load/forward. Source-file count is measured from the snapshot, never assumed to be 127. All nine assets are verified by the inherited offline identity gate before model load. Both tries must pass full metadata/limit checks before the first forward; comparison against the frozen tokenizer-only manifest must be exact.

Existing CPU/GPU environments, old wheels/targets and all prior observation files remain unchanged. Fresh wheel installation is only into a new --target directory, without dependencies. Outside-checkout tests use the new target, offline execution and disabled CUDA. The existing two new pure/backend suites are package checks, not new efficacy observations.

## Runtime/resource admission and retained arrays

Use the existing CUDA environment and inherited policy: explicit admission; current pinned torch/transformers/accelerate/bitsandbytes; available RAM at least 2 GiB at the core gate; free GPU at least 3 GiB before loading; CUDA device 0; 3 GiB PyTorch allocator cap; NF4 double quantization with FP16 compute; SDPA; eval; local_files_only; trust_remote_code=False. Root may apply its stricter established wrapper RAM/GPU startup gate. No threshold is reduced if resources are unavailable.

Each forward contains only its exact context, attention mask all ones, logits_to_keep=1 and use_cache=False. It must return shape [1,1,V], native FP16 and a fully finite vector. No dtype cast, candidate masking or vocabulary slicing is allowed. Both current/peak allocated and reserved CUDA bytes, forward/capture duration, wall duration, input tokens and attempted/completed calls must be measured; missing or invalid resource telemetry blocks success.

Retain every original vector in per-probe NPZ, one numeric nonobject member per node ID. Preserve its one-dimensional native FP16 bytes, shape, dtype, vector SHA256, context ID/SHA and original raw node receipt. Do not keep just selected logits or logZ. CPU readback uses allow_pickle=False and exact NPZ member inventory; it checks all full-vector bytes, finiteness, shape and vocab. Rebuild node observations from those vectors and call the unchanged pure kernel. Its output must exactly match the captured tree readout, including all complete name+EOS paths, selected edges, shifted/full normalizers, raw sums, absolute pool mass, conditional mass and underflow flags.

No test requires a candidate to be likely or correct. Qualification requires structure, finite rows, one row per context, byte-exact persistence/replay, strict node/pool probability gates and complete telemetry. Conditional candidate mass is not truth probability. Pure kernel model_calls=0 must not overwrite the runner's actual language-forward count.

## Failure and completion

Write the full planned node population and initial identities before observation. Persist raw arrays, costs and partial-failure receipts returned by the backend even when a gate fails. A hard process termination leaves an incomplete attempt; do not invent exact completed-call counts that were not captured. No automatic retry, clipping, epsilon change, alternate candidate, prompt or weight is allowed.

Seal successful output only after all two probes and the full CPU vector replay pass and all initial source/manifest/model/target/input/NPZ identities remain unchanged. Put complete/seal writing inside a revocation guard: a final integrity failure removes success markers while retaining evidence. A failed run remains failed.

Engineering success would establish the implemented shared-prefix probability tree for these synthetic contexts under this instrument, not exact ideal-real-arithmetic LM replay, speedup, context4096 capacity, H013 repair, natural factuality or promotion. Any later paired compound/atomic experiment requires its own frozen design and references release.

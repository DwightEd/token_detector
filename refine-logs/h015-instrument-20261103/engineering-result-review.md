# H015 actual engineering result: independent CPU review

**PASS** for the exact sealed synthetic engineering observation. All 17 payload members and the root seal were verified, with exact namespace and regular unredirected paths before payload hashing. The initial seal and every payload hash remained unchanged through review. No model was called.

## Frozen identity

- Initial/final seal SHA256: 471e953407fcb2fda086d449f88d25b1d086533237f9b4dd9eb868f6fe0724b9.
- Summary SHA256: 3ae0b0eb786c8c7c9646a94e22c1b88ea627abf54d57fca9dba8ee55ed3cc047.
- Pure kernel SHA256: 8b07e3da8476883dc062e49fd2df87485076976792f9443ba42f6bef76fd213b; matched the sealed source/installed package maps before import.
- Metadata freeze bindings matched. The observed model identity is the manifest identity plus the runner-declared execution provenance; all nine asset identities are retained.

## Checks and result

Each NPZ had exactly eight expected bounded node-array members, loaded with allow_pickle=False. All 16 arrays were finite native FP16 of shape [151936], with exact byte SHA256, dtype/shape metadata, context-ID identity, and canonical trie mapping. Total uncompressed vector bytes: 4,861,952.

Both tries rebuilt exactly from frozen completion metadata. All eight nodes per probe had exactly one successful forward and the planned context IDs; all 16 contexts were distinct. Each output shape was [1,1,151936]. Generated output tokens and nested pure-kernel model calls were zero.

| Probe | Nodes/forwards | Input tokens | Maximum context | Forward plus capture seconds |
| --- | ---: | ---: | ---: | ---: |
| short_shared_branch | 8 | 240 | 32 | 3.344960200 |
| bounded_512_shared_branch | 8 | 4080 | 512 | 1.584497700 |

Total: 16 attempted calls, 16 completed forwards, 4,320 input tokens, 0 generated tokens; 4.929457900 seconds forward plus capture. Recorded runner wall time: 110.845982100 seconds. Wall time is a sealed measurement, not independently timed by this CPU review.

Current allocator snapshots matched the separately sealed before files. Every before/after current and node peak counter was a nonnegative integer byte count with allocated <= reserved <= 3 GiB; final current counters did not exceed captured node peaks. Aggregate peaks reconstructed exactly:
- Allocated: 2,141,098,496 bytes.
- Reserved: 3,068,133,376 bytes.

- short_shared_branch current before/after: {"after": {"allocated_bytes": 2064498176, "reserved_bytes": 3068133376}, "before": {"allocated_bytes": 0, "reserved_bytes": 0}}.
- bounded_512_shared_branch current before/after: {"after": {"allocated_bytes": 2064498176, "reserved_bytes": 3068133376}, "before": {"allocated_bytes": 2064498176, "reserved_bytes": 3068133376}}.

Both probability trees were recomputed from the same persisted full FP16 rows with the byte-matched pure kernel. Results exactly matched both stored tree_result and sealed replay artifacts. Strict gates passed without clamping or epsilon admission. This is **same-kernel persistence replay**, not an independent mathematical implementation or oracle.

## Boundary

The captured scope is eight short contexts (maximum 32 tokens) and eight bounded contexts (maximum 512 tokens). It does not qualify arbitrary 4096-token use, establish exact language-model replay or factual correctness, repair H013, demonstrate natural-task effectiveness, or promote a method. No refs, target labels, H013 raw files, model weights, CUDA API, or inference were accessed by this reviewer. Only JSON/NPZ reads, hashes, NumPy checks, and the standard-library-only pure kernel were used.

Root reports two earlier resource-wait controllers ended with zero calls and were retained; this review does not independently audit those controller histories. It validates the first actual probe archive. No old output was changed; only this report was written.

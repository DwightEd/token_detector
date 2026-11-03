# H016 fresh static review: runner and launch delta

Decision: the prior missing live-hardware gate is fixed. No newly verified important blocker in the reviewed runner, bounded launcher, or tokenizer preparation script for the planned ordinary Python invocation. The previously reviewed core package and tests have unchanged source hashes. No model launch or result is asserted by this review.

The active scope remains the previously exposed 16-b0-case/eight-source toy, with all four fixed compound/atomic by independent/joint arms. No natural-task performance claim follows.

## Reviewed changes

- `run_once.py` now compares the live CUDA device name, capability, and torch CUDA runtime to the frozen donor hardware identity at lines 46-50, before constructing the backend or calling qualification. It also checks the offline-HF settings and admits only default or device-zero CUDA visibility. The pinned H015 package snapshot and Git byte-binding helper remains bound to SHA256 fd241fa9e9d9a9f3d36e01f041e0dd3ee9c8ccc0d051d8692dca7732e6c4a14e.
- `run_once.py` no longer eagerly imports the CPU evaluator into the isolated GPU environment. The evaluation protocol file is bound by its digest in the run spec. The CPU preparation subprocess produces that file from the installed evaluator contract before the tokenizer-only stage.
- `launch_once.py` checks the claim's launcher hash and run-spec digest, uses an exclusive persistent lock, and requires fresh state/output. It measures physical RAM and GPU free memory twice and requires at least 3 GiB RAM plus 3584 MiB GPU memory before starting one subprocess. It monitors STOP and an 1800-second child observation deadline. Its failure cleanup targets the PID tree created by that Popen call, uses bounded cleanup commands, retains failure state, and contains no model retry or relaunch loop.
- `prepare_actual.py` checks the admitted installed-file hashes, copies the existing engineering model identity while updating execution provenance, and invokes the integrated prepare-paired-scope tokenizer-only stage from the immutable installed target. The previous sklearn import issue is avoided by importing the evaluator contract only with the CPU interpreter. Existing model identity bytes are compared rather than overwritten.

## Runtime and review boundaries

These wrapper scripts use Python assert for admission checks. Their planned argv contains no optimization flag; PYTHONOPTIMIZE must remain unset or zero so those gates stay active. This review did not inspect a completed future run spec or claim because the parent stated they had not yet been written.

This was a static delta review. No reviewed script, numerical oracle, test, tokenizer, model, or CUDA call was executed by this reviewer. No real inference input, reference seal, design payload, construction truth, or gold data was opened. No core package file was modified. The parent separately owns preparation execution, installed-wheel tests, run registration, resource checks, and any model attempt.

## Exact reviewed bytes

The fingerprint below is SHA256 over the listed lines, in listed order, UTF-8, with two spaces between digest and path, LF line endings, and one final LF. The core hashes match the first review. A later wrapper edit requires a delta check against this byte inventory.

Reviewed-source fingerprint: e2fb45c9392ade65b8312ddf4bc42ac69d21ddca4a16b6046a3dfbacca51a8d4

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
ba0408cfd433454d721bff2097e2888f8fd20c5acd467ac09300b802ad02667f  outputs/h016-paired-scope-20261103/launch_once.py
ef7f549e20c35aa5562d47cd695becd03419ad8a842f6bd7f2dfc659a581bc77  outputs/h016-paired-scope-20261103/prepare_actual.py
fd241fa9e9d9a9f3d36e01f041e0dd3ee9c8ccc0d051d8692dca7732e6c4a14e  outputs/h015-prefix-trie-20261103/run_engineering_probe.py
```

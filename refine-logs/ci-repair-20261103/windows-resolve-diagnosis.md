# Windows concurrent path-resolution diagnosis and minimal repair

The failure was reproduced and explained: concurrent directory creation caused an extended-DOS path spelling mismatch, rather than an observed escape from the temporary project.

## Captured evidence

The diagnostic compiled a temporary AST-instrumented copy of tools/research_memory.py, original SHA256 f363a189a261a05fe383cc243f5b785af2d02797126147702f0ff8e0f9687629. It used eight threads and fresh temporary projects only. The production tool and tests were unchanged during diagnosis; no real research-state records, models, or labels were opened.

- windows-resolve-diagnostic.json: 3 rounds, 24 synthetic claim attempts, 5 containment exceptions.
- windows-resolve-trace-diagnostic.json: a separate 3-round/24-attempt reproduction, 5 exceptions, with thread-local Win32 call traces.
- diagnose_windows_resolve.py: isolated reproduction; --trace adds process-local tracing of ntpath._getfinalpathname. Temporary projects were explicitly retained.
- Runtime: Anaconda Python 3.13.9 on Windows. This does not assert the same platform behavior on Linux/Python 3.11 CI.

In a failing invocation, project.parts[0] was C:\, while path.resolve().parts[0] was \\?\C:\. All later project components were identical, followed by the expected research-state/runs/64-hex-id suffix. Lexical is_relative_to compared different anchors and rejected the path.

The trace and the installed runtime's ntpath.realpath implementation establish this sequence:

1. The first _getfinalpathname(full_child) raised WinError 3, directory not found.
2. The non-strict fallback walked upward and obtained an existing ancestor with an extended \\?\C:\ prefix.
3. Another claim thread created intermediate directories.
4. The final probe of that same full child raised WinError 2, file not found.
5. On an exception, ntpath.realpath removes its added extended prefix only when the final winerror equals the initial one. Here 2 differed from 3, so the prefix remained.

The first captured trace contains exactly that initial-3/final-2 sequence. Changing directory existence during the call changed the returned spelling.

## Implemented repair

After the root agent confirmed that the earlier full-suite process had terminated, the author changed only tools/research_memory.py and tests/test_research_memory.py. Both files were normalized to LF before testing.

The new _path_key helper is comparison-only. For PureWindowsPath instances, including Windows Path instances, it folds absolute extended drive aliases and valid extended UNC aliases into ordinary PureWindowsPath keys. It preserves the rest of each component and leaves other device namespaces unchanged. On POSIX paths it returns the original path. PureWindowsPath makes the alias tests executable on Linux as well.

The helper does not perform another resolve, strip trailing dots/spaces, or change physical I/O paths. At all three existing boundaries:

- _record_path resolves the project and child, compares keys, and returns the resolved entity path.
- _artifact_path retains its resolved entity path, compares keys, and uses the same keys for the bookkeeping self-certification relative path.
- finish keeps the resolved artifact for existence and hash checks and uses keys consistently for containment and the project-relative artifact identity.

Resolution still precedes containment, so an outside symlink/junction target has different path components and is rejected. No directory is created before that check. The change preserves the existing containment boundary; it does not claim to eliminate malicious filesystem replacement races after a check.

## Bound validation

Final LF source hashes:

| File | SHA256 |
| --- | --- |
| tools/research_memory.py | e001eced69f33b365ca59f487e6c39195a5062bd6d6f786245118b9a901dc475 |
| tests/test_research_memory.py | 714b2a2cef79a29c53b635695e6253a77ddcc783adce132588b106256c73c9f3 |
| .github/workflows/tests.yml, unchanged | f715c6ce5633fdedc3fbc453de5ca3db3dffb78b87ffb02041630ebb95bd9e3f |

Only the target test_research_memory.py was run, with --import-mode=importlib from the previously admitted outside-checkout working directory and the same installed target. All 126 installed package members and the admitted wheel hash were checked before and after. The repository root was not added to PYTHONPATH; CUDA was hidden and offline flags remained enabled.

Result: **40 passed in 0.90 seconds, 0 skipped**. See memory-path-fix-check.log and memory-path-fix-check.json; run_memory_path_fix_check.py records the exact command, source bindings, target, working directory, and checks. Both repaired Python files also passed static compilation before the run.

The new cases cover:

- DOS and UNC aliases in both directions, including case variants and identical relative identity.
- Different drives, UNC shares/servers, sibling-prefix paths, and other device namespaces remaining outside.
- Preservation of trailing dot/space component spelling.
- All three public path boundaries under deterministic mixed alias results, with spies verifying that entity paths, not comparison keys, reach I/O/hash publication.
- Actual temporary directory-symlink escapes for claim, artifact admission, and finish, rejected before outside publication or hashing. All three physical cases executed on this Windows host; the test permits an explicit Windows privilege skip elsewhere, while Linux errors propagate instead of skipping.
- The original eight-thread atomic-claim test, without sleeps or retry-based repair.

No full suite, model run, or push was performed by this author for the repair. The earlier root full-suite timeout remains an incomplete result; this targeted pass is not a remote CI-green claim. Root independent source review and the full Linux CI outcome remain pending.

# CI repository-tool import fix: independent static review

Decision: **PASS for the minimal test-only import fix**. The reviewed change addresses the reported collection failure without changing the installed detector packages, frozen research code, or the CI isolation boundary. This is a static review; the full CPU suite and remote CI outcomes remain separate evidence.

## Bound source bytes

| File | SHA256 |
| --- | --- |
| `tests/test_research_memory.py` | `3a6a1a7fe125c99f9b09ccda6a1033171f953a19e3ac8e2b809105d4fc3085f7` |
| `.github/workflows/tests.yml` | `f715c6ce5633fdedc3fbc453de5ca3db3dffb78b87ffb02041630ebb95bd9e3f` |
| `tools/research_memory.py` | `f363a189a261a05fe383cc243f5b785af2d02797126147702f0ff8e0f9687629` |

The reported failing GitHub run was `41563596975`, job `125679677985`: installation and CLI smoke checks succeeded, then outside-checkout collection failed on `from tools.research_memory ...`. The root agent supplied this log finding; this reviewer did not independently fetch the remote log.

## Reasoning

`pyproject.toml` explicitly packages `myproject`, `experiments`, and `experiments.research`; `tools` is repository tooling. The test now locates exactly `Path(__file__).resolve().parents[1] / 'tools' / 'research_memory.py'` and loads that file with `importlib.util.spec_from_file_location`, `module_from_spec`, and `loader.exec_module`. Its private module name avoids dependence on importability of a top-level `tools` package. No checkout directory is added to `sys.path`.

The current tool uses only standard-library imports, has no relative imports or dataclass/module-registration dependency, and protects its CLI with `if __name__ == '__main__'`. Executing the module under the private name therefore defines the functions without running the CLI. Registration in `sys.modules` is not required by this current implementation.

Both monkeypatch targets now address the loaded module directly: `research_memory.json, 'dump'` and `research_memory, 'file_hash'`. The local function aliases retain that module's globals, so the existing assertions still exercise the patched dependencies. No test is skipped or weakened.

The workflow continues to install the package, change to `$RUNNER_TEMP`, run the installed CLI smoke checks, and invoke `python -m pytest --import-mode=importlib "$GITHUB_WORKSPACE/tests" -q`. The repair does not make detector imports resolve from the checkout. Packaging the repository tool or changing global import paths would be unnecessary for this failure.

## Existing completed import inventory

Before this report was requested, the reviewer performed a read-only AST scan of **61 Python files under `tests/`**, inspecting `Import` and absolute `ImportFrom` statements. Excluding the explicitly packaged roots and the test package itself, **zero imports from other checkout top-level directories or scripts remained** in the repaired tree. The accompanying text search found no remaining test imports of `tools`, `scripts`, or `outputs`, and no test-level global `sys.path` insertion. Existing `tests.*` references are internal synthetic helper reuse, not this unpackaged repository-tool import.

There was no earlier common script-loader abstraction in `tests/` to preserve. The exporter synthetic tests already use an analogous explicit file-loading pattern; introducing a new shared abstraction for this one repair is unnecessary.

This report records that completed scan; it did not trigger another scan. The reviewer did not run tests, install packages, execute the research-memory tool, modify core/state/workflow files, or access models or target labels. Only this report was written for this follow-up. The root agent's full outside-checkout CPU run was still in progress when this report was prepared; no local-suite or remote-green result is asserted here.

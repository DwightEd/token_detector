"""Evidence reuse and concurrency protect the research loop from duplicate runs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from threading import Event
import pytest

from tools.research_memory import claim, finish, fingerprint, inspect_run


def spec():
    return {'hypothesis_id': 'H-new-information', 'intervention': 'target_blind_probe',
            'stage': 'scoring',
            'method': {'entrypoint': 'probe.score', 'config': {'temperature': 0}},
            'code_sha256': {'relevant_kernel.py': 'a'*64},
            'data_sha256': {'original_inputs': 'b'*64},
            'observer': {'weights': 'c'*64, 'backend': 'matched_sdpa'},
            'population': {'manifest': 'd'*64},
            'evaluation': {'case_panel': 'e'*64, 'gold_usage': 'after_seal'}, 'seed': 17}


def test_equivalent_prose_and_output_do_not_launch_again(tmp_path):
    first = spec()
    created = claim(tmp_path, first)
    evidence = tmp_path/'sealed_result.json'
    evidence.write_text('{"pilot":"negative"}', encoding='utf-8')
    finish(tmp_path, created['run_id'], 'refuted', 'No new semantic information', [evidence])
    equivalent = {**first, 'question': 'Rephrased question', 'output': 'another-folder',
                  'hypothesis_id': 'renamed-hypothesis', 'intervention': 'renamed-intervention',
                  'method': {**first['method'], 'display_name': 'New marketing name'}}
    result = claim(tmp_path, equivalent)
    assert result['action'] == 'reuse' and result['outcome'] == 'refuted'
    assert result['run_id'] == created['run_id']


@pytest.mark.parametrize('key', ['method', 'code_sha256', 'data_sha256', 'observer',
                                 'population', 'evaluation', 'seed', 'stage'])
def test_real_dependency_changes_have_new_identity(key):
    before, after = spec(), spec()
    if key == 'method':
        after[key]['config']['temperature'] = 1
    elif isinstance(after[key], dict):
        after[key] = {**after[key], 'new_dependency': 'f'*64}
    elif key == 'seed':
        after[key] += 1
    else:
        after[key] += '-different'
    assert fingerprint(before) != fingerprint(after)


def test_concurrent_claims_reserve_once(tmp_path):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: claim(tmp_path, spec()), range(8)))
    assert sum(r['action'] == 'claimed' for r in results) == 1
    assert all(r['action'] in {'claimed', 'reserved', 'already_claimed'} for r in results)


@pytest.mark.parametrize('change', ['missing', 'changed'])
def test_missing_or_changed_evidence_requires_recovery_not_silent_rerun(tmp_path, change):
    run = claim(tmp_path, spec())
    evidence = tmp_path/'output.json'; evidence.write_text('original', encoding='utf-8')
    finish(tmp_path, run['run_id'], 'supported_scope', 'Only this controlled scope', [evidence])
    if change == 'missing':
        evidence.unlink()
    else:
        evidence.write_text('modified', encoding='utf-8')
    assert claim(tmp_path, spec())['action'] == 'artifact_recovery_required'


def test_incomplete_identity_and_unclaimed_completion_fail(tmp_path):
    partial = spec(); partial.pop('population')
    with pytest.raises(ValueError): fingerprint(partial)
    with pytest.raises(ValueError): finish(tmp_path, 'a'*64, 'refuted', 'No claim', [tmp_path/'none'])
    invalid = spec(); invalid['code_sha256'] = {'kernel': 'not-a-hash'}
    with pytest.raises(ValueError): fingerprint(invalid)


def test_no_overwrite_or_self_certification(tmp_path):
    run = claim(tmp_path, spec()); evidence = tmp_path/'result.json'
    evidence.write_text('{}', encoding='utf-8')
    finish(tmp_path, run['run_id'], 'inconclusive', 'Insufficient information', [evidence])
    with pytest.raises(FileExistsError):
        finish(tmp_path, run['run_id'], 'supported_scope', 'Rewrite result', [evidence])
    another = spec(); another['method']['config']['control'] = True
    reserved = claim(tmp_path, another)
    own = tmp_path/'research-state/runs'/reserved['run_id']/'claim.json'
    with pytest.raises(ValueError):
        finish(tmp_path, reserved['run_id'], 'supported_scope', 'Circular proof', [own])


def test_finish_publishes_only_complete_json(tmp_path, monkeypatch):
    run = claim(tmp_path, spec())
    evidence = tmp_path/'output.json'; evidence.write_text('{}', encoding='utf-8')
    entered, release = Event(), Event()
    original = json.dump

    def slow_dump(value, stream, **kwargs):
        if value.get('outcome'):
            entered.set()
            assert release.wait(5)
        return original(value, stream, **kwargs)

    monkeypatch.setattr('tools.research_memory.json.dump', slow_dump)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(finish, tmp_path, run['run_id'], 'refuted', 'Negative', [evidence])
        assert entered.wait(5)
        try:
            assert inspect_run(tmp_path, spec())['action'] == 'already_claimed'
        finally:
            release.set()
        future.result()
    assert inspect_run(tmp_path, spec())['action'] == 'reuse'


@pytest.mark.parametrize('target,field,value', [
    ('claim.json','run_id','f'*64), ('result.json','run_id','f'*64),
    ('result.json','outcome','invented'), ('result.json','schema',99),
    ('result.json','artifacts',[{'path':'../external.json','sha256':'a'*64}]),
    ('result.json','artifacts',[{'path':'/external.json','sha256':'a'*64}]),
    ('result.json','artifacts',[{'path':'research-state/runs/fake/result.json','sha256':'a'*64}]),
])
def test_tampered_records_never_reuse_or_hash_escaped_paths(tmp_path, monkeypatch, target, field, value):
    run = claim(tmp_path, spec())
    evidence = tmp_path/'output.json'; evidence.write_text('{}', encoding='utf-8')
    finish(tmp_path, run['run_id'], 'refuted', 'Negative', [evidence])
    path = tmp_path/'research-state/runs'/run['run_id']/target
    saved = json.loads(path.read_text(encoding='utf-8')); saved[field] = value
    path.write_text(json.dumps(saved), encoding='utf-8')

    def never_hash(_):
        raise AssertionError('Corrupted record reached artifact hashing')

    monkeypatch.setattr('tools.research_memory.file_hash', never_hash)
    with pytest.raises(ValueError): inspect_run(tmp_path, spec())

"""Cross-release donor admission uses synthetic observations and never reads truth."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import data, observation_reuse as reuse, verdict_qualification as observer


OLD_MODEL = {'model_id': 'synthetic', 'model_files_sha256': {'weights': 'a' * 64},
             'runtime': {'torch': 'frozen'},
             'execution': {'installed_observer_wheel_sha256': 'b' * 64,
                           'CPU_module_target': 'old-target', 'threads': {'OMP_NUM_THREADS': '2'}}}
NEW_MODEL = deepcopy(OLD_MODEL)
NEW_MODEL['execution'].update(installed_observer_wheel_sha256='c' * 64, CPU_module_target='new-target')


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def reseal(root):
    for parent in ('cases', 'requests'):
        for child in (root / parent).iterdir():
            write(child / 'seal.json', {p.name: data.file_hash(p) for p in child.iterdir()
                                        if p.name != 'seal.json'})
    write(root / 'seal.json', {str(p.relative_to(root)): data.file_hash(p)
                              for p in root.rglob('*') if p.is_file() and p != root / 'seal.json'})


class Scorer:
    model_identity = OLD_MODEL

    def count_input_tokens(self, messages):
        return 20

    def validate_labels(self, messages, labels):
        return {'input_tokens': 20, 'label_token_ids': dict(zip(labels, [10, 11, 12])),
                'label_decodes': dict(zip(labels, labels)), 'prefix_alignment_verified': True}

    def score_labels(self, messages, labels):
        assignment = json.loads(messages[0]['content'][messages[0]['content'].rfind('{'):])
        logits = {'supported': 2., 'contradicted': 1., 'unresolved': -1.}
        return {**self.validate_labels(messages, labels),
                'label_logits': {label: logits[assignment[label]] for label in labels},
                'full_vocab_logsumexp': 4., 'output_tokens': 0,
                'actual_forward_passes': 1, 'seconds': 0.125}


@pytest.fixture
def donor(tmp_path, monkeypatch):
    source = {'source_sha256': {
        'experiments/semantic_qualification.py': data.file_hash(Path(reuse.backend.__file__)),
        'experiments.research/semantic_verdict.py': data.file_hash(Path(reuse.verdict.__file__)),
        'experiments/verdict_qualification.py': 'd' * 64}}
    monkeypatch.setattr(observer, 'research_identity', lambda: deepcopy(source))
    monkeypatch.setattr(observer, 'completed_identity', lambda old: old)

    def build(count=18):
        panel = tmp_path / 'panel'
        panel.mkdir()
        inputs = panel / 'inputs.jsonl'
        rows = [{'id': str(i), 'source_id': 's' + str(i), 'task': 'QA', 'split': 'test',
                 'source': 'Evidence X.', 'response': 'A has X.',
                 'prompt': 'Read Evidence X. Answer fully.', 'generator': 'original'} for i in range(count)]
        inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        (panel / 'evaluation_only.jsonl').write_text('DO NOT OPEN', encoding='utf-8')
        protocol = tmp_path / 'protocol.json'
        write(protocol, dict(observer.DEFAULT_PROTOCOL, expected_cases=count, max_forward_passes=count * 6))
        root = tmp_path / 'outputs' / 'donor'
        observer.qualify(inputs, protocol, root, Scorer(), OLD_MODEL)
        return root, inputs
    return build


@pytest.mark.parametrize('count', [18, 16])
def test_admits_whole_cohort_preserves_original_response_and_provenance(donor, monkeypatch, count):
    root, inputs = donor(count)
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name not in {'evaluation_only.jsonl', 'annotations.jsonl', 'manifest.json'}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    admitted = reuse.load_donor(root, NEW_MODEL, data.file_hash(inputs))
    assert len(admitted) == 6
    for key, item in admitted.items():
        origin = item['provenance']
        path = root / 'requests' / origin['old_request_key']
        assert item['response'] == data.read_json(path / 'response.json')
        assert item['response']['actual_forward_passes'] == 1
        assert item['response']['seconds'] == 0.125
        assert key == reuse.equivalence_key(data.read_json(path / 'request.json'))
        assert origin['model_identity'] == OLD_MODEL
        assert origin['root_seal_sha256'] == data.file_hash(root / 'seal.json')
        assert origin['files_sha256']['backend_raw.json'] == data.file_hash(path / 'backend_raw.json')
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('failure', ['missing', 'tamper', 'unbound_extra', 'case_ownership', 'costs'])
def test_rejects_broken_or_inconsistent_archive(donor, failure):
    root, inputs = donor()
    response = next((root / 'requests').glob('*/response.json'))
    if failure == 'missing':
        response.unlink()
    elif failure == 'tamper':
        response.write_text('{}', encoding='utf-8')
    elif failure == 'unbound_extra':
        (root / 'unbound.json').write_text('{}', encoding='utf-8')
    else:
        path = root / ('plan.json' if failure == 'case_ownership' else 'complete.json')
        value = data.read_json(path)
        if failure == 'case_ownership':
            value['records'][0]['source_id'] = 'different'
        else:
            value['costs']['actual_forward_passes'] += 1
        write(path, value)
        reseal(root)
    with pytest.raises(ValueError):
        reuse.load_donor(root, NEW_MODEL, data.file_hash(inputs))


@pytest.mark.parametrize('field', ['runtime', 'threads', 'model_file', 'input_hash'])
def test_rejects_effective_instrument_or_original_input_changes(donor, field):
    root, inputs = donor()
    model, digest = deepcopy(NEW_MODEL), data.file_hash(inputs)
    if field == 'runtime':
        model['runtime']['torch'] = 'changed'
    elif field == 'threads':
        model['execution']['threads']['OMP_NUM_THREADS'] = '8'
    elif field == 'model_file':
        model['model_files_sha256']['weights'] = 'f' * 64
    else:
        digest = 'f' * 64
    with pytest.raises(ValueError):
        reuse.load_donor(root, model, digest)


@pytest.mark.parametrize('failure', ['metadata_alias', 'unsafe_root_seal', 'unsafe_child_seal'])
def test_rejects_aliases_and_seal_paths_before_hash(donor, monkeypatch, failure):
    root, inputs = donor()
    digest = data.file_hash(inputs)
    if failure == 'metadata_alias':
        path = root / 'plan.json'
        path.unlink()
        try:
            path.symlink_to(inputs.parent / 'evaluation_only.jsonl')
        except OSError:
            pytest.skip('Symbolic links unavailable')
    else:
        path = (root if failure == 'unsafe_root_seal' else next((root / 'requests').iterdir())) / 'seal.json'
        value = data.read_json(path)
        value['..\\evaluation_only.jsonl'] = 'f' * 64
        write(path, value)
    monkeypatch.setattr(data, 'file_hash', lambda _: pytest.fail('Unsafe inventory reached hashing'))
    with pytest.raises(ValueError):
        reuse.load_donor(root, NEW_MODEL, digest)


def test_equivalence_ignores_only_administrative_fields(donor):
    root, _ = donor()
    execution = data.read_json(next((root / 'requests').glob('*/request.json')))
    changed = deepcopy(execution)
    changed.update(protocol_sha256='new-protocol', source_identity_sha256='new-code', model_identity=NEW_MODEL)
    assert reuse.equivalence_key(changed) == reuse.equivalence_key(execution)
    semantic = {k: v for k, v in execution.items() if k not in
                {'protocol_sha256', 'source_identity_sha256'}}
    assert reuse.equivalence_key(semantic) == reuse.equivalence_key(execution)
    before = deepcopy(NEW_MODEL)
    reuse.effective_model_identity(NEW_MODEL)
    assert NEW_MODEL == before
    changed['scoring']['use_cache'] = True
    assert reuse.equivalence_key(changed) != reuse.equivalence_key(execution)
    changed['extra_inference_option'] = True
    with pytest.raises(ValueError, match='Unknown request'):
        reuse.equivalence_key(changed)


def test_conflicting_observations_cannot_select_a_collision_winner(donor, monkeypatch):
    root, inputs = donor()
    monkeypatch.setattr(reuse, 'equivalence_key', lambda _: 'same-equivalence')
    with pytest.raises(ValueError, match='Conflicting responses'):
        reuse.load_donor(root, NEW_MODEL, data.file_hash(inputs))

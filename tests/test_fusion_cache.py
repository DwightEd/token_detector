"""Independent native-population checks for label-free historical cache export."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import data
from experiments.fusion_cache import export
from experiments.rank_fusion import AXES, _load_population, score
from experiments.run import main


def write_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def seal(root, names):
    write_json(root / 'seal.json', {name: data.file_hash(root / name) for name in names})


def fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    native, scores = tmp_path / 'native', tmp_path / 'scores'
    native.mkdir(); scores.mkdir(); (native / 'observations').mkdir()
    groups, prepared, records, scored = [], [], [], []
    for group in range(3):
        root = tmp_path / ('prepared' + str(group)); root.mkdir(); prepared.append(root)
        sample = {'id': 'sample' + str(group), 'source_id': 'source' + str(group),
                  'split': 'train' if group == 0 else 'test', 'task': 'QA',
                  'source': 'evidence' + str(group), 'prompt': 'Read evidence' + str(group),
                  'response': 'A文B'}
        (root / 'inputs.jsonl').write_text(json.dumps(sample) + '\n', encoding='utf-8')
        (root / 'annotations.jsonl').write_text('MUST NOT OPEN', encoding='utf-8')
        manifest = {'inputs_sha256': data.file_hash(root / 'inputs.jsonl'), 'samples': 1,
                    'coverage': {'complete': 1}, 'selection': {'limit': None},
                    'raw_files': {str(tmp_path / 'absent-private-raw.json'): 'not-verified'}}
        write_json(root / 'manifest.json', manifest)
        groups.append({'path': '/original/' + root.name, 'role': 'fit' if group == 0 else 'test',
                       'input_sha256': manifest['inputs_sha256'],
                       'manifest_sha256': data.file_hash(root / 'manifest.json')})
        target = np.array([0, 2, 3])
        raw = {'full_token_id': np.array([10, 999, 11, 12]),
               'offsets': np.array([[0, 1], [1, 1], [1, 2], [2, 3]]),
               'special_ids': np.array([999]), 'target': target,
               'token_id': np.array([10, 11, 12]), 'answer_index': np.full(3, group),
               'labels': np.array([{'poison': True}], dtype=object)}
        member = 'observations/' + str(group) + '.npz'
        np.savez(native / member, **raw)
        record = {'id': sample['id'], 'group': group, 'file': member,
                  'sha256': data.file_hash(native / member),
                  'response_sha256': data.text_hash(sample['response'])}
        records.append(record)
        if group:
            member = str(group) + '.npz'
            np.savez(scores / member, target=target, token_id=raw['token_id'],
                     offsets=raw['offsets'], answer_index=raw['answer_index'],
                     frozen_score=np.array([9., 1., 5.]), witness_score=np.array([0., 7., 4.]),
                     labels=raw['labels'])
            scored.append({**record, 'file': member, 'sha256': data.file_hash(scores / member)})
    write_json(native / 'plan.json', {'groups': groups, 'model_files_sha256': {
        '/original/model/config.json': 'a' * 64, '/original/model/weights.safetensors': 'b' * 64}})
    write_json(native / 'collection.json', {'records': records})
    seal(native, ['plan.json', 'collection.json'])
    write_json(scores / 'plan.json', {'native_seal_sha256': data.file_hash(native / 'seal.json'),
                                      'method': 'witness', 'primary': 'witness_score'})
    write_json(scores / 'complete.json', {'records': scored})
    seal(scores, ['plan.json', 'complete.json', '1.npz', '2.npz'])
    return scores, native, prepared, tmp_path / 'outputs' / 'export'


def reseal(scores, native):
    seal(native, ['plan.json', 'collection.json'])
    plan = data.read_json(scores / 'plan.json')
    plan['native_seal_sha256'] = data.file_hash(native / 'seal.json')
    write_json(scores / 'plan.json', plan)
    records = data.read_json(scores / 'complete.json')['records']
    seal(scores, ['plan.json', 'complete.json'] + [r['file'] for r in records])


def change_npz(path, **changes):
    # Fixture mutation only; the production exporter never loads object members.
    with np.load(path, allow_pickle=True) as saved:
        values = {k: saved[k] for k in saved.files}
    np.savez(path, **(values | changes))


def test_cli_exports_independent_population_and_fuses_without_label_access(tmp_path, monkeypatch):
    scores, native, prepared, output = fixture(tmp_path, monkeypatch)
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name == 'annotations.jsonl' or path.name == 'absent-private-raw.json':
            raise AssertionError('Annotation/raw acquisition read')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    monkeypatch.setattr('sys.argv', ['r4-experiment', 'transfer', 'prepare-fusion',
        '--scores', str(scores), '--native', str(native), '--prepared', *map(str, prepared),
        '--group', '1', '--task', 'FinQA', '--output', str(output)])
    main()
    data.verify_seal(output)
    population = _load_population(output / 'population.npz')
    assert set(population) == set(AXES)
    np.testing.assert_array_equal(population['token_index'], [0, 2, 3])
    np.testing.assert_array_equal(population['token_id'], [10, 11, 12])
    assert set(population['sample_id']) == {'sample1'}
    receipt = data.read_json(output / 'export.json')
    assert receipt['all_test_records_verified'] == 2 and receipt['annotation_reads'] == 0
    assert not receipt['metrics_computed'] and not receipt['raw_acquisition_files_verified']
    fused = tmp_path / 'outputs' / 'fused'
    score(output / 'r4.npz', output / 'witness.npz', output / 'protocol.json', fused)
    data.verify_seal(fused)
    with np.load(fused / 'scores.npz') as values:
        # Hand-counted ascending midranks for the two preserved input arrays.
        np.testing.assert_allclose(values['score'], [.5, .5, .5])
    with pytest.raises(FileExistsError):
        export(scores, native, prepared, 1, output, 'FinQA')


def test_dropped_unselected_score_record_is_rejected_even_when_resealed(tmp_path, monkeypatch):
    scores, native, prepared, output = fixture(tmp_path, monkeypatch)
    complete = data.read_json(scores / 'complete.json')
    complete['records'] = complete['records'][:1]
    write_json(scores / 'complete.json', complete)
    reseal(scores, native)
    with pytest.raises(ValueError, match='ALL native TEST'):
        export(scores, native, prepared, 1, output, 'FinQA')
    assert not output.exists()


def test_unselected_cross_group_score_identity_is_rejected(tmp_path, monkeypatch):
    scores, native, prepared, output = fixture(tmp_path, monkeypatch)
    complete = data.read_json(scores / 'complete.json'); complete['records'][1]['group'] = 1
    write_json(scores / 'complete.json', complete); reseal(scores, native)
    with pytest.raises(ValueError, match='ALL native TEST'):
        export(scores, native, prepared, 1, output, 'FinQA')


@pytest.mark.parametrize('member', ['scores', 'native'])
def test_corrupt_seal_is_rejected(tmp_path, monkeypatch, member):
    scores, native, prepared, output = fixture(tmp_path, monkeypatch)
    path = (scores if member == 'scores' else native) / 'plan.json'
    path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ValueError, match='Sealed artifact changed'):
        export(scores, native, prepared, 1, output, 'FinQA')


@pytest.mark.parametrize('field,value', [('target', [0, 1, 3]), ('token_id', [10, 11, 13]),
                                       ('answer_index', [0, 0, 0]), ('frozen_score', [0., np.nan, 1.])])
def test_score_axis_or_nonfinite_change_is_rejected_after_resealing(tmp_path, monkeypatch, field, value):
    scores, native, prepared, output = fixture(tmp_path, monkeypatch)
    change_npz(scores / '2.npz', **{field: np.array(value)})
    complete = data.read_json(scores / 'complete.json')
    complete['records'][1]['sha256'] = data.file_hash(scores / '2.npz')
    write_json(scores / 'complete.json', complete); reseal(scores, native)
    with pytest.raises(ValueError, match='axes differ|Finite aligned'):
        export(scores, native, prepared, 1, output, 'FinQA')


def test_consistently_dropped_native_and_score_token_still_fails_full_axis_check(tmp_path, monkeypatch):
    scores, native, prepared, output = fixture(tmp_path, monkeypatch)
    change_npz(native / 'observations/1.npz', target=np.array([0, 3]),
               token_id=np.array([10, 12]), answer_index=np.array([1, 1]))
    collection = data.read_json(native / 'collection.json')
    collection['records'][1]['sha256'] = data.file_hash(native / 'observations/1.npz')
    write_json(native / 'collection.json', collection); reseal(scores, native)
    with pytest.raises(ValueError, match='valid token population'):
        export(scores, native, prepared, 1, output, 'FinQA')


@pytest.mark.parametrize('change', ['order', 'filtered', 'count'])
def test_prepared_identity_and_complete_population_are_enforced(tmp_path, monkeypatch, change):
    scores, native, prepared, output = fixture(tmp_path, monkeypatch)
    if change == 'order':
        prepared = list(reversed(prepared))
    else:
        manifest = data.read_json(prepared[1] / 'manifest.json')
        if change == 'filtered': manifest['selection']['ids'] = ['sample1']
        else: manifest['samples'] = 2
        write_json(prepared[1] / 'manifest.json', manifest)
        plan = data.read_json(native / 'plan.json')
        plan['groups'][1]['manifest_sha256'] = data.file_hash(prepared[1] / 'manifest.json')
        write_json(native / 'plan.json', plan); reseal(scores, native)
    with pytest.raises(ValueError, match='digest changed|unfiltered|population count'):
        export(scores, native, prepared, 1, output, 'FinQA')


def test_fit_group_and_output_escape_are_rejected(tmp_path, monkeypatch):
    scores, native, prepared, output = fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match='TEST group'):
        export(scores, native, prepared, 0, output, 'FinQA')
    with pytest.raises(ValueError, match='outputs/<run-id>'):
        export(scores, native, prepared, 1, tmp_path / 'outside', 'FinQA')


def test_unexpected_sealed_annotation_is_rejected_before_any_read(tmp_path, monkeypatch):
    scores, native, prepared, output = fixture(tmp_path, monkeypatch)
    (scores / 'annotations.jsonl').write_text('POISON', encoding='utf-8')
    seal(scores, ['plan.json', 'complete.json', '1.npz', '2.npz', 'annotations.jsonl'])
    original = Path.open

    def guarded(path, *args, **kwargs):
        if path.name == 'annotations.jsonl':
            raise AssertionError('Unexpected sealed annotation was opened')
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    with pytest.raises(ValueError, match='member set changed'):
        export(scores, native, prepared, 1, output, 'FinQA')
    assert not output.exists()

"""Synthetic final-label joins and independent source-bootstrap checks."""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from experiments import fusion_evaluation as evaluation
from experiments.data import file_hash, text_hash, verify_seal
from experiments.datasets import Sample
from experiments.rank_fusion import AXES, RULE, score


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


def bind_prepared(prepared):
    manifest = {'dataset': 'synthetic-character-spans',
                'inputs_sha256': file_hash(prepared / 'inputs.jsonl'),
                'annotations_sha256': file_hash(prepared / 'annotations.jsonl')}
    (prepared / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')


def fixture(tmp_path, monkeypatch, bad_bounds=False):
    monkeypatch.chdir(tmp_path)
    samples = [Sample(str(i), 'source-' + str(i // 2), 'test', 'QA',
                      'Use source for a response.', 'source', 'ab cd', 'generator')
               for i in range(4)]
    prepared = tmp_path / 'prepared'
    prepared.mkdir()
    write_rows(prepared / 'inputs.jsonl', [asdict(sample) for sample in samples])
    annotations = []
    labels = []
    for i, sample in enumerate(samples):
        positive = i % 2 == 0
        annotations.append({'id': sample.id, 'response_sha256': text_hash(sample.response),
                            'granularity': 'token', 'coverage': 'complete',
                            'spans': [{'start': 3 if positive else 0,
                                       'end': 5 if positive else 2, 'label': 1}]})
        labels.extend([not positive, positive])
    write_rows(prepared / 'annotations.jsonl', annotations)
    bind_prepared(prepared)
    common = {'sample_id': np.repeat([s.id for s in samples], 2),
              'source_id': np.repeat([s.source_id for s in samples], 2),
              'response_sha256': np.repeat([text_hash(s.response) for s in samples], 2),
              'token_index': np.tile([0, 2], 4), 'token_id': np.tile([10, 12], 4),
              'start': np.tile([0, 3], 4), 'end': np.tile([2, 5], 4)}
    if bad_bounds:
        common['end'][1] = 6
    population = tmp_path / 'population.npz'
    np.savez(population, **common)
    paths = []
    for method, values in [('r4', [8., 1., 2., 7., 6., 3., 4., 5.]),
                           ('witness', [2., 8., 1., 7., 3., 6., 5., 4.])]:
        path = tmp_path / (method + '.npz')
        np.savez(path, **common, score=np.array(values), method=np.array(method),
                 observer_id=np.array('fixture-observer'))
        paths.append(path)
    protocol = tmp_path / 'protocol.json'
    manifest = json.loads((prepared / 'manifest.json').read_text())
    protocol.write_text(json.dumps({**RULE, 'task': 'fixture', 'observer_id': 'fixture-observer',
                                   'population_manifest': str(population),
                                   'population_sha256': file_hash(population),
                                   'prepared_manifest_sha256': file_hash(prepared / 'manifest.json'),
                                   'prepared_inputs_sha256': manifest['inputs_sha256'],
                                   'prepared_annotations_sha256': manifest['annotations_sha256']}))
    sealed = tmp_path / 'outputs/scores'
    score(*paths, protocol, sealed)
    return sealed, prepared, np.array(labels, dtype=bool)


def rewrite_scores(sealed, **updates):
    with np.load(sealed / 'scores.npz', allow_pickle=False) as saved:
        values = {key: saved[key] for key in saved.files}
    values.update(updates)
    np.savez(sealed / 'scores.npz', **values)
    reseal(sealed)


def reseal(sealed):
    (sealed / 'seal.json').write_text(json.dumps(
        {name: file_hash(sealed / name) for name in ['scores.npz', 'run.json']}))


def refresh_prepared_binding(sealed, prepared):
    """Declare a new synthetic frozen fixture before testing a later join guard."""
    run = json.loads((sealed / 'run.json').read_text())
    protocol_path = Path(run['population_manifest']).parent / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    manifest = json.loads((prepared / 'manifest.json').read_text())
    protocol.update(prepared_manifest_sha256=file_hash(prepared / 'manifest.json'),
                    prepared_inputs_sha256=manifest['inputs_sha256'],
                    prepared_annotations_sha256=manifest['annotations_sha256'])
    protocol_path.write_text(json.dumps(protocol))
    run['input_sha256'][str(protocol_path)] = file_hash(protocol_path)
    (sealed / 'run.json').write_text(json.dumps(run))
    reseal(sealed)


def forbid_labels(monkeypatch, prepared):
    def reject(_path):
        pytest.fail('Annotation content was opened before pre-label validation passed')
    monkeypatch.setattr(evaluation, 'read_jsonl', reject)
    original_open = Path.open
    annotations = (prepared / 'annotations.jsonl').resolve()

    def guarded_open(path, *args, **kwargs):
        if path.resolve() == annotations:
            pytest.fail('Annotation bytes were opened before pre-label validation passed')
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded_open)


def test_three_independent_metrics_and_source_cluster_intervals(tmp_path, monkeypatch):
    sealed, prepared, labels = fixture(tmp_path, monkeypatch)
    before = {name: file_hash(sealed / name) for name in ['scores.npz', 'run.json', 'seal.json']}
    output = tmp_path / 'outputs/evaluated'
    report = evaluation.evaluate(sealed, prepared, output, repetitions=30, seed=17)
    verify_seal(output, expected={'evaluation.json'})
    with np.load(sealed / 'scores.npz', allow_pickle=False) as saved:
        vectors = {name: saved[key] for name, key in
                   [('r4', 'r4_score'), ('witness', 'witness_score'), ('fusion', 'score')]}
        sources = saved['source_id']
    for name, values in vectors.items():
        assert report['methods'][name]['auroc'] == roc_auc_score(labels, values)
        assert report['methods'][name]['ap'] == average_precision_score(labels, values)
    assert len({v['auroc'] for v in report['methods'].values()}) == 3
    for baseline in ['r4', 'witness']:
        draws = {'auroc': [], 'ap': []}
        rng = np.random.default_rng(17)
        groups = np.unique(sources)
        for _ in range(30):
            selected = np.concatenate([np.flatnonzero(sources == groups[g])
                                       for g in rng.integers(0, len(groups), len(groups))])
            for key, metric in [('auroc', roc_auc_score), ('ap', average_precision_score)]:
                draws[key].append(metric(labels[selected], vectors['fusion'][selected]) -
                                  metric(labels[selected], vectors[baseline][selected]))
        actual = report['paired_differences']['fusion_minus_' + baseline]
        for key in ['auroc', 'ap']:
            assert actual['point'][key] == report['methods']['fusion'][key] - report['methods'][baseline][key]
            np.testing.assert_allclose(actual['bootstrap'][key]['interval95'],
                                       np.quantile(draws[key], [.025, .975]), atol=1e-15)
    assert report['annotation_first_read_after_score_seal'] is True
    assert report['population']['tokens_scored'] == report['population']['tokens_evaluated'] == 8
    assert report['population']['sources_evaluated'] == 2
    assert before == {name: file_hash(sealed / name) for name in before}


def test_central_cli_uses_the_separate_final_evaluator(tmp_path, monkeypatch):
    from experiments.run import main

    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    output = tmp_path / 'outputs/cli-evaluation'
    monkeypatch.setattr('sys.argv', ['r4-experiment', 'transfer', 'evaluate-fusion',
                                    '--scores', str(sealed), '--prepared', str(prepared),
                                    '--output', str(output), '--repetitions', '7', '--seed', '19'])
    main()
    result = json.loads((output / 'evaluation.json').read_text())
    assert result['paired_differences']['fusion_minus_r4']['bootstrap']['repetitions'] == 7
    assert result['paired_differences']['fusion_minus_r4']['bootstrap']['seed'] == 19


def test_seal_mismatch_precedes_any_label_content_read(tmp_path, monkeypatch):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    with (sealed / 'scores.npz').open('ab') as stream:
        stream.write(b'changed-after-freeze')
    forbid_labels(monkeypatch, prepared)
    with pytest.raises(ValueError, match='Sealed artifact changed'):
        evaluation.evaluate(sealed, prepared, tmp_path / 'outputs/rejected')


def test_run_status_cannot_bypass_final_label_gate(tmp_path, monkeypatch):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    run = json.loads((sealed / 'run.json').read_text())
    run['status'] = 'not-frozen'
    (sealed / 'run.json').write_text(json.dumps(run))
    reseal(sealed)
    forbid_labels(monkeypatch, prepared)
    with pytest.raises(ValueError, match='frozen, unevaluated'):
        evaluation.evaluate(sealed, prepared, tmp_path / 'outputs/rejected')


@pytest.mark.parametrize('key', ['score', 'r4_score', 'witness_score'])
def test_nonfinite_scores_are_rejected_before_labels(tmp_path, monkeypatch, key):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    rewrite_scores(sealed, **{key: np.full(8, np.nan)})
    forbid_labels(monkeypatch, prepared)
    with pytest.raises(ValueError, match='finite and aligned'):
        evaluation.evaluate(sealed, prepared, tmp_path / 'outputs/rejected')


def test_unrelated_object_fields_are_not_opened(tmp_path, monkeypatch):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    rewrite_scores(sealed, poison_labels=np.array([{'must_not_load': True}], dtype=object))
    result = evaluation.evaluate(sealed, prepared, tmp_path / 'outputs/evaluated', repetitions=3)
    assert result['population']['tokens_evaluated'] == 8


@pytest.mark.parametrize('change', ['order', 'response', 'missing'])
def test_complete_prepared_identity_is_checked_before_labels(tmp_path, monkeypatch, change):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (prepared / 'inputs.jsonl').read_text().splitlines()]
    if change == 'order':
        rows[0], rows[1] = rows[1], rows[0]
    elif change == 'response':
        rows[0]['response'] = 'xy zz'
    else:
        rows.pop()
    write_rows(prepared / 'inputs.jsonl', rows)
    bind_prepared(prepared)
    refresh_prepared_binding(sealed, prepared)
    forbid_labels(monkeypatch, prepared)
    with pytest.raises(ValueError, match='sample IDs/order|source/response identity'):
        evaluation.evaluate(sealed, prepared, tmp_path / 'outputs/rejected')


def test_response_bounds_checked_on_frozen_original_offsets(tmp_path, monkeypatch):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch, bad_bounds=True)
    forbid_labels(monkeypatch, prepared)
    with pytest.raises(ValueError, match='character bounds'):
        evaluation.evaluate(sealed, prepared, tmp_path / 'outputs/rejected')


@pytest.mark.parametrize('change', ['response_hash', 'duplicate', 'granularity'])
def test_released_annotation_identity_and_granularity(tmp_path, monkeypatch, change):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (prepared / 'annotations.jsonl').read_text().splitlines()]
    if change == 'response_hash':
        rows[0]['response_sha256'] = text_hash('another response')
    elif change == 'duplicate':
        rows[-1] = rows[0]
    else:
        rows[0]['granularity'] = 'sentence'
    write_rows(prepared / 'annotations.jsonl', rows)
    bind_prepared(prepared)
    refresh_prepared_binding(sealed, prepared)
    with pytest.raises(ValueError, match='Annotation response hash|unique|native token'):
        evaluation.evaluate(sealed, prepared, tmp_path / 'outputs/rejected', repetitions=3)
    assert not (tmp_path / 'outputs/rejected').exists()


def test_undefined_spans_change_metrics_only(tmp_path, monkeypatch):
    sealed, prepared, labels = fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (prepared / 'annotations.jsonl').read_text().splitlines()]
    rows[0]['spans'].append({'start': 0, 'end': 2, 'label': None})
    write_rows(prepared / 'annotations.jsonl', rows)
    bind_prepared(prepared)
    refresh_prepared_binding(sealed, prepared)
    original_score_hash = file_hash(sealed / 'scores.npz')
    result = evaluation.evaluate(sealed, prepared, tmp_path / 'outputs/evaluated', repetitions=3)
    assert result['population']['tokens_scored'] == 8
    assert result['population']['tokens_evaluated'] == 7
    assert result['population']['unknown_tokens_excluded'] == 1
    assert file_hash(sealed / 'scores.npz') == original_score_hash
    with np.load(sealed / 'scores.npz', allow_pickle=False) as saved:
        assert result['methods']['fusion']['auroc'] == roc_auc_score(labels[1:], saved['score'][1:])


def test_changed_annotation_bytes_during_metrics_are_not_published(tmp_path, monkeypatch):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    original = evaluation.ranking
    changed = False

    def mutate(labels, values):
        nonlocal changed
        result = original(labels, values)
        if not changed:
            with (prepared / 'annotations.jsonl').open('ab') as stream:
                stream.write(b'\n')
            changed = True
        return result

    monkeypatch.setattr(evaluation, 'ranking', mutate)
    output = tmp_path / 'outputs/rejected'
    with pytest.raises(ValueError, match='input changed'):
        evaluation.evaluate(sealed, prepared, output, repetitions=3)
    assert not output.exists()


def test_output_containment_and_nonoverwrite_precede_labels(tmp_path, monkeypatch):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    existing = tmp_path / 'outputs/existing'
    existing.mkdir()
    forbid_labels(monkeypatch, prepared)
    with pytest.raises(ValueError, match='current-project outputs'):
        evaluation.evaluate(sealed, prepared, tmp_path / 'outside')
    with pytest.raises(FileExistsError):
        evaluation.evaluate(sealed, prepared, existing)


def test_annotation_alias_cannot_enter_inference_binding_roster(tmp_path, monkeypatch):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    run = json.loads((sealed / 'run.json').read_text())
    annotation_path = prepared / 'annotations.jsonl'
    run['input_sha256'][str(annotation_path)] = file_hash(annotation_path)
    (sealed / 'run.json').write_text(json.dumps(run))
    reseal(sealed)
    forbid_labels(monkeypatch, prepared)
    with pytest.raises(ValueError, match='binding roster'):
        evaluation.evaluate(sealed, prepared, tmp_path / 'outputs/rejected')


def test_replaced_manifest_cannot_reauthorize_annotation_changes(tmp_path, monkeypatch):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    rows = [json.loads(line) for line in (prepared / 'annotations.jsonl').read_text().splitlines()]
    rows[0]['spans'] = []
    write_rows(prepared / 'annotations.jsonl', rows)
    bind_prepared(prepared)
    forbid_labels(monkeypatch, prepared)
    with pytest.raises(ValueError, match='Frozen input binding changed'):
        evaluation.evaluate(sealed, prepared, tmp_path / 'outputs/rejected')


def test_missing_population_is_rejected_before_annotation_bytes(tmp_path, monkeypatch):
    sealed, prepared, _ = fixture(tmp_path, monkeypatch)
    population = tmp_path / 'population.npz'
    population.unlink()
    forbid_labels(monkeypatch, prepared)
    with pytest.raises(FileNotFoundError):
        evaluation.evaluate(sealed, prepared, tmp_path / 'outputs/rejected')

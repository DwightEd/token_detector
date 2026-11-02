"""Synthetic whole-panel CPU review gates; no real model, cases, or truth."""

from copy import deepcopy
import builtins
import json
from pathlib import Path

import pytest

from experiments import data, verdict_evaluation as module
from experiments import verdict_qualification as observer


MODEL = {'model_id': 'synthetic-finite-label-instrument',
         'model_files_sha256': {'weights.bin': 'a' * 64}}
HELPERS = {name: digit * 64 for name, digit in zip((
    'experiments/verdict_qualification.py', 'experiments.research/semantic_verdict.py',
    'experiments/semantic_qualification.py', 'experiments.research/semantic_probe.py'), 'bcde')}
OLD_SOURCE = {'source_sha256': {**HELPERS, 'experiments/run.py': '1' * 64}, 'release': 'observer'}
NEW_SOURCE = {'source_sha256': {**HELPERS, 'experiments/run.py': '2' * 64,
                              'experiments/verdict_evaluation.py': '3' * 64}, 'release': 'reader'}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def change_json(path, change):
    value = data.read_json(path)
    change(value)
    write(path, value)


def seal(root):
    members = [p for p in root.rglob('*') if p.is_file() and p != root / 'seal.json']
    write(root / 'seal.json', {str(p.relative_to(root)): data.file_hash(p) for p in members})


def reseal_observations(root):
    for parent in ('cases', 'requests'):
        for child in (root / parent).iterdir():
            seal(child)
    seal(root)


class FakeScorer:
    model_identity = MODEL

    def __init__(self, rejected=False, tied=False, ordered_seconds=False):
        self.rejected, self.tied, self.ordered_seconds, self.calls = rejected, tied, ordered_seconds, []

    def count_input_tokens(self, messages):
        payload = json.loads(messages[-1]['content'])
        return 4097 if self.rejected and payload['response'] == 'Answer 0: A has X.' else 20

    def validate_labels(self, messages, labels):
        return {'input_tokens': self.count_input_tokens(messages),
                'label_token_ids': dict(zip(labels, [10, 11, 12])),
                'label_decodes': dict(zip(labels, labels)), 'prefix_alignment_verified': True}

    def score_labels(self, messages, labels):
        self.calls.append(deepcopy(messages))
        assignment = json.loads(messages[0]['content'][messages[0]['content'].rfind('{'):])
        scores = {'supported': 1.0, 'contradicted': 0.0, 'unresolved': 3.0}
        return {**self.validate_labels(messages, labels),
                'label_logits': {label: 0.0 if self.tied else scores[assignment[label]] for label in labels},
                'full_vocab_logsumexp': 5.0, 'output_tokens': 0,
                'actual_forward_passes': 1,
                'seconds': ([1e16, 1., 1., 1., 1., 1.][len(self.calls) - 1]
                            if self.ordered_seconds else 0.125)}


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    monkeypatch.setattr(observer, 'research_identity', lambda: deepcopy(OLD_SOURCE))
    monkeypatch.setattr(observer, 'completed_identity', lambda old: old)
    monkeypatch.setattr(module, 'research_identity', lambda: deepcopy(NEW_SOURCE))
    monkeypatch.setattr(module, 'completed_identity', lambda old: old)

    def build(*, unique=False, rejected=False, tied=False, ordered_seconds=False):
        panel = tmp_path / 'panel'
        panel.mkdir()
        rows = [{'id': 'original-' + str(i), 'source_id': 'source-' + str(i),
                 'split': 'test', 'task': 'QA', 'prompt': 'Question? Read Evidence X. Answer fully.',
                 'source': 'Evidence X.', 'response': 'Answer ' + str(i if unique else 0) + ': A has X.',
                 'generator': 'original-generator'} for i in range(18)]
        (panel / 'inputs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        released = []
        for i, row in enumerate(rows):
            released.append({'id': row['id'], 'source_id': row['source_id'],
                             'cohort': 'cohort-' + str(i % 2),
                             'response_sha256': data.text_hash(row['response']),
                             'human_mechanism_expectation': {'mechanism': 'synthetic',
                                 'expectation': 'Old expectation retained for human review only.',
                                 'label_scope_note': 'Native units are not per-word factual truth.'},
                             'released_annotation_record': {'id': row['id'],
                                 'response_sha256': data.text_hash(row['response']),
                                 'granularity': 'token' if i % 2 == 0 else 'sentence',
                                 'coverage': 'complete' if i % 2 == 0 else 'annotated_sentences',
                                 'spans': [{'start': 0, 'end': len(row['response']), 'label': i % 2}]},
                             'original_annotation': {'path': '/NEVER_OPEN/annotations.jsonl',
                                                     'record_index': i}})
        (panel / 'evaluation_only.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in released), encoding='utf-8')
        manifest = {'schema': module.PANEL_SCHEMA,
                    'counts': {'total_inputs': 18, 'cohorts': {'cohort-0': 9, 'cohort-1': 9}},
                    'selection': {'selected_id_order_sha256': module._digest([row['id'] for row in rows])},
                    'input_rows': [{**{k: row[k] for k in ('id', 'source_id', 'generator')},
                                    'original_task': row['task'], 'cohort': 'cohort-' + str(i % 2),
                                    **{name + '_sha256': data.text_hash(row[name])
                                       for name in ('prompt', 'source', 'response')}} for i, row in enumerate(rows)],
                    'inputs_sha256': data.file_hash(panel / 'inputs.jsonl'),
                    'evaluation_only_sha256': data.file_hash(panel / 'evaluation_only.jsonl')}
        write(panel / 'manifest.json', manifest)
        seal(panel)
        observation_protocol = tmp_path / 'observation_protocol.json'
        write(observation_protocol, observer.DEFAULT_PROTOCOL)
        observations = tmp_path / 'outputs' / 'observations'
        scorer = FakeScorer(rejected, tied, ordered_seconds)
        observer.qualify(panel / 'inputs.jsonl', observation_protocol, observations, scorer, MODEL)
        protocol = tmp_path / 'evaluation_protocol.json'
        write(protocol, module.DEFAULT_PROTOCOL)
        return (observations, panel, tmp_path / 'outputs' / 'evaluation', protocol), scorer
    return build


def first_case(root):
    return root / 'cases' / data.text_hash('original-0') / 'case.json'


def first_request(root):
    return next((root / 'requests').iterdir())


def guard_release(args, monkeypatch, *, hashes=False):
    """Any failure before the release gate must avoid both manifest and truth."""
    original_open, original_hash = Path.open, data.file_hash
    forbidden = {args[1] / 'manifest.json', args[1] / 'evaluation_only.jsonl'}
    def guarded_open(path, *pos, **kw):
        assert path not in forbidden, 'Released material opened before complete validation'
        return original_open(path, *pos, **kw)
    def guarded_hash(path):
        assert Path(path) not in forbidden, 'Released material hashed before complete validation'
        if hashes:
            pytest.fail('Path inventory must reject this artifact before any file hash')
        return original_hash(path)
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(data, 'file_hash', guarded_hash)


def test_complete_unique_roster_and_separate_evaluator_identity(make_run):
    args, scorer = make_run(unique=True)
    assert len(scorer.calls) == 108
    result = module.evaluate(*args)
    assert result['state'] == 'review_sealed_human_review_pending'
    assert [r['id'] for r in result['records']] == ['original-' + str(i) for i in range(18)]
    assert result['model_calls'] == 0 and result['scalar_scores_created'] is False
    assert result['observer_source_identity_sha256'] == module._digest(OLD_SOURCE)
    assert result['evaluator_source_identity_sha256'] == module._digest(NEW_SOURCE)
    assert result['observer_source_identity_sha256'] != result['evaluator_source_identity_sha256']
    summary = result['coverage']
    assert summary['whole_case_denominator'] == summary['review_cases'] == summary['scored_cases'] == 18
    assert summary['scored_permutations'] == 108 and summary['unanimous_unique_argmax_cases'] == 18
    assert summary['unique_permutation_argmax_counts'] == {'unresolved': 108}
    assert summary['cohorts']['cohort-0']['cases'] == summary['cohorts']['cohort-1']['cases'] == 9
    assert summary['off_label_mass']['count'] == 108
    assert sum(summary['full_vocab_class_mass_means'].values()) + summary['off_label_mass']['mean'] == pytest.approx(1.0)
    assert summary['released_annotation_granularities'] == {'token': 9, 'sentence': 9}
    assert summary['automatic_accuracy_created'] is summary['auroc_created'] is False
    data.verify_seal(args[2], required={'summary.json', 'evaluator_source_identity.json', 'observer_identity.json'})
    packet = data.read_json(args[2] / result['records'][1]['file'])
    assert packet['original_inputs'] == {'prompt': 'Question? Read Evidence X. Answer fully.',
                                         'source': 'Evidence X.', 'response': 'Answer 1: A has X.'}
    assert packet['unscored_unknown'] is False and packet['verdict']['class_scores']['unresolved'] > .8
    assert len(packet['observations']) == len(packet['verdict']['permutations']) == 6
    assert packet['annotation_scope']['granularity'] == 'sentence'
    assert packet['annotation_scope']['per_token_fact_truth_created'] is False
    assert packet['released_evaluation_record']['human_mechanism_expectation']['expectation']
    assert packet['human_semantic_review'] == {'state': 'pending', 'automatic_correctness_judgment': False}
    assert 'accuracy' not in result and 'auroc' not in result


def test_exact_reuse_retains_all108_logical_owners(make_run):
    args, scorer = make_run(unique=False)
    assert len(scorer.calls) == 6
    complete = data.read_json(args[0] / 'complete.json')
    assert complete['costs']['logical_requests'] == 108
    assert complete['costs']['unique_requests'] == 6 and complete['costs']['exact_request_reuses'] == 102
    result = module.evaluate(*args)
    assert len(result['records']) == 18 and result['coverage']['scored_permutations'] == 108


def test_forward_seconds_use_collector_insertion_order(make_run, monkeypatch):
    args, _ = make_run(ordered_seconds=True)
    plan = data.read_json(args[0] / 'plan.json')
    keys = list(dict.fromkeys(item['request_key'] for row in plan['records'] for item in row['requests']))
    seconds = {key: data.read_json(args[0] / 'requests' / key / 'response.json')['seconds'] for key in keys}
    expected = sum(seconds[key] for key in keys)
    assert data.read_json(args[0] / 'complete.json')['costs']['seconds_forward'] == expected
    # Keep the historical addition order even on Python releases whose builtin
    # sum uses compensated floating arithmetic and masks the numeric difference.
    observed = []
    def ordered_sum(values, *pos):
        values = list(values)
        if values and all(type(value) is float for value in values):
            assert values == [seconds[key] for key in keys]
            observed.append(values)
        return builtins.sum(values, *pos)
    monkeypatch.setattr(module, 'sum', ordered_sum, raising=False)
    assert module.evaluate(*args)['coverage']['review_cases'] == 18
    assert observed


def test_unscored_unknown_is_distinct_from_semantic_unresolved(make_run):
    args, scorer = make_run(unique=True, rejected=True)
    assert len(scorer.calls) == 102
    result = module.evaluate(*args)
    assert result['coverage']['scored_cases'] == 17
    assert result['coverage']['unscored_unknown_cases'] == 1
    unknown = data.read_json(args[2] / result['records'][0]['file'])
    assert unknown['unscored_unknown'] is True
    assert unknown['verdict'] == {'status': 'unknown', 'reason': 'input_token_limit', 'class_scores': None}
    assert all(row['execution']['status'] == 'input_rejected' for row in unknown['observations'])
    resolved = data.read_json(args[2] / result['records'][1]['file'])
    assert resolved['unscored_unknown'] is False and resolved['verdict']['class_scores']['unresolved'] > .8


def test_ties_are_retained_without_a_class_name_tiebreak(make_run):
    args, _ = make_run(tied=True)
    summary = module.evaluate(*args)['coverage']
    assert summary['tied_permutations'] == 108 and summary['cases_with_tied_permutations'] == 18
    assert summary['unanimous_unique_argmax_cases'] == 0
    assert summary['unique_permutation_argmax_counts'] == {}


def test_evaluator_is_frozen_before_any_manifest_or_gold_access(make_run, monkeypatch):
    args, _ = make_run()
    original_open, original_hash = Path.open, data.file_hash
    original_completion = module._completion
    seen, checked = [], []
    def completion(*pos, **kw):
        value = original_completion(*pos, **kw)
        assert value == data.read_json(args[0] / 'complete.json')
        checked.append('all_observation_reductions_and_costs')
        return value
    monkeypatch.setattr(module, '_completion', completion)
    def check(path):
        if Path(path) in {args[1] / 'manifest.json', args[1] / 'evaluation_only.jsonl'}:
            assert checked == ['all_observation_reductions_and_costs']
            plan = data.read_json(args[2] / 'plan.json')
            assert plan['state'] == 'evaluator_frozen_before_release'
            assert plan['evaluation_material_accessed'] is False
            assert data.read_json(args[2] / 'protocol.json') == module.DEFAULT_PROTOCOL
            assert data.read_json(args[2] / 'evaluator_source_identity.json') == NEW_SOURCE
            assert data.read_json(args[2] / 'observer_identity.json')['source_identity'] == OLD_SOURCE
            seen.append(Path(path).name)
    def checked_open(path, *pos, **kw):
        check(path)
        return original_open(path, *pos, **kw)
    def checked_hash(path):
        check(path)
        return original_hash(path)
    monkeypatch.setattr(Path, 'open', checked_open)
    monkeypatch.setattr(data, 'file_hash', checked_hash)
    module.evaluate(*args)
    assert {'manifest.json', 'evaluation_only.jsonl'} <= set(seen)


@pytest.mark.parametrize('failure', ['root_alias', 'metadata_alias', 'unsafe_root_seal', 'unsafe_child_seal'])
def test_path_inventory_rejects_before_hashing_any_file(make_run, monkeypatch, failure):
    args, _ = make_run()
    if failure == 'root_alias':
        args = (args[1], *args[1:])
    elif failure == 'metadata_alias':
        path = args[0] / 'plan.json'
        path.unlink()
        try:
            path.symlink_to(args[1] / 'evaluation_only.jsonl')
        except OSError:
            pytest.skip('This platform does not allow symbolic links')
    elif failure == 'unsafe_root_seal':
        change_json(args[0] / 'seal.json', lambda value: value.update({'..\\panel\\evaluation_only.jsonl': 'a' * 64}))
    else:
        change_json(first_request(args[0]) / 'seal.json', lambda value: value.update({'..\\evaluation_only.jsonl': 'a' * 64}))
    guard_release(args, monkeypatch, hashes=True)
    with pytest.raises(ValueError):
        module.evaluate(*args)


@pytest.mark.parametrize('failure', ['missing_case', 'missing_request', 'unsealed_request', 'missing_id', 'wrong_order',
                                    'input_hash', 'model', 'source', 'protocol', 'request_ownership', 'raw',
                                    'prefix', 'token_ids', 'logits', 'case_reduction', 'coverage', 'costs'])
def test_incomplete_or_tampered_archive_rejects_before_release(make_run, monkeypatch, failure):
    args, _ = make_run()
    root = args[0]
    if failure in {'missing_case', 'missing_request'}:
        path = first_case(root) if failure == 'missing_case' else first_request(root) / 'response.json'
        path.unlink()
    elif failure == 'unsealed_request':
        (first_request(root) / 'seal.json').unlink()
    elif failure in {'missing_id', 'wrong_order', 'input_hash', 'request_ownership'}:
        def change(value):
            if failure == 'missing_id':
                value['records'].pop()
            elif failure == 'wrong_order':
                value['records'].reverse()
            elif failure == 'input_hash':
                value['records'][0]['input_text_sha256']['prompt'] = '0' * 64
            else:
                value['records'][0]['requests'][0]['request_key'] = '0' * 64
        change_json(root / 'plan.json', change)
    elif failure == 'model':
        change_json(root / 'model_identity.json', lambda value: value.update(model_id='other-instrument'))
    elif failure == 'source':
        change_json(root / 'source_identity.json', lambda value: value.update(release='changed'))
    elif failure == 'protocol':
        change_json(root / 'protocol.json', lambda value: value.update(notes='changed'))
    elif failure == 'raw':
        change_json(first_request(root) / 'backend_raw.json', lambda value: value['value'].update(seconds=99))
    elif failure in {'prefix', 'token_ids', 'logits'}:
        path = first_request(root)
        response = data.read_json(path / 'response.json')
        if failure == 'prefix':
            response['prefix_alignment_verified'] = False
        elif failure == 'token_ids':
            response['label_token_ids']['C'] = response['label_token_ids']['B']
        else:
            response.update(label_logits=dict.fromkeys('ABC', 0.0), full_vocab_logsumexp=0.0)
        write(path / 'response.json', response)
        write(path / 'backend_raw.json', {'value': {k: v for k, v in response.items() if k != 'status'}})
    elif failure == 'case_reduction':
        change_json(first_case(root), lambda value: value['verdict']['class_scores'].update(supported=.99))
    elif failure == 'coverage':
        change_json(root / 'complete.json', lambda value: value['coverage'].update(scored_cases=17))
    else:
        change_json(root / 'complete.json', lambda value: value['costs'].update(actual_forward_passes=107))
    # These mutations pass cryptographic seal checks, so semantic validation must
    # detect them without touching either released material file.
    if failure not in {'missing_case', 'missing_request', 'unsealed_request'}:
        reseal_observations(root)
    guard_release(args, monkeypatch)
    with pytest.raises((ValueError, RuntimeError)):
        module.evaluate(*args)


@pytest.mark.parametrize('helper', list(HELPERS))
def test_changed_reused_helper_requires_compatible_reader_before_release(make_run, monkeypatch, helper):
    args, _ = make_run()
    changed = deepcopy(NEW_SOURCE)
    changed['source_sha256'][helper] = 'f' * 64
    monkeypatch.setattr(module, 'research_identity', lambda: changed)
    guard_release(args, monkeypatch)
    with pytest.raises(ValueError, match='incompatible'):
        module.evaluate(*args)


@pytest.mark.parametrize('when', ['interpretation', 'evaluator_freeze'])
def test_root_reseal_after_initial_snapshot_rejects_before_release(make_run, monkeypatch, when):
    args, _ = make_run()
    root = args[0]
    def mutate():
        # Identical content with a different valid serialization is still a
        # different archive seal, even though verify_seal alone would accept it.
        value = data.read_json(root / 'seal.json')
        (root / 'seal.json').write_text(json.dumps(value, indent=3) + '\n', encoding='utf-8')
    if when == 'interpretation':
        original = module._roster
        def roster(*pos, **kw):
            result = original(*pos, **kw)
            mutate()
            return result
        monkeypatch.setattr(module, '_roster', roster)
    else:
        original = data.write_json
        def writing(path, value):
            original(path, value)
            if Path(path) == args[2] / 'observer_identity.json':
                mutate()
        monkeypatch.setattr(data, 'write_json', writing)
    guard_release(args, monkeypatch)
    with pytest.raises(ValueError, match='seal changed'):
        module.evaluate(*args)


@pytest.mark.parametrize('when', ['review', 'complete', 'output_seal'])
def test_postexport_root_reseal_is_detected(make_run, monkeypatch, when):
    args, _ = make_run()
    original = data.write_json
    def writing(path, value):
        original(path, value)
        path = Path(path)
        trigger = (when == 'review' and path == args[2] / 'summary.json'
                   or when == 'complete' and path == args[2] / 'complete.json'
                   or when == 'output_seal' and path == args[2] / 'seal.json')
        if trigger:
            seal_value = data.read_json(args[0] / 'seal.json')
            (args[0] / 'seal.json').write_text(json.dumps(seal_value, indent=3) + '\n', encoding='utf-8')
    monkeypatch.setattr(data, 'write_json', writing)
    with pytest.raises(ValueError, match='seal changed'):
        module.evaluate(*args)
    assert not (args[2] / 'complete.json').exists()
    assert not (args[2] / 'seal.json').exists()


def test_existing_evaluation_is_never_overwritten(make_run, monkeypatch):
    args, _ = make_run()
    module.evaluate(*args)
    before = data.file_hash(args[2] / 'seal.json')
    guard_release(args, monkeypatch)
    with pytest.raises(FileExistsError, match='not be overwritten'):
        module.evaluate(*args)
    assert data.file_hash(args[2] / 'seal.json') == before


@pytest.mark.parametrize('failure', ['wrong_source', 'wrong_response', 'missing_id', 'bad_span',
                                    'manifest_schema', 'manifest_input_binding'])
def test_released_identity_scope_and_manifest_binding_are_validated(make_run, failure):
    args, _ = make_run()
    panel = args[1]
    if failure.startswith('manifest_'):
        change_json(panel / 'manifest.json', lambda value: value.update(schema='constructed-controls-v1')
                    if failure == 'manifest_schema' else value['input_rows'][0].update(prompt_sha256='0' * 64))
    else:
        rows = [json.loads(line) for line in (panel / 'evaluation_only.jsonl').read_text(encoding='utf-8').splitlines()]
        if failure == 'wrong_source':
            rows[0]['source_id'] = 'another-source'
        elif failure == 'wrong_response':
            rows[0]['released_annotation_record']['response_sha256'] = '0' * 64
        elif failure == 'missing_id':
            rows.pop()
        else:
            rows[0]['released_annotation_record']['spans'][0]['end'] = 10000
        (panel / 'evaluation_only.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        change_json(panel / 'manifest.json', lambda value: value.update(
            evaluation_only_sha256=data.file_hash(panel / 'evaluation_only.jsonl')))
    seal(panel)
    with pytest.raises(ValueError):
        module.evaluate(*args)
    # Release was authorized only after the full observation gate; malformed
    # evaluation material cannot produce a successful exported review.
    assert (args[2] / 'plan.json').is_file()
    assert not (args[2] / 'complete.json').exists()
    assert not (args[2] / 'seal.json').exists()

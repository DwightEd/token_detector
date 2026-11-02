"""Synthetic post-seal evaluation gates and whole-population review artifacts."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import data, semantic_evaluation as module
from experiments import semantic_qualification as observer
from experiments.run import main


MODEL = {'model_id': 'synthetic-instrument', 'model_files_sha256': {'weights.bin': 'a' * 64}}
HELPERS = {'experiments/semantic_qualification.py': 'b' * 64,
           'experiments.research/semantic_probe.py': 'f' * 64}
OLD_SOURCE = {'source_sha256': dict(HELPERS), 'release': 'observer94'}
NEW_SOURCE = {'source_sha256': {**HELPERS, 'experiments/semantic_evaluation.py': 'c' * 64},
              'release': 'cpu-evaluator'}
SOURCE, RESPONSE = 'Evidence X.', 'A has X.'


class SyntheticGenerator:
    model_identity = MODEL

    def __init__(self, proposal='normal', observations='normal'):
        self.proposal, self.observations = proposal, observations

    def count_input_tokens(self, messages):
        return 20

    def __call__(self, messages, *, max_new_tokens):
        payload = json.loads(messages[-1]['content'])
        finish = 'eos'
        if 'response' in payload:
            text = payload['response']
            start = text.index('X')
            slots = [{'role': 'actor', 'start': start, 'end': start + 1, 'text': 'X'}]
            claims = [{'start': 0, 'end': len(text), 'text': text,
                       'slots': [] if self.proposal == 'no_slots' else slots}]
            if self.proposal == 'empty':
                claims = []
            elif self.proposal == 'mixed_invalid':
                claims.append({'text': 'absent from answer', 'slots': []})
            value = {'claims': claims}
            if self.proposal == 'invalid_limited':
                value, finish = '{broken', 'max_new_tokens'
        elif self.observations == 'mixed' and 'claim' in payload:
            value = 'invalid JSON'
        elif self.observations == 'mixed' and 'claimed_value' not in payload:
            value = {'status': 'unknown', 'answer': None, 'quotes': [], 'reason': 'ambiguous source'}
            finish = 'max_new_tokens'
        else:
            value = {'status': 'answerable', 'answer': 'supported' if 'claim' in payload else 'an alias for X',
                     'quotes': [{'text': SOURCE}], 'reason': ''}
        text = value if isinstance(value, str) else json.dumps(value)
        return {'text': text, 'raw_text': text + '<eos>', 'input_tokens': 20,
                'output_tokens': 12, 'finish_reason': finish}


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def seal(root):
    members = [p for p in root.rglob('*') if p.is_file() and p != root / 'seal.json']
    write(root / 'seal.json', {str(p.relative_to(root)): data.file_hash(p) for p in members})


@pytest.fixture
def make_run(tmp_path, monkeypatch):
    monkeypatch.setattr(observer, 'research_identity', lambda: deepcopy(OLD_SOURCE))
    monkeypatch.setattr(observer, 'completed_identity', lambda old: old)
    monkeypatch.setattr(module, 'research_identity', lambda: deepcopy(NEW_SOURCE))
    monkeypatch.setattr(module, 'completed_identity', lambda old: old)

    def build(*, proposal='normal', observations='normal', response=RESPONSE, prompt=None):
        panel = tmp_path / 'panel'; panel.mkdir()
        inputs = [{'id': 'original-' + str(i), 'source_id': 'source-' + str(i), 'split': 'test',
                   'task': 'QA', 'prompt': 'Read ' + SOURCE if prompt is None else prompt,
                   'source': SOURCE, 'response': response,
                   'generator': 'original-generator'} for i in range(18)]
        (panel / 'inputs.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in inputs), encoding='utf-8')
        annotations = []
        for i, sample in enumerate(inputs):
            annotation = {'id': sample['id'], 'response_sha256': data.text_hash(response),
                          'granularity': 'sentence' if i % 2 else 'token',
                          'coverage': 'annotated_sentences' if i % 2 else 'complete',
                          'spans': [{'start': 0, 'end': len(response), 'label': 1}],
                          'label_provenance': 'synthetic broad native unit'}
            annotations.append({'id': sample['id'], 'source_id': sample['source_id'], 'cohort': 'synthetic',
                                'response_sha256': data.text_hash(response),
                                'selection_role': 'prescribed_known_exploratory_case',
                                'access_rule': 'after sealed observations only',
                                'human_mechanism_expectation': {'mechanism': 'synthetic_binding',
                                    'expectation': 'Human review only.', 'label_scope_note': 'Broad label is not word truth.'},
                                'released_annotation_record': annotation,
                                'original_annotation': {'path': '/NEVER_OPEN/annotations.jsonl',
                                    'file_sha256': 'd' * 64, 'record_index': i, 'raw_line_sha256': 'e' * 64}})
        (panel / 'evaluation_only.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in annotations), encoding='utf-8')
        manifest = {'schema': module.PANEL_SCHEMA, 'scope': 'Synthetic previously exposed cases.',
                    'counts': {'total_inputs': 18, 'cohorts': {'synthetic': 18}},
                    'selection': {'selected_id_order_sha256': module._digest([r['id'] for r in inputs])},
                    'input_rows': [{**{k: r[k] for k in ('id', 'source_id', 'generator')},
                                    'original_task': r['task'], 'cohort': 'synthetic',
                                    **{name + '_sha256': data.text_hash(r[name])
                                       for name in ('prompt', 'source', 'response')}} for r in inputs],
                    'inputs_sha256': data.file_hash(panel / 'inputs.jsonl'),
                    'evaluation_only_sha256': data.file_hash(panel / 'evaluation_only.jsonl')}
        write(panel / 'manifest.json', manifest)
        seal(panel)
        observation_protocol = tmp_path / 'observation_protocol.json'
        write(observation_protocol, observer.DEFAULT_PROTOCOL)
        observations_root = tmp_path / 'outputs' / 'observations'
        observer.qualify(panel / 'inputs.jsonl', observation_protocol, observations_root,
                         SyntheticGenerator(proposal, observations), MODEL)
        protocol = tmp_path / 'evaluation_protocol.json'; write(protocol, module.DEFAULT_PROTOCOL)
        return observations_root, panel, tmp_path / 'outputs' / 'evaluation', protocol
    return build


def first_case(observations):
    return observations / 'cases' / data.text_hash('original-0') / 'case.json'


def guard_gold(monkeypatch):
    original_open, original_hash = Path.open, data.file_hash
    def guarded_open(path, *args, **kwargs):
        assert path.name not in {'evaluation_only.jsonl', 'annotations.jsonl'}, 'Gold opened before release'
        return original_open(path, *args, **kwargs)
    def guarded_hash(path):
        assert Path(path).name not in {'evaluation_only.jsonl', 'annotations.jsonl'}, 'Gold hashed before release'
        return original_hash(path)
    monkeypatch.setattr(Path, 'open', guarded_open)
    monkeypatch.setattr(data, 'file_hash', guarded_hash)


def change_json(path, change):
    value = data.read_json(path)
    change(value)
    write(path, value)


def test_finished_panel_retains_every_id_and_annotation_scope(make_run):
    args = make_run()
    result = module.evaluate(*args)
    assert result['state'] == 'review_sealed_human_review_pending'
    assert [r['id'] for r in result['records']] == ['original-' + str(i) for i in range(18)]
    assert result['coverage']['whole_case_denominator'] == result['coverage']['review_cases'] == 18
    assert result['coverage']['released_annotation_granularities'] == {'token': 9, 'sentence': 9}
    assert result['model_calls'] == 0 and result['scalar_scores_created'] is False
    assert result['observer_source_identity_sha256'] == module._digest(OLD_SOURCE)
    assert result['evaluator_source_identity_sha256'] == module._digest(NEW_SOURCE)
    assert result['observer_source_identity_sha256'] != result['evaluator_source_identity_sha256']
    data.verify_seal(args[2], required={'summary.json', 'evaluator_source_identity.json', 'observer_identity.json'})
    review = data.read_json(args[2] / result['records'][1]['file'])
    assert review['original_inputs'] == {'prompt': 'Read ' + SOURCE, 'source': SOURCE, 'response': RESPONSE}
    assert review['pre_generation_context'] == 'Read ' + observer.probe.SOURCE_PLACEHOLDER
    assert review['pre_generation_context_sha256'] == data.text_hash(review['pre_generation_context'])
    assert review['annotation_scope']['granularity'] == 'sentence'
    assert review['annotation_scope']['per_token_fact_truth_created'] is False
    assert review['released_evaluation_record']['released_annotation_record']['spans'] == [
        {'start': 0, 'end': len(RESPONSE), 'label': 1}]
    assert review['released_evaluation_record']['human_mechanism_expectation']['label_scope_note']
    pair, = review['paired_observations']
    assert set(pair['modes']) == {'direct', 'blind', 'leaky'}
    assert pair['modes']['blind']['observation']['answer'] == 'an alias for X'
    assert review['human_semantic_review'] == {'state': 'pending', 'automatic_correctness_judgment': False,
                                             'literal_answer_inequality_is_semantic_conflict': False}
    assert 'accuracy' not in result and 'auroc' not in result


def test_evaluator_is_frozen_on_disk_before_first_gold_hash_or_open(make_run, monkeypatch):
    args = make_run()
    original_hash, original_open = data.file_hash, Path.open
    seen = []
    def check():
        assert (args[2] / 'plan.json').is_file()
        assert data.read_json(args[2] / 'plan.json')['evaluation_material_accessed'] is False
        assert data.read_json(args[2] / 'protocol.json') == module.DEFAULT_PROTOCOL
        assert data.read_json(args[2] / 'evaluator_source_identity.json') == NEW_SOURCE
    def guarded_hash(path):
        if Path(path).name == 'evaluation_only.jsonl':
            check(); seen.append('hash')
        return original_hash(path)
    def guarded_open(path, *values, **kwargs):
        assert path.name != 'annotations.jsonl'
        if path.name == 'evaluation_only.jsonl':
            check(); seen.append('open')
        return original_open(path, *values, **kwargs)
    monkeypatch.setattr(data, 'file_hash', guarded_hash)
    monkeypatch.setattr(Path, 'open', guarded_open)
    module.evaluate(*args)
    assert 'hash' in seen and 'open' in seen


@pytest.mark.parametrize('problem', ['missing_complete', 'wrong_state', 'case_tamper', 'partial_cases',
                                    'plan_order', 'case_population', 'bad_coverage', 'missing_requests',
                                    'unreferenced_request', 'bad_parsed_observation', 'bad_source_identity',
                                    'bad_context_identity', 'bad_context_transform', 'v1_protocol'])
def test_invalid_observations_never_open_or_hash_gold(make_run, monkeypatch, problem):
    args = make_run(); observations = args[0]
    if problem == 'missing_complete':
        (observations / 'complete.json').unlink()
    elif problem == 'wrong_state':
        change_json(observations / 'complete.json', lambda v: v.update(state='partial'))
        seal(observations)
    elif problem == 'case_tamper':
        first_case(observations).write_text('{}', encoding='utf-8')
    elif problem == 'partial_cases':
        first_case(observations).unlink()
    elif problem == 'plan_order':
        change_json(observations / 'plan.json', lambda v: v['records'].reverse())
        seal(observations)
    elif problem == 'case_population':
        extra = observations / 'cases' / ('f' * 64); extra.mkdir()
        write(extra / 'case.json', {})
        seal(extra); seal(observations)
    elif problem == 'bad_coverage':
        change_json(observations / 'complete.json', lambda v: v['coverage'].update(valid_slots=1))
        seal(observations)
    elif problem == 'missing_requests':
        next((observations / 'requests').iterdir()).joinpath('response.json').unlink()
    elif problem == 'unreferenced_request':
        # A complete, properly named cache with no original case reference.
        original = next((observations / 'requests').iterdir())
        request = data.read_json(original / 'request.json')
        request['messages'][0]['content'] += ' altered'
        extra = observations / 'requests' / module._digest(request); extra.mkdir()
        write(extra / 'request.json', request)
        for name in ('response.json', 'backend_raw.json'):
            write(extra / name, data.read_json(original / name))
        seal(extra); seal(observations)
    elif problem == 'bad_parsed_observation':
        change_json(first_case(observations), lambda v: v['observations'][0]['observation'].update(answer='conflict'))
        seal(first_case(observations).parent); seal(observations)
    elif problem in {'bad_context_identity', 'bad_context_transform'}:
        key, value = ('pre_generation_context_sha256', '0' * 64) if problem == 'bad_context_identity' else ('context_transformation', {})
        change_json(first_case(observations), lambda v: v.update({key: value}))
        seal(first_case(observations).parent); seal(observations)
    elif problem == 'v1_protocol':
        change_json(observations / 'protocol.json', lambda v: v.update(schema='semantic-qualification-v1',
                                                                      contract_version='semantic-probe-v1'))
        seal(observations)
    else:
        write(observations / 'source_identity.json', {'source_sha256': {}})
        seal(observations)
    guard_gold(monkeypatch)
    with pytest.raises((ValueError, FileNotFoundError)):
        module.evaluate(*args)
    assert not args[2].exists()


def test_v2_context_roundtrip_preserves_original_question_and_prior_value(make_run):
    prompt = 'Original question: Does X belong to A?\n' + SOURCE + '\nUse the original scope.'
    args = make_run(prompt=prompt)
    result = module.evaluate(*args)
    review = data.read_json(args[2] / result['records'][0]['file'])
    expected = prompt.replace(SOURCE, observer.probe.SOURCE_PLACEHOLDER)
    assert review['pre_generation_context'] == expected
    assert 'Does X belong to A?' in expected
    assert review['original_inputs']['prompt'] == prompt
    requests = [data.read_json(path / 'request.json') for path in (args[0] / 'requests').iterdir()]
    assert len(requests) == 4
    assert all(json.loads(r['messages'][-1]['content'])['pre_generation_context'] == expected for r in requests)


@pytest.mark.parametrize('problem', ['changed_inputs', 'manifest_order', 'manifest_hash', 'constructed_schema'])
def test_panel_population_changes_fail_before_gold_access(make_run, monkeypatch, problem):
    args = make_run(); panel = args[1]
    if problem == 'changed_inputs':
        (panel / 'inputs.jsonl').write_text((panel / 'inputs.jsonl').read_text(encoding='utf-8').replace(RESPONSE, 'A has Y.'), encoding='utf-8')
    elif problem == 'manifest_order':
        change_json(panel / 'manifest.json', lambda v: v['input_rows'].reverse())
        seal(panel)
    elif problem == 'manifest_hash':
        change_json(panel / 'manifest.json', lambda v: v['input_rows'][0].update(source_sha256='0' * 64))
        seal(panel)
    else:
        change_json(panel / 'manifest.json', lambda v: v.update(schema='constructed-controls-v1'))
        seal(panel)
    guard_gold(monkeypatch)
    with pytest.raises(ValueError):
        module.evaluate(*args)
    assert not args[2].exists()


def test_unknown_invalid_and_output_limited_observations_stay_in_denominator(make_run):
    args = make_run(observations='mixed')
    result = module.evaluate(*args)
    modes = result['coverage']['slot_modes']
    assert all(m['whole_case_denominator'] == m['proposed_slot_denominator'] == 18 for m in modes.values())
    assert all(m['retained_observations'] == 18 for m in modes.values())
    assert modes['direct']['unknown'] == 18 and modes['direct']['parse_statuses'] == {'invalid': 18}
    assert modes['blind']['unknown'] == modes['blind']['output_limited'] == 18
    assert modes['leaky']['answerable'] == 18
    assert result['coverage']['automatic_accuracy_created'] is False


@pytest.mark.parametrize('proposal', ['empty', 'no_slots', 'invalid_limited'])
def test_empty_unobserved_cases_are_exported_with_explicit_count(make_run, proposal):
    args = make_run(proposal=proposal)
    result = module.evaluate(*args)
    assert len(result['records']) == 18
    assert result['coverage']['unobserved_case_count'] == 18
    assert result['coverage']['proposed_slots'] == 0
    assert result['coverage']['unproposed_facts_count'] is None
    assert all(m['unobserved_cases'] == 18 and m['proposed_slot_denominator'] == 0
               for m in result['coverage']['slot_modes'].values())
    review = data.read_json(args[2] / result['records'][0]['file'])
    assert review['unobserved_case_count'] == 1 and review['paired_observations'] == []
    if proposal == 'invalid_limited':
        assert review['proposal_output_limited'] is True
        assert review['proposal_envelope_error']['code'] == 'invalid_json'


def test_invalid_proposed_frames_are_preserved_without_filtering_case_population(make_run):
    args = make_run(proposal='mixed_invalid')
    result = module.evaluate(*args)
    assert result['coverage']['whole_case_denominator'] == result['coverage']['invalid_frames'] == 18
    assert result['coverage']['proposed_slots'] == 18
    review = data.read_json(args[2] / result['records'][0]['file'])
    assert len(review['invalid_frames']) == len(review['paired_observations']) == 1


def test_request_invalid_modes_are_retained_as_unknown(make_run):
    args = make_run(response='A [VALUE_REMOVED] has X.')
    result = module.evaluate(*args)
    assert result['coverage']['slot_modes']['direct']['answerable'] == 18
    for mode in ('blind', 'leaky'):
        assert result['coverage']['slot_modes'][mode]['parse_statuses'] == {'request_invalid': 18}
        assert result['coverage']['slot_modes'][mode]['unknown'] == 18


@pytest.mark.parametrize('target', ['panel', 'observation', 'ancestor'])
def test_output_containment_rejected_before_gold(make_run, monkeypatch, target):
    args = list(make_run())
    args[2] = {'panel': args[1] / 'evaluation', 'observation': args[0] / 'evaluation',
               'ancestor': args[1].parent}[target]
    guard_gold(monkeypatch)
    with pytest.raises(ValueError, match='disjoint'):
        module.evaluate(*args)


def test_unsafe_root_seal_never_hashes_escape_or_gold(make_run, monkeypatch):
    args = make_run()
    change_json(args[0] / 'seal.json', lambda v: v.update({'../evaluation_only.jsonl': '0' * 64}))
    guard_gold(monkeypatch)
    with pytest.raises(ValueError, match='member set'):
        module.evaluate(*args)


def test_redirected_observation_metadata_cannot_open_or_hash_gold(make_run, monkeypatch):
    args = make_run()
    path = args[0] / 'protocol.json'
    path.unlink()
    try:
        path.symlink_to(args[1] / 'evaluation_only.jsonl')
    except OSError:
        pytest.skip('Creating a synthetic symlink requires unavailable OS privileges')
    guard_gold(monkeypatch)
    with pytest.raises(ValueError, match='Redirected'):
        module.evaluate(*args)


def test_helper_change_fails_before_gold_without_requiring_observer_release_identity(make_run, monkeypatch):
    args = make_run()
    changed_source = deepcopy(NEW_SOURCE)
    changed_source['source_sha256']['experiments.research/semantic_probe.py'] = '0' * 64
    monkeypatch.setattr(module, 'research_identity', lambda: changed_source)
    guard_gold(monkeypatch)
    with pytest.raises(ValueError, match='restore a compatible CPU reader'):
        module.evaluate(*args)
    assert not args[2].exists()


def test_request_duration_sum_matches_observer_sorted_order_under_reverse_enumeration(make_run, monkeypatch):
    args = make_run(); root = args[0]
    durations = dict(zip(sorted(p.name for p in (root / 'requests').iterdir()), [.1, .2, .3, .4], strict=True))
    for key, seconds in durations.items():
        change_json(root / 'requests' / key / 'response.json', lambda v: v.update(seconds=seconds))
        seal(root / 'requests' / key)
    for path in (root / 'cases').glob('*/case.json'):
        value = data.read_json(path)
        for event in [value['proposal'], *value['observations']]:
            event['execution']['seconds'] = durations[event['request_key']]
        write(path, value)
        seal(path.parent)
    value = data.read_json(root / 'complete.json')
    value['costs']['seconds_generation'] = sum(durations[key] for key in sorted(durations))
    for record in value['records']:
        record['sha256'] = data.file_hash(root / record['file'])
    write(root / 'complete.json', value); seal(root)
    original = Path.iterdir
    def reversed_requests(path):
        children = list(original(path))
        return iter(reversed(children)) if path == root / 'requests' else iter(children)
    monkeypatch.setattr(Path, 'iterdir', reversed_requests)
    assert module.evaluate(*args)['coverage']['whole_case_denominator'] == 18


def test_resealed_observations_during_preflight_are_rejected_before_gold(make_run, monkeypatch):
    args = make_run()
    original = module._case
    def change_after_last_case(sample, *values):
        result = original(sample, *values)
        if sample.id == 'original-17':
            change_json(args[0] / 'plan.json', lambda v: v.update(changed_during_preflight=True))
            seal(args[0])
        return result
    monkeypatch.setattr(module, '_case', change_after_last_case)
    guard_gold(monkeypatch)
    with pytest.raises(ValueError, match='seal changed'):
        module.evaluate(*args)
    assert not args[2].exists()


def test_resealed_observations_during_export_cannot_claim_completed_review(make_run, monkeypatch):
    args = make_run()
    original = module._summary
    def change_during_export(*values):
        result = original(*values)
        change_json(args[0] / 'plan.json', lambda v: v.update(changed_during_export=True))
        seal(args[0])
        return result
    monkeypatch.setattr(module, '_summary', change_during_export)
    with pytest.raises(ValueError, match='seal changed'):
        module.evaluate(*args)
    assert (args[2] / 'plan.json').is_file()
    assert not (args[2] / 'complete.json').exists()
    assert not (args[2] / 'seal.json').exists()


def test_existing_output_is_never_overwritten(make_run, monkeypatch):
    args = make_run()
    module.evaluate(*args)
    before = data.file_hash(args[2] / 'seal.json')
    guard_gold(monkeypatch)
    with pytest.raises(FileExistsError, match='not be overwritten'):
        module.evaluate(*args)
    assert data.file_hash(args[2] / 'seal.json') == before


@pytest.mark.parametrize('problem', ['changed_gold', 'wrong_source', 'wrong_response', 'missing_id', 'bad_span'])
def test_released_material_must_match_every_original_case(make_run, problem):
    args = make_run(); panel = args[1]
    rows = [json.loads(line) for line in (panel / 'evaluation_only.jsonl').read_text(encoding='utf-8').splitlines()]
    if problem == 'changed_gold':
        rows[0]['human_mechanism_expectation']['expectation'] = 'tampered'
    elif problem == 'wrong_source':
        rows[0]['source_id'] = 'wrong'
    elif problem == 'wrong_response':
        rows[0]['released_annotation_record']['response_sha256'] = '0' * 64
    elif problem == 'missing_id':
        rows.pop()
    else:
        rows[0]['released_annotation_record']['spans'][0]['end'] = 9999
    (panel / 'evaluation_only.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows), encoding='utf-8')
    if problem != 'changed_gold':
        change_json(panel / 'manifest.json', lambda v: v.update(evaluation_only_sha256=data.file_hash(panel / 'evaluation_only.jsonl')))
        seal(panel)
    with pytest.raises(ValueError):
        module.evaluate(*args)
    assert (args[2] / 'plan.json').is_file()
    assert not (args[2] / 'complete.json').exists()


def test_protocol_prohibitions_are_frozen_and_cannot_enable_scores(make_run, monkeypatch):
    args = make_run()
    change_json(args[3], lambda v: v.update(scalar_scores_created=True))
    guard_gold(monkeypatch)
    with pytest.raises(ValueError, match='prohibitions'):
        module.evaluate(*args)


def test_cli_routes_to_cpu_evaluation_stage_without_model_backend(tmp_path, monkeypatch):
    called = []
    monkeypatch.setattr(module, 'evaluate', lambda *args: called.append(args))
    monkeypatch.setattr(observer, 'HFGenerator', lambda *args, **kwargs: pytest.fail('No model backend in evaluation'))
    argv = ['r4-experiment', 'transfer', 'evaluate-semantic', '--observations', str(tmp_path / 'obs'),
            '--panel', str(tmp_path / 'panel'), '--output', str(tmp_path / 'eval'),
            '--protocol', str(tmp_path / 'protocol.json')]
    monkeypatch.setattr('sys.argv', argv)
    main()
    assert called == [(Path(argv[4]), Path(argv[6]), Path(argv[8]), Path(argv[10]))]

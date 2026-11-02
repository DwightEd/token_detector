"""Qualification isolation, exact request reuse, and sealed-resume controls."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import data, semantic_qualification as module
from experiments.run import main


IDENTITY = {'model_id': 'fake-observation-instrument',
            'model_files_sha256': {'weights.bin': 'a' * 64}}
RESPONSE = 'A has X.'
SOURCE = 'Evidence X.'
PROMPT = 'Question: Is X assigned to A?\nRead ' + SOURCE + '\nAnswer completely.'


def proposal():
    return {'claims': [{'start': 0, 'end': len(RESPONSE), 'text': RESPONSE,
                        'slots': [{'role': 'actor', 'start': 6, 'end': 7, 'text': 'X'}]}]}


class FakeGenerator:
    model_identity = IDENTITY

    def __init__(self, *, proposal_value=None, observation_value=None, token_count=20):
        self.calls, self.counts = [], []
        self.proposal_value = proposal_value
        self.observation_value = observation_value
        self.token_count = token_count

    def count_input_tokens(self, messages):
        self.counts.append(deepcopy(messages))
        return self.token_count(messages) if callable(self.token_count) else self.token_count

    def __call__(self, messages, *, max_new_tokens):
        self.calls.append((deepcopy(messages), max_new_tokens))
        payload = json.loads(messages[-1]['content'])
        if 'response' in payload:
            result = self.proposal_value if self.proposal_value is not None else proposal()
        else:
            result = self.observation_value
            if result is None:
                result = {'status': 'answerable', 'answer': 'supported' if 'claim' in payload else 'X',
                          'quotes': [{'text': SOURCE}], 'reason': ''}
        text = result if isinstance(result, str) else json.dumps(result)
        count = self.token_count(messages) if callable(self.token_count) else self.token_count
        return {'text': text, 'raw_text': text + '<eos>', 'input_tokens': count,
                'output_tokens': 12, 'finish_reason': 'eos'}


@pytest.fixture
def panel(tmp_path, monkeypatch):
    monkeypatch.setattr(module, 'research_identity', lambda: {'source': 'fixed'})
    monkeypatch.setattr(module, 'completed_identity', lambda old: old)
    source = tmp_path / 'panel'; source.mkdir()
    rows = [{'id': 'META_ONLY_' + str(i), 'source_id': 'S' + str(i), 'split': 'test',
             'task': 'QA', 'prompt': PROMPT, 'source': SOURCE, 'response': RESPONSE,
             'generator': 'original-generator'} for i in range(18)]
    inputs = source / 'inputs.jsonl'
    inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    for name in ['evaluation_only.jsonl', 'annotations.jsonl', 'manifest.json']:
        (source / name).write_text('GOLD MUST NOT BE OPENED', encoding='utf-8')
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(module.DEFAULT_PROTOCOL), encoding='utf-8')
    return inputs, protocol, tmp_path / 'outputs' / 'qualification'


def run(panel, generator=None, identity=IDENTITY):
    return module.qualify(*panel, generator or FakeGenerator(), identity)


def first_case(output):
    return data.read_json(output / 'cases' / data.text_hash('META_ONLY_0') / 'case.json')


def test_full_panel_never_reads_gold_or_puts_case_ids_in_messages(panel, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name not in {'evaluation_only.jsonl', 'annotations.jsonl', 'manifest.json'}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    generator = FakeGenerator()
    complete = run(panel, generator)
    context = module.probe.pre_generation_context(PROMPT, SOURCE)
    assert [r['id'] for r in complete['records']] == ['META_ONLY_' + str(i) for i in range(18)]
    assert complete['state'] == 'observations_sealed_not_evaluated'
    assert complete['evaluation_material_accessed'] is False
    assert complete['scalar_scores_created'] is False
    assert len(generator.calls) == 4
    assert complete['costs']['logical_requests'] == 72
    assert complete['costs']['exact_request_reuses'] == 68
    assert complete['costs']['actual_model_calls'] == 4
    assert complete['costs']['input_tokens_generated'] == 80
    assert complete['coverage']['valid_frames'] == complete['coverage']['valid_slots'] == 18
    assert complete['coverage']['observation_statuses'] == {'valid': 54}
    for messages, cap in generator.calls:
        assert 'META_ONLY' not in json.dumps(messages)
        payload = json.loads(messages[-1]['content'])
        if 'response' in payload:
            assert payload == {'response': RESPONSE, 'pre_generation_context': context} and cap == 1536
        else:
            assert cap == 256 and payload['source'] == SOURCE
            assert payload['pre_generation_context'] == context
            if 'claimed_value' not in payload and 'claim' not in payload:
                assert set(payload) == {'source', 'pre_generation_context', 'masked_claim', 'role'}
                assert payload['masked_claim'] == 'A has [VALUE_REMOVED].'
    case = first_case(panel[2])
    assert case['schema'] == 'semantic-qualification-case-v2'
    assert case['pre_generation_context_sha256'] == data.text_hash(context)
    assert case['context_transformation']['original_source_span'] == [PROMPT.index(SOURCE), PROMPT.index(SOURCE) + len(SOURCE)]
    assert case['coverage']['response_characters'] == len(RESPONSE)
    assert case['proposal']['execution']['raw_text'].endswith('<eos>')
    assert len(case['observations']) == 3
    data.verify_seal(panel[2], required={'plan.json', 'complete.json', 'source_identity.json'})


def test_completed_resume_never_calls_or_counts_again(panel):
    complete = run(panel)
    generator = FakeGenerator()
    generator.count_input_tokens = lambda _: pytest.fail('Completed request must not be counted again')
    assert run(panel, generator) == complete
    assert not generator.calls


def test_mixed_invalid_frame_is_retained_and_other_frame_observed(panel):
    payload = proposal()
    payload['claims'].append({'start': 0, 'end': 999, 'text': 'bad', 'slots': []})
    complete = run(panel, FakeGenerator(proposal_value=payload))
    case = first_case(panel[2])
    assert case['raw_claim_count'] == 2 and len(case['invalid_frames']) == 1
    assert case['invalid_frames'][0]['raw']['end'] == 999
    assert len(case['frames']) == 1 and len(case['observations']) == 3
    assert complete['coverage']['invalid_frames'] == 18


def test_invalid_json_and_length_finish_are_preserved(panel):
    generator = FakeGenerator(proposal_value='{"claims":[')
    original = generator.__call__
    class LengthGenerator(FakeGenerator):
        def __call__(self, messages, *, max_new_tokens):
            return {**original(messages, max_new_tokens=max_new_tokens), 'finish_reason': 'max_new_tokens'}
    complete = run(panel, LengthGenerator())
    case = first_case(panel[2])
    assert case['proposal_envelope_error']['code'] == 'invalid_json'
    assert case['proposal']['execution']['text'] == '{"claims":['
    assert case['proposal']['execution']['finish_reason'] == 'max_new_tokens'
    assert case['proposal_output_limited'] is True
    assert not case['observations'] and complete['coverage']['proposal_envelope_failures'] == 18


def test_valid_json_at_output_limit_is_still_coverage_limited(panel):
    class LimitedGenerator(FakeGenerator):
        def __call__(self, messages, *, max_new_tokens):
            result = super().__call__(messages, max_new_tokens=max_new_tokens)
            if 'response' in json.loads(messages[-1]['content']):
                result['finish_reason'] = 'max_new_tokens'
            return result
    complete = run(panel, LimitedGenerator())
    case = first_case(panel[2])
    assert case['frames'] and len(case['observations']) == 3
    assert case['proposal_output_limited'] is True
    assert 'proposal_output_token_limit' in case['coverage_limited_reasons']
    assert complete['coverage']['proposal_output_token_limits'] == 18


def test_invalid_observation_becomes_explicit_unknown_with_raw_output(panel):
    run(panel, FakeGenerator(observation_value='bad JSON'))
    case = first_case(panel[2])
    assert all(o['parse_status'] == 'invalid' and o['execution']['text'] == 'bad JSON'
               and o['observation']['status'] == 'unknown' for o in case['observations'])


def test_empty_slots_remain_in_coverage_without_inventing_queries(panel):
    payload = proposal(); payload['claims'][0]['slots'] = []
    complete = run(panel, FakeGenerator(proposal_value=payload))
    case = first_case(panel[2])
    assert case['coverage']['claims_without_slots'] == 1
    assert not case['observations'] and complete['costs']['actual_model_calls'] == 1


@pytest.mark.parametrize('proposal_too_long', [True, False])
def test_input_token_limit_rejects_without_truncating_or_generating(panel, proposal_too_long):
    def count(messages):
        is_proposal = 'response' in json.loads(messages[-1]['content'])
        return 4097 if is_proposal == proposal_too_long else 20
    generator = FakeGenerator(token_count=count)
    complete = run(panel, generator)
    case = first_case(panel[2])
    assert complete['costs']['actual_model_calls'] == (0 if proposal_too_long else 1)
    assert complete['costs']['input_rejections'] == (1 if proposal_too_long else 3)
    if proposal_too_long:
        assert not generator.calls and not case['frames']
    else:
        assert len(case['observations']) == 3
        assert all(o['parse_status'] == 'input_rejected' for o in case['observations'])


@pytest.mark.parametrize('changed', ['input', 'context', 'protocol', 'model', 'code'])
def test_resume_rejects_changed_dependencies_before_inference(panel, monkeypatch, changed):
    run(panel)
    identity = IDENTITY
    if changed == 'input':
        panel[0].write_text(panel[0].read_text(encoding='utf-8').replace('A has X.', 'A has Y.'), encoding='utf-8')
    elif changed == 'context':
        panel[0].write_text(panel[0].read_text(encoding='utf-8').replace('Is X assigned to A?', 'Is X assigned to B?'), encoding='utf-8')
    elif changed == 'protocol':
        value = dict(module.DEFAULT_PROTOCOL, notes='changed frozen protocol')
        panel[1].write_text(json.dumps(value), encoding='utf-8')
    elif changed == 'model':
        identity = {**IDENTITY, 'model_files_sha256': {'weights.bin': 'b' * 64}}
    else:
        monkeypatch.setattr(module, 'research_identity', lambda: {'source': 'changed'})
    generator = FakeGenerator(); generator.model_identity = identity
    with pytest.raises(ValueError, match='Resume plan differs'):
        run(panel, generator, identity)
    assert not generator.calls


def test_unfinished_case_and_existing_claim_require_recovery(panel):
    class FailedGenerator(FakeGenerator):
        def __call__(self, messages, *, max_new_tokens):
            raise RuntimeError('simulated failure')
    with pytest.raises(RuntimeError, match='simulated'):
        run(panel, FailedGenerator())
    with pytest.raises(RuntimeError, match='Unfinished case'):
        run(panel)
    (panel[2] / '.qualification.lock').write_text('claim', encoding='utf-8')
    with pytest.raises(RuntimeError, match='liveness/recovery'):
        run(panel)


def test_bad_backend_metadata_is_saved_and_never_silently_reissued(panel):
    class BadBackend(FakeGenerator):
        def __call__(self, messages, *, max_new_tokens):
            return {**super().__call__(messages, max_new_tokens=max_new_tokens), 'input_tokens': 999}
    with pytest.raises(ValueError, match='invalid text/token-count'):
        run(panel, BadBackend())
    raw = list((panel[2] / 'requests').glob('*/backend_raw.json'))
    assert len(raw) == 1 and data.read_json(raw[0])['value']['input_tokens'] == 999
    with pytest.raises(RuntimeError, match='Unfinished case'):
        run(panel)


def test_sealed_case_tampering_fails_without_generation(panel):
    run(panel)
    path = panel[2] / 'cases' / data.text_hash('META_ONLY_0') / 'case.json'
    path.write_text('{}', encoding='utf-8')
    generator = FakeGenerator()
    with pytest.raises(ValueError, match='Sealed artifact changed'):
        run(panel, generator)
    assert not generator.calls


def test_protocol_limits_cannot_be_changed(panel):
    panel[1].write_text(json.dumps({**module.DEFAULT_PROTOCOL, 'max_input_tokens': 100}), encoding='utf-8')
    with pytest.raises(ValueError, match='fixed v2'):
        run(panel)
    assert not panel[2].exists()


def test_v1_observation_protocol_is_rejected_without_model_calls(panel):
    old = {**module.DEFAULT_PROTOCOL, 'schema': 'semantic-qualification-v1', 'contract_version': 'semantic-probe-v1'}
    old.pop('context_policy')
    panel[1].write_text(json.dumps(old), encoding='utf-8')
    generator = FakeGenerator()
    with pytest.raises(ValueError, match='Unsupported observation protocol'):
        run(panel, generator)
    assert not generator.calls and not panel[2].exists()


def test_reserved_context_marker_rejects_the_whole_input_contract_before_generation(panel, monkeypatch):
    rows = [json.loads(line) for line in panel[0].read_text(encoding='utf-8').splitlines()]
    rows[-1]['prompt'] += module.probe.SOURCE_PLACEHOLDER
    panel[0].write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name not in {'evaluation_only.jsonl', 'annotations.jsonl', 'manifest.json'}
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    generator = FakeGenerator()
    with pytest.raises(module.probe.ProbeValidationError, match='reserved source placeholder'):
        run(panel, generator)
    assert not generator.calls and not generator.counts and not panel[2].exists()


def test_request_cache_key_includes_input_admission_limit(tmp_path):
    requests = tmp_path / 'requests'; requests.mkdir()
    generator = FakeGenerator(token_count=4097)
    request = module.probe.build_proposer(RESPONSE)
    rejected_key, rejected, _ = module._request(generator, request, module.DEFAULT_PROTOCOL, IDENTITY, requests)
    accepted_key, accepted, _ = module._request(generator, request,
        {**module.DEFAULT_PROTOCOL, 'max_input_tokens': 4097}, IDENTITY, requests)
    assert rejected_key != accepted_key
    assert rejected['status'] == 'input_rejected' and accepted['status'] == 'generated'
    assert len(generator.calls) == 1


def test_annotation_input_alias_is_rejected_before_any_read(panel, monkeypatch):
    original = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name != 'evaluation_only.jsonl'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    with pytest.raises(ValueError, match='original inputs.jsonl'):
        module.qualify(panel[0].parent / 'evaluation_only.jsonl', panel[1], panel[2],
                       FakeGenerator(), IDENTITY)


def test_hf_is_lazy_and_denies_loading_without_admission_or_ram(monkeypatch, tmp_path):
    identity = {**IDENTITY, 'model_id': 'Qwen/Qwen2.5-3B-Instruct', 'runtime': module.HF_RUNTIME}
    backend = module.HFGenerator(tmp_path, identity)
    assert backend.tokenizer is None and backend.model is None
    with pytest.raises(RuntimeError, match='admission must be explicit'):
        backend.count_input_tokens([])
    backend = module.HFGenerator(tmp_path, identity, admit_resources=True)
    monkeypatch.setattr(module, '_available_ram', lambda: 1)
    with pytest.raises(RuntimeError, match='RAM below'):
        backend.count_input_tokens([])
    assert backend.tokenizer is None and backend.model is None


def test_hf_rejects_unfrozen_or_wrong_runtime_before_heavy_imports(monkeypatch, tmp_path):
    identity = {**IDENTITY, 'model_id': 'Qwen/Qwen2.5-3B-Instruct'}
    with pytest.raises(ValueError, match='exact CUDA'):
        module.HFGenerator(tmp_path, identity)
    identity['runtime'] = module.HF_RUNTIME
    backend = module.HFGenerator(tmp_path, identity, admit_resources=True)
    monkeypatch.setattr(module, '_available_ram', lambda: 4 * 1024**3)
    monkeypatch.setattr(module, 'version', lambda package: 'wrong' if package == 'bitsandbytes' else module.HF_RUNTIME[package])
    with pytest.raises(RuntimeError, match='pinned CUDA'):
        backend.count_input_tokens([])
    assert backend.tokenizer is None and backend.model is None


def test_hf_model_manifest_binds_every_loaded_file(tmp_path):
    values = {'config.json': '{}', 'tokenizer_config.json': '{}', 'tokenizer.json': '{}',
              'model.safetensors.index.json': json.dumps({'weight_map': {'layer': 'weights.safetensors'}}),
              'weights.safetensors': 'not-real-test-weights'}
    for name, content in values.items():
        (tmp_path / name).write_text(content, encoding='utf-8')
    identity = {'model_id': 'Qwen/Qwen2.5-3B-Instruct', 'runtime': module.HF_RUNTIME,
                'model_files_sha256': {name: data.file_hash(tmp_path / name) for name in values}}
    backend = module.HFGenerator(tmp_path, identity)
    backend.verify_frozen_identity()
    (tmp_path / 'generation_config.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='unbound files'):
        backend.verify_frozen_identity()


def test_hf_manifest_cannot_name_annotations_before_any_hash_or_read(tmp_path, monkeypatch):
    identity = {'model_id': 'Qwen/Qwen2.5-3B-Instruct', 'runtime': module.HF_RUNTIME,
                'model_files_sha256': {'config.json': 'a' * 64, 'annotations.jsonl': 'b' * 64}}
    backend = module.HFGenerator(tmp_path, identity)
    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: pytest.fail('Manifest paths must be admitted before I/O'))
    with pytest.raises(ValueError, match='Unsafe model manifest path'):
        backend.verify_frozen_identity()


def test_cli_routes_to_shared_qualify_without_real_hf(panel, monkeypatch):
    identity_path = panel[0].parent.parent / 'model_identity.json'
    identity_path.write_text(json.dumps(IDENTITY), encoding='utf-8')
    generator = FakeGenerator()
    monkeypatch.setattr(module, 'HFGenerator', lambda *args, **kwargs: generator)
    argv = ['r4-experiment', 'transfer', 'semantic-qualify', '--inputs', str(panel[0]),
            '--protocol', str(panel[1]), '--model', 'unused-model-path',
            '--model-identity', str(identity_path), '--output', str(panel[2])]
    monkeypatch.setattr('sys.argv', argv)
    with pytest.raises(SystemExit):
        main()
    assert not generator.calls
    monkeypatch.setattr('sys.argv', argv + ['--admit-resources'])
    main()
    assert len(generator.calls) == 4

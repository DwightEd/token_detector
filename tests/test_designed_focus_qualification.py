"""H007-specific admission/freeze/reuse checks; fake backend, synthetic only."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import data, designed_focus_qualification as module
from experiments import verdict_qualification as shared
from experiments.datasets import read_samples

MODEL = dict(model_id='synthetic-designed-focus', model_files_sha256={'weights': 'a' * 64})
SOURCE = 'Frozen source.'


class Backend:
    model_identity = MODEL

    def __init__(self, check=None, rejected=False):
        self.check, self.rejected, self.calls, self.preflights = check, rejected, [], []

    def count_input_tokens(self, messages):
        return 4097 if self.rejected else 20

    def validate_labels(self, messages, labels):
        self.preflights.append(deepcopy(messages))
        return dict(input_tokens=self.count_input_tokens(messages),
            label_token_ids=dict(zip(labels, [10, 11, 12])), label_decodes=dict(zip(labels, labels)),
            prefix_alignment_verified=True)

    def score_labels(self, messages, labels):
        if self.check:
            self.check(messages)
        self.calls.append(deepcopy(messages))
        return dict(self.validate_labels(messages, labels), label_logits=dict(zip(labels, [2., 1., 0.])),
            full_vocab_logsumexp=4., actual_forward_passes=1, output_tokens=0, seconds=.125)


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    def write(path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    monkeypatch.setattr(data, 'write_json', write)
    for mod in (module, shared):
        monkeypatch.setattr(mod, 'research_identity', lambda: {'source_sha256': {'fixed': 'a' * 64}})
        monkeypatch.setattr(mod, 'completed_identity', lambda source: source)

    def build(count=48, response='Target 3.14. Distractor one. Distractor two.'):
        panel = tmp_path / 'panel'
        panel.mkdir()
        rows = [dict(id=f'original-{i}', source_id=f'source-{i}', split='test', task='QA',
            prompt='Question: ' + SOURCE + ' Respond.', source=SOURCE, response=response,
            generator='synthetic') for i in range(count)]
        inputs = panel / 'inputs.jsonl'
        inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
        for name in ('manifest.json', 'truth.jsonl', 'evaluation_only.jsonl'):
            (panel / name).write_text('MUST NOT READ', encoding='utf-8')
        protocol = tmp_path / 'protocol.json'
        write(protocol, dict(module.DEFAULT_PROTOCOL, original_inputs_sha256=data.file_hash(inputs)))
        return inputs, protocol, tmp_path / 'outputs' / 'designed', rows
    return build


def test_all48_both_scopes_and_six_assignments_frozen_before_first_forward(prepared, monkeypatch):
    inputs, protocol, output, rows = prepared()
    original_open = Path.open
    def guard(path, *args, **kwargs):
        assert path.name not in {'manifest.json', 'truth.jsonl', 'evaluation_only.jsonl'}
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guard)
    def frozen(messages):
        plan = data.read_json(output / 'plan.json')
        assert len(plan['records']) == 48 and len(plan['flat_records']) == 96
        assert plan['logical_requests'] == plan['max_forward_passes'] == 576
        assert {row['mode'] for row in plan['flat_records']} == {'whole', 'target_first_unit'}
        assert all(len(row['units']) == 1 and row['units'][0]['index'] == 0 for row in plan['records'])
        data.verify_seal(output / 'prepared_flat', expected={'inputs.jsonl'})
        child = data.read_json(output / 'observations' / 'plan.json')
        assert len(child['records']) == 96 and all(len(row['requests']) == 6 for row in child['records'])
        payload = json.loads(messages[-1]['content'])
        assert payload['source'] == SOURCE
        assert payload['response_before_focus'] + payload['focus_text'] + payload['response_after_focus'] == rows[0]['response']
        assert 'original-' not in json.dumps(messages)
    backend = Backend(frozen)
    result = module.qualify(inputs, protocol, output, backend, MODEL)
    assert result['coverage'] == dict(original_cases=48, whole_focuses=48, unit_focuses=48,
        flat_cases=96, scored_focuses=96, unknown_focuses=0, unknown_units=0, empty_responses=0)
    assert len(result['records']) == 48 and result['costs']['logical_requests'] == 576
    assert len(backend.calls) == 12
    for row in result['records']:
        assert row['units'][0]['text'] == 'Target 3.14. '
    assert all(sample.source == SOURCE and sample.response == rows[0]['response']
               for sample in read_samples(output / 'prepared_flat' / 'inputs.jsonl'))
    data.verify_seal(output)


def test_zero_distractor_exact_requests_reuse_but_all_instances_remain(prepared):
    inputs, protocol, output, _ = prepared(response='Target 3.14.')
    backend = Backend()
    result = module.qualify(inputs, protocol, output, backend, MODEL)
    assert len(backend.calls) == result['costs']['actual_forward_passes'] == 6
    assert result['costs']['exact_request_reuses'] == 570
    assert len(result['records']) == 48 and len({row['id'] for row in result['records']}) == 48
    for row in result['records']:
        assert row['whole']['flat_id'] != row['units'][0]['flat_id']
        assert row['whole']['risk'] == row['units'][0]['risk']


def test_length_unknowns_keep_all48_both_scopes_at_risk_one(prepared):
    inputs, protocol, output, _ = prepared()
    backend = Backend(rejected=True)
    result = module.qualify(inputs, protocol, output, backend, MODEL)
    assert not backend.calls and result['coverage']['unknown_focuses'] == 96
    assert result['coverage']['unknown_units'] == 48
    assert all(row['whole']['risk'] == 1 and row['units'][0]['risk'] == 1 for row in result['records'])


@pytest.mark.parametrize('failure', ['missing_original', 'hash', 'mode', 'empty_response'])
def test_invalid_designed_population_rejects_before_backend_or_output(prepared, failure):
    inputs, protocol, output, _ = prepared(count=47 if failure == 'missing_original' else 48,
                                           response='' if failure == 'empty_response' else 'Target. Tail.')
    if failure in {'hash', 'mode'}:
        value = data.read_json(protocol)
        value['original_inputs_sha256'] = 'f' * 64 if failure == 'hash' else value['original_inputs_sha256']
        if failure == 'mode':
            value['modes'] = ['whole', 'unit']
        data.write_json(protocol, value)
    backend = Backend()
    with pytest.raises(ValueError):
        module.qualify(inputs, protocol, output, backend, MODEL)
    assert backend.calls == backend.preflights == [] and not output.exists()

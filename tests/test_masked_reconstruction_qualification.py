"""CPU-only integration checks of the frozen masked bridge and shared collector."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import data, masked_reconstruction_qualification as module
from tests.test_reconstruction_qualification import Backend, MODEL, fake48


@pytest.fixture
def paths(tmp_path, monkeypatch):
    def write(path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    monkeypatch.setattr(data, 'write_json', write)
    monkeypatch.setattr(module.collector, 'research_identity', lambda: {'source_sha256': {'fixed': 'b' * 64}})
    monkeypatch.setattr(module.collector, 'completed_identity', lambda source: source)
    inference = tmp_path / 'inference'
    inference.mkdir()
    inputs = inference / 'inputs.jsonl'
    inputs.write_text(''.join(json.dumps(row) + '\n' for row in fake48()), encoding='utf-8')
    protocol = tmp_path / 'protocol.json'
    write(protocol, dict(module.DEFAULT_PROTOCOL, original_inputs_sha256=data.file_hash(inputs)))
    return inputs, protocol, tmp_path / 'outputs' / 'masked'


def test_all96_observations_frozen_and_entire_response_masked_before_first_forward(paths, monkeypatch):
    inputs, protocol, output = paths
    original_open = Path.open
    def guarded(path, *args, **kwargs):
        assert path.name not in {'construction_truth.jsonl', 'evaluation_only.jsonl', 'manifest.json'}
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    checked = set()
    def first(messages, candidate):
        assert backend.preflights == 96
        plan = data.read_json(output / 'plan.json')
        assert len(plan['executions']) == 96 and len(plan['prepared']['originals']) == 48
        assert sum(name.endswith('request.json') for name in data.read_json(output / 'freeze.json')) == 96
        assert plan['response_used_in_model_requests'] is True
        assert plan['response_person_values_exposed_to_qa'] is False
        body = json.loads(messages[-1]['content'])
        people = module.collector.kernel.parse_source(body['source'])['people']
        assert candidate in people and all(person not in body['question'] for person in people)
        assert '[TARGET]' in body['question'] and '[OTHER]' in body['question']
        checked.add(json.dumps([messages, candidate], sort_keys=True))
    backend = Backend(first)
    result = module.qualify(inputs, protocol, output, backend, MODEL)
    assert backend.calls == len(checked) == 96
    assert result['coverage'] == dict(original_cases=48, unique_sources=8, unique_queries=48,
        unique_completions=96, response_slots=96, unknown_cases=0, unknown_slots=0,
        logical_completion_requests=192)
    assert result['evaluation_material_accessed'] is False
    assert result['masked_response_used_in_qa'] is True
    assert result['response_values_exposed_to_qa'] is False
    module.collector._inventory(output, has_complete=True, has_seal=True, expected_requests=96)
    data.verify_seal(output)
    with pytest.raises(FileExistsError):
        module.qualify(inputs, protocol, output, backend, MODEL)
    assert backend.calls == 96


def test_unknown_compiler_case_forbids_all_model_calls_without_removing_original(paths):
    inputs, protocol, output = paths
    rows = [json.loads(line) for line in inputs.read_text(encoding='utf-8').splitlines()]
    rows[0]['response'] += ' An unadmitted unrelated extra sentence.'
    inputs.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    value = data.read_json(protocol)
    value['original_inputs_sha256'] = data.file_hash(inputs)
    data.write_json(protocol, value)
    backend = Backend()
    with pytest.raises(ValueError):
        module.qualify(inputs, protocol, output, backend, MODEL)
    assert backend.calls == backend.preflights == 0
    assert not output.exists()
    assert len(rows) == 48


def test_frozen_masking_protocol_cannot_change_to_local_suffix_or_name_selection(paths):
    inputs, protocol, output = paths
    value = deepcopy(data.read_json(protocol))
    value['response_suffix_preserved'] = False
    data.write_json(protocol, value)
    backend = Backend()
    with pytest.raises(ValueError, match='Exact fixed'):
        module.qualify(inputs, protocol, output, backend, MODEL)
    assert backend.calls == backend.preflights == 0
    assert not output.exists()

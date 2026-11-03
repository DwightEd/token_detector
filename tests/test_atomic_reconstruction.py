"""Synthetic affirmative-conjunct transformation and controlled-adapter checks."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from experiments import atomic_reconstruction_qualification as profile
from experiments import data
from experiments.datasets import Sample
from experiments.research import atomic_reconstruction as kernel
from experiments.research import masked_reconstruction as prior
from tests.test_reconstruction_qualification import fake48, Backend, MODEL


def test_all_original_slots_retained_and_Rdirection_burden_collapse_to16_queries32_completions():
    rows = fake48()
    before = deepcopy(rows)
    result = kernel.prepare_atomic_queries(rows)
    assert rows == before
    assert result['counts'] == dict(originals=48, sources=8, logical_queries=96,
        unique_queries=16, logical_requests=192, unique_requests=32, unknown_occurrences=0)
    old = prior.prepare_masked_queries(rows)
    for original in result['originals']:
        row = next(row for row in rows if row['id'] == original['id'])
        for slot in original['occurrences']:
            assert row['response'][slot['start']:slot['end']] == slot['text']
            query = result['queries'][slot['query_key']]
            donor = old['queries'][slot['donor_H011_query_key']]
            assert query['source'] == donor['source']
            assert query['candidates'] == donor['candidates']
            assert query['messages'][0] == donor['messages'][0]
            assert query['event'] == donor['event']
            assert query['question'] == prior.QUESTION_TEMPLATE.format(masked_response=query['masked_response'])
            assert query['dangling_other_instruction_retained'] is True
            assert query['masked_response'].count('[TARGET]') == 1 and '[OTHER]' not in query['masked_response']
            assert query['masked_response'].endswith('. ')
            assert query['question'] != donor['question']
            assert all(person not in query['question'] for person in query['candidates'])
            assert ', and ' not in query['masked_response']
            assert slot['atomic_terminal_added'] == ('. ' if slot['index'] == 0 else '')
            start, end = slot['atomic_original_span']
            actual = row['response'][start:end] + slot['atomic_terminal_added']
            assert actual.replace(slot['text'], '[TARGET]') == query['masked_response']


def test_no_event_is_looked_up_as_a_source_truth_answer_and_data_is_detached():
    rows = fake48()
    result = kernel.prepare_atomic_queries(rows)
    query = next(iter(result['queries'].values()))
    query['candidates'][0] = 'mutated outside'
    fresh = kernel.prepare_atomic_queries([SimpleNamespace(**row) for row in rows])
    assert fresh['counts']['unique_queries'] == 16
    assert all('mutated outside' not in q['candidates'] for q in fresh['queries'].values())
    assert {q['event'] for q in fresh['queries'].values()} == {
        slot['event'] for row in fresh['originals'] for slot in row['occurrences']}


def test_unsupported_original_is_retained_as_two_unknown_occurrences():
    rows = fake48()
    rows[0]['response'] += ' Unadmitted statement.'
    result = kernel.prepare_atomic_queries(rows)
    assert len(result['originals']) == 48
    assert result['originals'][0]['status'] == 'unknown'
    assert len(result['originals'][0]['occurrences']) == result['counts']['unknown_occurrences'] == 2
    with pytest.raises(ValueError, match='Complete one-conjunct population'):
        profile._prepare([Sample(**row) for row in rows])


def test_shared_collector_freezes_all32_atomic_events_and_original96_slot_mappings(tmp_path, monkeypatch):
    def write(path, value):
        Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    monkeypatch.setattr(data, 'write_json', write)
    monkeypatch.setattr(profile.collector, 'research_identity', lambda: {'source_sha256': {'fixed': 'a' * 64}})
    monkeypatch.setattr(profile.collector, 'completed_identity', lambda value: value)
    inference = tmp_path / 'inference'
    inference.mkdir()
    inputs = inference / 'inputs.jsonl'
    inputs.write_text(''.join(json.dumps(row) + '\n' for row in fake48()), encoding='utf-8')
    protocol = tmp_path / 'protocol.json'
    write(protocol, dict(profile.DEFAULT_PROTOCOL, original_inputs_sha256=data.file_hash(inputs)))
    output = tmp_path / 'outputs' / 'atomic'
    seen = []
    def frozen(messages, candidate):
        if seen:
            return
        assert backend.preflights == 32
        plan = data.read_json(output / 'plan.json')
        assert len(plan['prepared']['originals']) == 48
        assert len(plan['executions']) == 32 and len(plan['prepared']['queries']) == 16
        assert len([name for name in data.read_json(output / 'freeze.json') if name.endswith('request.json')]) == 32
        assert plan['primary_control'] == 'sealed_H011_burden_zero_only'
        seen.append(True)
    backend = Backend(frozen)
    result = profile.qualify(inputs, protocol, output, backend, MODEL)
    assert backend.calls == 32 and seen == [True]
    assert result['coverage']['original_cases'] == 48 and result['coverage']['response_slots'] == 96
    assert result['coverage']['logical_completion_requests'] == 192
    assert result['coverage']['unique_queries'] == 16
    assert result['suffix_invariance_is_by_construction'] is True
    assert result['method_promoted'] is False and result['natural_task_performance_claim'] is False
    data.verify_seal(output)

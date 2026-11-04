"""Synthetic source/quote/coverage counterexamples; no tokenizer or model."""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import os
import subprocess
import sys

import pytest

from experiments.research import source_first_inventory as module


@pytest.fixture
def prepared():
    prefix = 'Original task and question: how to cook?\n'
    bodies = [
        '1 Grill brats for 5 to 10 minutes, turning once. A banana banana.',
        '2 Simmer pot for 15 to 20 minutes. Simmer pot for 15 to 20 minutes. URL https://x.test/2026/2.',
        '3 Brown onions in butter for 1-2 minutes. Do not store for more than three days.',
    ]
    blocks = [f'passage {i}:{body}\n\n' for i, body in enumerate(bodies, 1)]
    suffix = 'In case the passages do not contain needed information, reply unknown.\noutput:'
    prompt = prefix+''.join(blocks)+suffix
    boundaries, position = [], len(prefix)
    for block in blocks:
        boundaries.append([position, position+len(block)]); position += len(block)
    source_records = [dict(source_id=s, prompt=prompt) for s in ('14304', '14315', '14325', '14375')]
    first, second = prompt.index('5 to 10'), prompt.index('1-2')
    protocol = dict(schema='h020-source-first-record-proposal-v1', input_whitelist=['source_id', 'prompt'],
        requests=[dict(source_id=s, world=w) for s, w in module.REQUESTS],
        control_source_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        source14375=dict(characters=len(prompt), header=[0, len(prefix)], passage_blocks=boundaries,
            footer=[position, len(prompt)], reorder_original_passage_ids=[3, 1, 2],
            swap_literals=[dict(span=[first, first+7], old='5 to 10', new='1-2'),
                           dict(span=[second, second+3], old='1-2', new='5 to 10')]))
    return module.prepare_inventory(source_records, protocol), source_records, protocol


def coverage(case, mapped=()):
    return [dict(passage=p, unknown=[['unresolved synthetic obligations', [a['id'] for a in case['anchors']
        if a['passage'] == p and a['id'] not in mapped]]], nonrelational=[], scope_unknown=[]) for p in (1, 2, 3)]


def record(case, identity='r1'):
    quote = 'Grill brats for 5 to 10 minutes, turning once.'
    passage = next(p for p in case['passages'] if p['id'] == 1)
    start = passage['text'].index(quote)
    keys = [a['id'] for a in case['anchors'] if a['passage'] == 1 and start <= a['span'][0]
            and a['span'][1] <= start+len(quote)]
    return dict(id=identity, status='complete', speech='instruction', anchors=keys,
        cite=[[1, quote, 0]], pred=['Grill', 0],
        args=[['object', 'brats', 0], ['value', '5 to 10', 0]],
        ops=[['unit', 'minutes', 0, 'record'], ['method', 'turning once', 0, 'record']], links=[], unknown=[])


def validate(case, records, cov=None, finish='eos'):
    mapped = [key for r in records if isinstance(r, dict) for key in r.get('anchors', [])]
    return module.validate_proposal(json.dumps(dict(records=records, coverage=coverage(case, mapped) if cov is None else cov)),
                                    case, finish)


def test_six_worlds_lossless_controls_and_wire_has_no_metadata(prepared):
    value = prepared[0]
    assert value['request_count'] == 6 and not value['model_calls'] and not value['token_counts_measured']
    original, reordered, swapped = value['cases'][3:]
    assert [p['id'] for p in reordered['passages']] == [3, 1, 2]
    assert {p['id']: p['text'] for p in original['passages']} == {p['id']: p['text'] for p in reordered['passages']}
    assert [(a['id'], a['literal'], a['span']) for a in original['anchors']] == [
        (a['id'], a['literal'], a['span']) for a in reordered['anchors']]
    for case in value['cases']:
        assert list(case['model_data']) == ['task_prefix', 'task_suffix', 'passages', 'anchors']
        assert all(len(row) == 4 for row in case['model_data']['anchors'])
        assert all(set(p) == {'id', 'text'} for p in case['model_data']['passages'])
        for p in case['passages']:
            assert case['prompt'][slice(*p['prompt_span'])] == p['text'] and not p['text'].endswith('\n\n')
            assert p['separator'] == '\n\n'
    for case in (reordered, swapped):
        assert Counter(case['prompt'].encode()) == Counter(original['prompt'].encode())
        for part in case['transform']['segments']:
            if part['kind'] != 'replacement':
                assert case['prompt'][slice(*part['world_span'])] == original['prompt'][slice(*part['original_span'])]
    assert len(swapped['transform']['edits']) == 2
    for edit in swapped['transform']['edits']:
        assert original['prompt'][slice(*edit['original_span'])] == edit['old']
        assert swapped['prompt'][slice(*edit['world_span'])] == edit['new']
    assert 'turning once' in swapped['prompt'] and 'Brown onions in butter' in swapped['prompt']


def test_numeric_steps_url_words_and_cues_are_not_filtered(prepared):
    case = prepared[0]['cases'][3]
    atoms = {(a['kind'], a['literal']) for a in case['anchors']}
    assert {('number', '1'), ('number', '2026'), ('number', '2'), ('number_word', 'once'),
            ('number_word', 'three'), ('unit', 'days'), ('scope_cue', 'not'), ('scope_cue', 'more')} <= atoms
    for anchor in case['anchors']:
        passage = next(p for p in case['passages'] if p['id'] == anchor['passage'])
        assert passage['text'][slice(*anchor['span'])] == anchor['literal']
        assert case['prompt'][slice(*anchor['prompt_span'])] == anchor['literal']


def test_source_only_boundary_and_source_drift_refused(prepared):
    _, sources, protocol = prepared
    with pytest.raises(ValueError, match='R is forbidden'):
        module.prepare_inventory([{**r, 'response': 'secret existing R'} for r in sources], protocol)
    changed = deepcopy(sources); changed[-1]['prompt'] += ' '
    with pytest.raises(ValueError, match='Control source/span'):
        module.prepare_inventory(changed, protocol)
    changed_protocol = deepcopy(protocol); changed_protocol['source14375']['swap_literals'][0]['new'] = '9'
    with pytest.raises(ValueError, match='literal multiset'):
        module.prepare_inventory(sources, changed_protocol)


def test_reused_anchor_and_same_citation_are_legal_without_semantic_promotion(prepared):
    case = prepared[0]['cases'][3]
    r1, r2 = record(case), record(case, 'r2')
    result = validate(case, [r1, r2])
    assert result['mechanical_gate_passed'] and result['mechanically_complete_record_ids'] == ['r1', 'r2']
    assert result['structural_citation_reuse'][0]['uses'] == [[0, 0], [1, 0]]
    assert all(a['raw_record_indices'] == a['valid_record_indices'] == [0, 1]
               for a in result['anchor_accounting'] if a['category'] == 'mapped')
    assert not result['semantic_licensed'] and result['independent_source_review_required']


@pytest.mark.parametrize('mutation', [
    lambda r: r['cite'][0].__setitem__(2, 9),
    lambda r: r['cite'][0].__setitem__(1, 'Grill  brats'),
    lambda r: r['cite'][0].__setitem__(0, True),
    lambda r: r['args'][0].__setitem__(2, True),
    lambda r: r.__setitem__('speech', []),
    lambda r: r.__setitem__('links', [['before', 'missing', 0]]),
])
def test_bad_record_is_retained_and_rawmapped_is_not_validmapped(prepared, mutation):
    case = prepared[0]['cases'][3]; proposed = record(case); mutation(proposed)
    result = validate(case, [proposed])
    assert result['envelope_error'] is None and result['records'][0]['raw'] == proposed
    assert result['raw_model_mapped'] and not result['valid_mapped']
    assert not result['mechanical_gate_passed'] and result['any_raw_complete_invalid']
    assert all(a['category'] == 'invalid_mapped' for a in result['anchor_accounting'] if a['id'] in proposed['anchors'])


def test_duplicate_record_ids_are_all_rejected_not_first_selected(prepared):
    case = prepared[0]['cases'][3]; proposed = record(case)
    result = validate(case, [proposed, deepcopy(proposed)])
    assert not result['valid_mapped'] and not result['mechanically_complete_record_ids']
    assert all(any(i['code'] == 'duplicate_record_id' for i in r['issues']) for r in result['records'])


def test_overlapping_quote_ordinal_and_all_role_matches_are_preserved(prepared):
    case = prepared[0]['cases'][3]
    partial = dict(id='r', status='partial', speech='unknown', anchors=[], cite=[[1, 'ana', 1]],
        pred=None, args=[], ops=[], links=[], unknown=['source fragment and speech unknown'])
    result = validate(case, [partial])
    cite = result['records'][0]['citations'][0]
    body = next(p['text'] for p in case['passages'] if p['id'] == 1)
    assert cite['body_span'] == [body.index('banana')+3, body.index('banana')+6]
    assert result['mechanical_gate_passed']
    partial.update(speech='asserted', cite=[[1, 'banana banana', 0]], pred=['banana', 0])
    result = validate(case, [partial])
    assert len(result['records'][0]['literal_boundaries'][0]['body_matches']) == 2
    assert any(u['reason'] == 'boundary_ambiguity' for u in result['records'][0]['unresolved'])
    assert not result['mechanically_complete_record_ids']
    partial.update(status='complete', unknown=[])
    assert not validate(case, [partial])['mechanical_gate_passed']


def test_explicit_null_does_not_bypass_cite_or_unknown_requirement(prepared):
    case = prepared[0]['cases'][3]; proposed = record(case)
    proposed.update(status='partial', unknown=['owner unresolved'], args=[['subject', None, 0]])
    assert validate(case, [proposed])['mechanical_gate_passed']
    proposed['args'][0][2] = 99
    assert not validate(case, [proposed])['mechanical_gate_passed']
    proposed['args'][0][2] = 0; proposed['unknown'] = []
    assert not validate(case, [proposed])['mechanical_gate_passed']


def test_coverage_conflicts_and_nonrelational_are_not_truth(prepared):
    case = prepared[0]['cases'][3]; proposed = record(case)
    cov = coverage(case)  # mapped anchors also explicitly unknown -> conflict
    result = validate(case, [proposed], cov)
    assert not result['mechanical_gate_passed'] and any(a['category'] == 'conflict' for a in result['anchor_accounting'])
    cov = coverage(case)
    key = cov[0]['unknown'][0][1].pop()
    cov[0]['nonrelational'] = [['proposal says markup; independent review still required', [key]]]
    result = validate(case, [], cov)
    assert result['mechanical_gate_passed'] and result['nonrelational'] == [key]
    assert not result['semantic_licensed']
    cov[0]['nonrelational'][0][1].append(key)
    assert not validate(case, [], cov)['mechanical_gate_passed']


@pytest.mark.parametrize('text', [
    '```json\n{"records":[],"coverage":[]}\n```',
    '{"records":[],"records":[],"coverage":[]}',
    '{"records":[NaN],"coverage":[]}',
    '{"records":[1e999],"coverage":[]}',
    '{"records":[],"coverage":[],"x":"\\ud800"}',
    '{"records":[],"coverage":[]',
])
def test_invalid_json_is_request_unknown_without_partial_salvage(prepared, text):
    result = module.validate_proposal(text, prepared[0]['cases'][3], 'eos')
    assert result['request_status'] == 'unknown' and result['records'] == []
    assert not result['mechanical_gate_passed'] and not result['semantic_licensed']
    assert all(a['category'] == 'request_unknown' and not a['model_explicit'] for a in result['anchor_accounting'])


def test_output_limit_does_not_salvage_even_complete_json(prepared):
    case = prepared[0]['cases'][3]
    result = validate(case, [record(case)], finish='max_new_tokens')
    assert result['envelope_error'] == 'output_token_limit' and not result['records']


def test_cold_import_is_standard_library_only():
    code = ('import sys; from experiments.research.source_first_inventory import prepare_inventory; '
            "assert not {'torch','transformers','numpy'} & set(sys.modules)")
    env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, timeout=30, env=env)

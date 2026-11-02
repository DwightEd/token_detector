"""Synthetic H011 information-flow/offset checks; no files, models or gold."""
from copy import deepcopy
from itertools import product
import json
from types import SimpleNamespace

import pytest

from experiments.research import masked_reconstruction as kernel
from experiments.research import source_reconstruction as source_kernel


def source(family='winner', vocabulary=0, direction=0, people=None):
    people = people or (('Ava Lorn', 'Bex Pell') if vocabulary == 0 else ('Cara Wynn', 'Dax Ford'))
    prefix = 'Amber' if vocabulary == 0 else 'Silver'
    suffix = 'Cobalt' if vocabulary == 0 else 'Violet'
    events = (f'{prefix} Cup', f'{suffix} Cup') if family == 'winner' else (f'{prefix} Accord', f'{suffix} Accord')
    assigned = people if direction == 0 else people[::-1]
    if family == 'winner':
        facts = [line for event, person in zip(events, assigned) for line in (
            f'The {event} final was a match between {people[0]} and {people[1]}.',
            f'{person} was the sole winner of the {event} final.')]
    else:
        facts = [f'{person} was the sole signer of the {event}.' for person, event in zip(assigned, events)]
    return ' '.join([*facts, *source_kernel.FILLERS]), people, events


def response(family, people, events, direction=0, burden=0):
    a, b = people if direction == 0 else people[::-1]
    x, y = events
    target = (f'{a} won the {x} final, and {b} won the {y} final. ' if family == 'winner' else
              f'{a} was the sole signer of the {x}, and {b} was the sole signer of the {y}. ')
    return target + ' '.join(source_kernel.FILLERS[:burden])


def sample(family='winner', vocabulary=0, s=0, r=0, burden=0, identifier='case'):
    passage, people, events = source(family, vocabulary, s)
    return dict(id=identifier, source=passage, response=response(family, people, events, r, burden))


def test_complete48_grid_retains96_slots48_queries96_unique_and192_logical_requests():
    rows = [sample(f, v, s, r, b, f'{f}:{v}:{s}:{r}:{b}')
            for f, v, s, r, b in product(('winner', 'sole_signer'), (0, 1), (0, 1), (0, 1), (0, 4, 12))]
    frozen = deepcopy(rows)
    result = kernel.prepare_masked_queries(rows)
    assert rows == frozen
    assert result['counts'] == dict(originals=48, sources=8, logical_queries=96, unique_queries=48,
                                   logical_requests=192, unique_requests=96, unknown_occurrences=0)
    assert len(result['originals']) == 48
    for row, original in zip(rows, result['originals']):
        assert original['id'] == row['id'] and original['status'] == 'ok'
        assert len(original['occurrences']) == 2
        for occurrence in original['occurrences']:
            assert row['response'][occurrence['start']:occurrence['end']] == occurrence['text']
            query = result['queries'][occurrence['query_key']]
            assert query['candidates'][occurrence['candidate_index']] == occurrence['text']
            assert query['event'] == occurrence['event']
            assert len(occurrence['observation_keys']) == 2
            assert all(key in result['requests'] for key in occurrence['observation_keys'])


@pytest.mark.parametrize('family', ['winner', 'sole_signer'])
def test_actual_response_value_swap_exactly_collapses_every_query_and_message(family):
    for s, burden in product((0, 1), (0, 4, 12)):
        a = kernel.prepare_masked_queries([sample(family, s=s, r=0, burden=burden)])
        b = kernel.prepare_masked_queries([sample(family, s=s, r=1, burden=burden)])
        assert a['queries'] == b['queries'] and a['requests'] == b['requests']
        assert a['originals'][0]['occurrences'][0]['candidate_index'] != b['originals'][0]['occurrences'][0]['candidate_index']


def test_question_depends_on_actual_event_wording_slot_and_full_burden():
    original = sample()
    empty = kernel.prepare_masked_queries([original])
    burden = kernel.prepare_masked_queries([sample(burden=4)])
    swapped_events = dict(original, response=original['response'].replace('Amber Cup', 'TEMP').replace('Cobalt Cup', 'Amber Cup').replace('TEMP', 'Cobalt Cup'))
    event = kernel.prepare_masked_queries([swapped_events])
    first, second = empty['originals'][0]['occurrences']
    assert first['query_key'] != second['query_key']
    assert set(empty['queries']).isdisjoint(burden['queries'])
    assert set(empty['queries']).isdisjoint(event['queries'])
    for query in burden['queries'].values():
        assert query['masked_response'].endswith(' '.join(source_kernel.FILLERS[:4]))
        assert query['question'] == kernel.QUESTION_TEMPLATE.format(masked_response=query['masked_response'])


def test_query_contains_no_response_person_values_except_legitimate_source_evidence():
    result = kernel.prepare_masked_queries([sample()])
    for query in result['queries'].values():
        payload = json.loads(query['messages'][1]['content'])
        assert set(payload) == {'source', 'question'}
        assert payload['source'] == query['source']
        assert all(name not in payload['question'] for name in query['candidates'])
        assert all(name in payload['source'] for name in query['candidates'])
        assert all(key not in payload for key in ('text', 'candidate_index', 'alignment', 'mask_spans', 'candidates'))
        assert '[TARGET]' in query['masked_response'] and '[OTHER]' in query['masked_response']
    for request in result['requests'].values():
        assert request['eos_policy'] == source_kernel.EOS_POLICY
        assert request['candidate'] in result['queries'][request['query_key']]['candidates']


def test_candidate_order_source_only_and_source_flip_never_selects_a_binding():
    a = kernel.prepare_masked_queries([sample(s=0)])
    b = kernel.prepare_masked_queries([sample(s=1)])
    aq, bq = list(a['queries'].values()), list(b['queries'].values())
    assert [q['candidates'] for q in aq] == [q['candidates'] for q in bq]
    assert [q['question'] for q in aq] == [q['question'] for q in bq]
    assert set(a['queries']).isdisjoint(b['queries'])
    assert all('bindings' not in metadata for metadata in a['sources'].values())


def test_mask_every_repeated_literal_and_preserve_unicode_punctuation_and_whitespace():
    text = '😀 Ava Lorn met Bex Pell.\nAva Lorn / Bex Pell — Cafe\u0301.  '
    result = kernel._mask_person_values(text, 'Ava Lorn', 'Bex Pell')
    assert result['status'] == 'ok'
    assert result['masked_response'] == '😀 [TARGET] met [OTHER].\n[TARGET] / [OTHER] — Cafe\u0301.  '
    assert len(result['mask_spans']) == 4
    for span in result['mask_spans']:
        assert text[span['start']:span['end']] == span['text']
    assert len({(span['start'], span['end']) for span in result['mask_spans']}) == 4


@pytest.mark.parametrize('kind', ['placeholder', 'overlap', 'missing', 'same'])
def test_mask_ambiguity_is_explicit_unknown(kind):
    text, a, b = 'Ava Lorn met Bex Pell.', 'Ava Lorn', 'Bex Pell'
    if kind == 'placeholder': text += ' [TARGET]'
    elif kind == 'overlap': text, a, b = 'Ava Lorna met Ava Lorn.', 'Ava Lorn', 'Ava Lorna'
    elif kind == 'missing': text = 'Ava Lorn left.'
    else: b = a
    result = kernel._mask_person_values(text, a, b)
    assert result['status'] == 'unknown' and result['reason'] and 'masked_response' not in result


@pytest.mark.parametrize('kind', ['source', 'alias', 'suffix', 'duplicate_person', 'invalid_type'])
def test_invalid_toy_input_retains_original_and_two_unknown_occurrences_without_queries(kind):
    row = sample()
    if kind == 'source': row['source'] = row['source'].replace(source_kernel.FILLERS[0], 'Unsupported source fact.')
    elif kind == 'alias': row['response'] = row['response'].replace('Ava Lorn', 'A. Lorn')
    elif kind == 'suffix': row['response'] += 'Ava Lorn was mentioned again.'
    elif kind == 'duplicate_person': row['response'] = row['response'].replace('Bex Pell', 'Ava Lorn')
    else: row['response'] = None
    result = kernel.prepare_masked_queries([row])
    assert len(result['originals']) == 1
    original = result['originals'][0]
    assert original['status'] == 'unknown' and original['reason']
    assert len(original['occurrences']) == result['counts']['unknown_occurrences'] == 2
    assert all(slot['status'] == 'unknown' and slot['query_key'] is None for slot in original['occurrences'])
    assert result['queries'] == result['requests'] == {}


def test_overlapping_source_person_names_are_retained_unknown_not_partially_erased():
    passage, people, events = source(people=('Ava Lorn', 'Ava Lorna'))
    row = dict(id='overlap', source=passage, response=response('winner', people, events))
    result = kernel.prepare_masked_queries([row])
    assert result['counts']['unknown_occurrences'] == 2
    assert result['originals'][0]['reason'] == 'overlapping_literal_person_values'


def test_attribute_samples_supported_and_result_mutation_cannot_mutate_future_payloads():
    row = SimpleNamespace(**sample())
    a = kernel.prepare_masked_queries([row])
    request = next(iter(a['requests'].values()))
    request['messages'][0]['content'] = 'changed locally'
    b = kernel.prepare_masked_queries([row])
    assert all(item['messages'][0]['content'] != 'changed locally' for item in b['requests'].values())
    assert all(query['messages'][0]['content'] != 'changed locally' for query in a['queries'].values())


def test_duplicate_identifiers_fail_before_a_cohort_can_be_ambiguously_joined():
    with pytest.raises(ValueError, match='unique'):
        kernel.prepare_masked_queries([sample(), sample(r=1)])

"""H008 pure source-only construction/slot tests; no models or truth artifacts."""
from copy import deepcopy
from itertools import product
import json
import math

import pytest

from experiments.research import source_reconstruction as kernel


def source(family='winner', vocabulary=0, direction=0):
    people = ('Ava Lorn', 'Bex Pell') if vocabulary == 0 else ('Cara Wynn', 'Dax Ford')
    events = ('Amber Cup', 'Cobalt Cup') if family == 'winner' else ('Amber Accord', 'Cobalt Accord')
    assigned = people if direction == 0 else people[::-1]
    if family == 'winner':
        facts = [text for event, person in zip(events, assigned) for text in (
            f'The {event} final was a match between {people[0]} and {people[1]}.',
            f'{person} was the sole winner of the {event} final.')]
    else:
        facts = [f'{person} was the sole signer of the {event}.' for person, event in zip(assigned, events)]
    return ' '.join([*facts, *kernel.FILLERS]), people, events


def response(family, people, events, direction=0, burden=0):
    a, b = people if direction == 0 else people[::-1]
    x, y = events
    target = (f'{a} won the {x} final, and {b} won the {y} final. ' if family == 'winner' else
              f'{a} was the sole signer of the {x}, and {b} was the sole signer of the {y}. ')
    return target + ' '.join(kernel.FILLERS[:burden])


def test_all48_cells_produce16_source_queries32_completions_and96_exact_slots():
    queries, requests, slots = {}, {}, 0
    for family, vocabulary, s, r, burden in product(('winner', 'sole_signer'), (0, 1), (0, 1), (0, 1), (0, 4, 12)):
        passage, people, events = source(family, vocabulary, s)
        prepared = kernel.candidate_requests(passage)
        assert prepared['status'] == 'ok' and len(prepared['queries']) == 2 and len(prepared['requests']) == 4
        queries.update({q['query_key']: q for q in prepared['queries']})
        requests.update({item['observation_key']: item for item in prepared['requests']})
        text = response(family, people, events, r, burden)
        aligned = kernel.align_response(passage, text)
        assert aligned['status'] == 'ok' and len(aligned['slots']) == 2
        for slot in aligned['slots']:
            assert text[slot['start']:slot['end']] == slot['text']
            assert slot['query_key'] in queries
        slots += len(aligned['slots'])
    assert len(queries) == 16 and len(requests) == 32 and slots == 96


def test_query_bytes_ignore_every_response_flip_burden_and_unrecognized_response():
    passage, people, events = source()
    frozen = json.dumps(kernel.candidate_requests(passage), sort_keys=True)
    for r, burden in product((0, 1), (0, 4, 12)):
        kernel.align_response(passage, response('winner', people, events, r, burden))
        assert json.dumps(kernel.candidate_requests(passage), sort_keys=True) == frozen
    unknown = kernel.align_response(passage, 'Eve Mann supplied a response-side hint.')
    assert unknown['status'] == 'unknown'
    assert json.dumps(kernel.candidate_requests(passage), sort_keys=True) == frozen
    assert 'Eve Mann' not in frozen


@pytest.mark.parametrize('family', ['winner', 'sole_signer'])
def test_source_flip_keeps_candidates_and_event_questions_in_canonical_order(family):
    a = kernel.source_queries(source(family, direction=0)[0])
    b = kernel.source_queries(source(family, direction=1)[0])
    assert a['people'] == b['people'] == sorted(a['people'])
    assert a['events'] == b['events'] == sorted(a['events'])
    assert [q['question'] for q in a['queries']] == [q['question'] for q in b['queries']]
    assert [q['query_key'] for q in a['queries']] != [q['query_key'] for q in b['queries']]


def test_event_flip_is_a_distinct_query_not_always_the_first_source_person():
    passage, people, events = source('sole_signer')
    prepared = kernel.source_queries(passage)
    assert prepared['queries'][0]['query_key'] != prepared['queries'][1]['query_key']
    for query in prepared['queries']:
        assert all(person not in query['question'] for person in people)
        payload = json.loads(query['messages'][1]['content'])
        assert set(payload) == {'source', 'question'} and payload['source'] == passage
    aligned = kernel.align_response(passage, response('sole_signer', people, events))
    assert aligned['slots'][0]['text'] != aligned['slots'][1]['text']
    assert aligned['slots'][0]['query_key'] != aligned['slots'][1]['query_key']


def test_repeated_response_name_has_two_distinct_exact_occurrences():
    passage, people, events = source()
    text = response('winner', people, events).replace(people[1], people[0])
    aligned = kernel.align_response(passage, text)
    assert aligned['status'] == 'ok'
    first, second = aligned['slots']
    assert first['text'] == second['text'] == people[0]
    assert first['end'] < second['start'] and first['query_key'] != second['query_key']
    assert text[first['start']:first['end']] == text[second['start']:second['end']] == people[0]


@pytest.mark.parametrize('change', ['extra_person', 'missing_filler', 'nonbijective', 'reflexive'])
def test_unsupported_or_ambiguous_source_is_retained_as_unknown(change):
    passage, people, _ = source()
    if change == 'extra_person': passage = passage.replace('was the sole winner', 'and Eve Mann were the winners', 1)
    elif change == 'missing_filler': passage = passage.replace(kernel.FILLERS[3] + ' ', '')
    elif change == 'nonbijective': passage = passage.replace(people[1] + ' was the sole winner', people[0] + ' was the sole winner')
    else: passage = passage.replace(f'between {people[0]} and {people[1]}', f'between {people[0]} and {people[0]}')
    parsed = kernel.candidate_requests(passage)
    assert parsed['status'] == 'unknown' and parsed['reason'] and parsed['queries'] == parsed['requests'] == []


@pytest.mark.parametrize('change', ['outside_person', 'outside_event', 'duplicate_event', 'unsupported_wording', 'empty'])
def test_response_failures_do_not_create_guessed_slots(change):
    passage, people, events = source()
    text = response('winner', people, events)
    if change == 'outside_person': text = text.replace(people[0], 'Eve Mann')
    elif change == 'outside_event': text = text.replace(events[0], 'Emerald Cup')
    elif change == 'duplicate_event': text = text.replace(events[1], events[0])
    elif change == 'unsupported_wording': text = text.replace(' won ', ' might win ', 1)
    else: text = ''
    aligned = kernel.align_response(passage, text)
    assert aligned['status'] == 'unknown' and aligned['reason'] and aligned['slots'] == []


def test_completion_rendering_has_full_names_and_explicit_eos_without_answer_list():
    passage, people, _ = source()
    requests = kernel.candidate_requests(passage)['requests']
    for item in requests:
        assert item['candidate'] in people and ' ' in item['candidate']
        assert item['eos_policy'] == kernel.EOS_POLICY
        assert item['eos_policy']['include_in_sequence_logprob'] is True
        payload = json.loads(item['messages'][1]['content'])
        assert 'candidates' not in payload and all(person not in payload['question'] for person in people)
        assert 'response' not in payload and 'masked_claim' not in payload
    requests[0]['messages'][0]['content'] = 'mutated local request'
    assert kernel.candidate_requests(passage)['requests'][0]['messages'][0]['content'] != 'mutated local request'


def test_full_sequence_normalization_is_stable_and_has_no_length_or_temperature_fit():
    query = kernel.source_queries(source()[0])['queries'][0]
    a, b = query['candidates']
    normal = kernel.candidate_distribution(query, {a: -12., b: -14.})
    assert normal['status'] == 'ok' and normal['probabilities'][a] == pytest.approx(1 / (1 + math.exp(-2)))
    extreme = kernel.candidate_distribution(query, {a: -1e300, b: -1e300})
    assert extreme['probabilities'] == {a: .5, b: .5}
    assert normal['conditional_on_candidate_set'] is True and normal['calibrated_truth_probability'] is False


def test_equal_extremely_negative_finite_scores_stay_normalized():
    query = kernel.source_queries(source()[0])['queries'][0]
    a, b = query['candidates']
    result = kernel.candidate_distribution(query, {a: -1e308, b: -1e308})
    assert result['status'] == 'ok' and result['probabilities'] == {a: .5, b: .5}
    assert math.fsum(result['probabilities'].values()) == pytest.approx(1.0)


@pytest.mark.parametrize('bad', ['nan', 'bool', 'missing', 'extra'])
def test_malformed_candidate_scores_are_unknown(bad):
    query = kernel.source_queries(source()[0])['queries'][0]
    a, b = query['candidates']; scores = {a: -2., b: -3.}
    if bad == 'nan': scores[a] = float('nan')
    elif bad == 'bool': scores[a] = True
    elif bad == 'missing': del scores[b]
    else: scores['Eve Mann'] = -1.
    result = kernel.candidate_distribution(query, scores)
    assert result['status'] == 'unknown' and result['probabilities'] is None

"""Synthetic H016 plans/readouts only: no tokenizer, model, files or labels."""
from copy import deepcopy
from itertools import product
import json
import math
from types import SimpleNamespace

import pytest

from experiments.research import paired_scope as kernel
from experiments.research import masked_reconstruction as masked
from experiments.research import atomic_reconstruction as atomic
from experiments.research.bijection_readout import assignment_distribution
from experiments.research.source_reconstruction import FILLERS, candidate_distribution


def samples():
    rows = []
    for family, vocabulary, source_direction, response_direction in product(range(2), repeat=4):
        people = ('Ava Lorn', 'Bex Pell') if vocabulary == 0 else ('Cara Wynn', 'Dax Ford')
        prefix, suffix = ('Amber', 'Cobalt') if vocabulary == 0 else ('Silver', 'Violet')
        events = [f'{prefix} Cup', f'{suffix} Cup'] if family == 0 else [f'{prefix} Accord', f'{suffix} Accord']
        assigned = people if source_direction == 0 else people[::-1]
        if family == 0:
            facts = [fact for event, person in zip(events, assigned) for fact in (
                f'The {event} final was a match between {people[0]} and {people[1]}.',
                f'{person} was the sole winner of the {event} final.')]
        else:
            facts = [f'{person} was the sole signer of the {event}.' for event, person in zip(events, assigned)]
        a, b = people if response_direction == 0 else people[::-1]
        response = (f'{a} won the {events[0]} final, and {b} won the {events[1]} final. ' if family == 0 else
                    f'{a} was the sole signer of the {events[0]}, and {b} was the sole signer of the {events[1]}. ')
        rows.append(dict(id=f'{family}:{vocabulary}:{source_direction}:{response_direction}',
            source_id=f'{family}:{vocabulary}:{source_direction}', source=' '.join([*facts, *FILLERS]), response=response))
    return rows


def observations(plan, matrix=((-1., -2.), (-1.5, -4.))):
    result = {}
    for key, query in plan['queries'].items():
        source = plan['sources'][query['source_sha256']]
        scores = matrix[source['events'].index(query['event'])]
        result[key] = dict(status='ok', reason=None, paths={name: dict(raw_joint_log_probability=value)
            for name, value in zip(query['candidates'], scores)}, candidate_pool_mass=sum(math.exp(v) for v in scores))
    return result


def test_all16_originals_two_scopes_keep_every_logical_slot_and_exact_frozen_query():
    rows = samples(); unchanged = deepcopy(rows)
    plan = kernel.prepare(SimpleNamespace(**row) for row in rows)
    assert rows == unchanged
    assert plan['counts']['originals'] == 16 and plan['counts']['sources'] == 8
    assert plan['counts']['unique_queries'] == 32 and plan['counts']['unique_candidates'] == 64
    for scope, compiler in (('compound', masked.prepare_masked_queries), ('atomic', atomic.prepare_atomic_queries)):
        old = compiler(rows)
        assert plan['counts']['by_scope'][scope] == dict(originals=16, sources=8, logical_queries=32,
            unique_queries=16, logical_candidates=64, unique_candidates=32, unknown_slots=0)
        assert {key: {k: v for k, v in query.items() if k != 'scope'} for key, query in plan['queries'].items()
                if query['scope'] == scope} == old['queries']
        assert {key: {k: v for k, v in row.items() if k != 'scope'} for key, row in plan['candidate_rows'].items()
                if row['scope'] == scope} == old['requests']
        for original, previous in zip(plan['originals'], old['originals']):
            assert original['scopes'][scope] == previous
            for slot in previous['occurrences']:
                assert original['sample']['response'][slot['start']:slot['end']] == slot['text']
                assert len(slot['observation_keys']) == 2
                assert all(key in plan['candidate_rows'] for key in slot['observation_keys'])


def test_response_swap_collapses_questions_candidates_and_preserves_system_template():
    rows = samples()
    a, b = kernel.prepare(rows[::2]), kernel.prepare(rows[1::2])
    assert a['sources'] == b['sources'] and a['queries'] == b['queries'] and a['candidate_rows'] == b['candidate_rows']
    for key, query in a['queries'].items():
        payload = json.loads(query['messages'][1]['content'])
        assert payload == dict(source=query['source'], question=query['question'])
        assert query['question'] == masked.QUESTION_TEMPLATE.format(masked_response=query['masked_response'])
        assert all(name not in query['question'] for name in query['candidates'])
        if query['scope'] == 'atomic':
            assert '[OTHER]' not in query['masked_response'] and '[OTHER]' in query['question']
            assert ', and ' not in query['masked_response']
    assert {q['messages'][0]['content'] for q in a['queries'].values()} == {
        q['messages'][0]['content'] for q in masked.prepare_masked_queries(rows)['queries'].values()}


def test_canonical_order_survives_source_flip_and_event_mention_order():
    rows = samples()
    row = deepcopy(rows[0])
    row['response'] = 'Bex Pell won the Cobalt Cup final, and Ava Lorn won the Amber Cup final. '
    plan = kernel.prepare([row])
    output = kernel.derive(plan, observations(plan))
    for matrix in output['matrices'].values():
        assert matrix['events'] == ['Amber Cup', 'Cobalt Cup']
        assert matrix['candidates'] == ['Ava Lorn', 'Bex Pell']
    assert all(value['observed_candidate_indices_by_canonical_event'] == [0, 1]
               for value in output['cases'][0]['scopes'].values())
    assert {tuple(s['people']) for s in kernel.prepare(rows[:4])['sources'].values()} == {('Ava Lorn', 'Bex Pell')}


def test_all_four_arms_reuse_the_two_numeric_kernels_with_raw_joint_not_pool_normalization(monkeypatch):
    plan = kernel.prepare(samples()); raw = observations(plan)
    frozen = deepcopy((plan, raw)); joint_calls, row_calls = [], []
    def joint(matrix, **kwargs):
        joint_calls.append(deepcopy((matrix, kwargs)))
        return assignment_distribution(matrix, **kwargs)
    def row(query, scores):
        row_calls.append((query['query_key'], deepcopy(scores)))
        return candidate_distribution(query, scores)
    monkeypatch.setattr(kernel, 'assignment_distribution', joint)
    monkeypatch.setattr(kernel, 'candidate_distribution', row)
    result = kernel.derive(plan, raw)
    assert (plan, raw) == frozen
    assert len(joint_calls) == 16 and len(row_calls) == 32
    assert all(matrix == [[-1., -2.], [-1.5, -4.]] for matrix, _ in joint_calls)
    assert all(kwargs['source_is_admitted_bijection'] is True for _, kwargs in joint_calls)
    assert len(result['cases']) == 16 and len(result['matrices']) == 16 and len(result['queries']) == 32
    for case in result['cases']:
        assert set(case['readouts']) == set(kernel.ARMS)
        for scope in kernel.SCOPES:
            matrix = result['matrices'][case['scopes'][scope]['matrix_id']]
            assert matrix['independent_same_name_collapse'] is True
            assert matrix['bijection']['map_assignment'] == [1, 0]
            index = case['scopes'][scope]['observed_assignment_index']
            assert case['readouts'][f'{scope}_{kernel.READOUTS[1]}']['risk'] == 1. - matrix['bijection']['assignment_probabilities'][index]
            slots = case['scopes'][scope]['alignment']['occurrences']
            assert case['readouts'][f'{scope}_{kernel.READOUTS[0]}']['risk'] == max(
                1. - result['queries'][slot['query_key']]['distribution']['probabilities'][slot['text']] for slot in slots)
    assert result['evaluation_material_accessed'] is False and result['pure_kernel_model_calls'] == 0


def test_scope_specific_evidence_is_not_pooled_or_used_to_choose_an_arm():
    plan = kernel.prepare(samples()); raw = observations(plan)
    for key, query in plan['queries'].items():
        if query['scope'] == 'atomic':
            event_index = plan['sources'][query['source_sha256']]['events'].index(query['event'])
            for name, value in zip(query['candidates'], ((-1., -4.), (-4., -1.))[event_index]):
                raw[key]['paths'][name]['raw_joint_log_probability'] = value
    result = kernel.derive(plan, raw)
    assert all(matrix['bijection']['map_assignment'] == ([0, 1] if matrix['scope'] == 'atomic' else [1, 0])
               for matrix in result['matrices'].values())
    assert result['arms'] == list(kernel.ARMS) and result['all_fixed_readouts_retained'] is True


def test_exact_ties_and_extreme_finite_scores_remain_explicit_without_tolerance():
    plan = kernel.prepare(samples())
    result = kernel.derive(plan, observations(plan, ((-1e308, -1e308), (-1e308, -1e308))))
    assert all(row['distribution']['probabilities'] == {name: .5 for name in row['candidates']}
               and row['tie'] is True and row['candidate_argmax_index'] is None for row in result['queries'].values())
    # Summing finite extreme rows overflows the *existing* assignment kernel;
    # independent rows stay .5 while joint must retain unknown/risk1.
    assert all(matrix['bijection']['status'] == 'unknown' for matrix in result['matrices'].values())
    assert all(case['readouts'][arm]['risk'] == (1. if 'bijection' in arm else .5) for case in result['cases'] for arm in kernel.ARMS)
    tied = kernel.derive(plan, observations(plan, ((-1., -2.), (-1., -2.))))
    assert all(matrix['bijection']['tie'] is True and matrix['bijection']['map_assignment'] is None
               and matrix['bijection']['assignment_probabilities'] == [.5, .5] for matrix in tied['matrices'].values())


def test_explicit_unknown_and_nonfinite_readouts_retain_all_cases_and_other_scope():
    plan = kernel.prepare(samples()); raw = observations(plan)
    key = next(iter(plan['queries'])); scope = plan['queries'][key]['scope']
    raw[key] = dict(status='unknown', reason='engineering_gate_failed')
    result = kernel.derive(plan, raw)
    assert len(result['cases']) == 16 and len(result['queries']) == 32
    affected = [case for case in result['cases'] if any(slot['query_key'] == key
                for slot in case['scopes'][scope]['alignment']['occurrences'])]
    assert len(affected) == 2
    assert all(case['readouts'][f'{scope}_{readout}']['unknown_fallback'] is True
               and case['readouts'][f'{scope}_{readout}']['risk'] == 1. for case in affected for readout in kernel.READOUTS)
    raw = observations(plan); name = plan['queries'][key]['candidates'][0]
    raw[key]['paths'][name]['raw_joint_log_probability'] = float('nan')
    invalid = kernel.derive(plan, raw)
    assert invalid['queries'][key]['status'] == 'unknown'
    assert 'nonfinite' in invalid['queries'][key]['reason']
    json.dumps(invalid, allow_nan=False)


def test_non_b0_invalid_sources_and_nonbijective_response_are_retained_without_repair():
    rows = samples()[:3]
    rows[0]['response'] += ' '.join(FILLERS[:4])
    rows[1]['source'] = 'unsupported source grammar'
    rows[2]['response'] = rows[2]['response'].replace('Bex Pell', 'Ava Lorn')
    plan = kernel.prepare(rows)
    assert len(plan['originals']) == 3 and not plan['queries'] and not plan['candidate_rows']
    assert all(plan['counts']['by_scope'][scope]['unknown_slots'] == 6 for scope in kernel.SCOPES)
    result = kernel.derive(plan, {})
    assert len(result['cases']) == 3
    assert all(value['risk'] == 1. and value['unknown_fallback'] is True
               for case in result['cases'] for value in case['readouts'].values())


def test_plan_and_score_population_mismatches_are_rejected_not_interpreted_as_semantic_unknown():
    plan = kernel.prepare(samples()); raw = observations(plan)
    changed = deepcopy(plan); next(iter(changed['queries'].values()))['messages'][0]['content'] += ' changed'
    with pytest.raises(ValueError, match='plan changed'):
        kernel.derive(changed, raw)
    changed = deepcopy(plan)
    changed['originals'][0]['scopes']['compound']['occurrences'][1]['index'] = True
    with pytest.raises(ValueError, match='plan changed'):
        kernel.derive(changed, raw)
    for bad in ({key: value for i, (key, value) in enumerate(raw.items()) if i}, dict(raw, extra={} )):
        with pytest.raises(ValueError, match='Exact planned query'):
            kernel.derive(plan, bad)
    changed = deepcopy(raw); next(iter(changed.values()))['paths']['unplanned person'] = dict(raw_joint_log_probability=-1.)
    with pytest.raises(ValueError, match='Exact complete candidate paths'):
        kernel.derive(plan, changed)
    with pytest.raises(ValueError, match='Unique nonempty original IDs'):
        kernel.prepare([samples()[0], samples()[0]])


def test_empty_and_nonstring_unknown_inputs_do_not_invent_sources_or_observations():
    empty = kernel.prepare([])
    assert kernel.derive(empty, {})['cases'] == []
    result = kernel.derive(kernel.prepare([dict(id='unknown', source=None, response=None)]), {})
    assert len(result['cases']) == 1 and result['matrices'] == result['queries'] == {}
    assert all(value['unknown_fallback'] is True and value['risk'] == 1. for value in result['cases'][0]['readouts'].values())


def test_compiler_framing_failure_is_retained_and_output_does_not_alias_input():
    rows = samples()[:1]; rows[0]['response'] = rows[0]['response'].rstrip()
    plan = kernel.prepare(rows)
    assert plan['originals'][0]['scopes']['atomic']['status'] == 'unknown'
    assert plan['originals'][0]['scopes']['atomic']['reason'].startswith('atomic_compiler:')
    output = kernel.derive(plan, observations(plan))
    output['cases'][0]['scopes']['compound']['alignment']['occurrences'][0]['text'] = 'outside mutation'
    assert plan['originals'][0]['scopes']['compound']['occurrences'][0]['text'] != 'outside mutation'

"""Tiny tokenizer/semantic contracts; no original tokenizer, weights or labels."""

import ast
from copy import deepcopy
from dataclasses import replace
import re

import pytest

from experiments.datasets import Sample
from experiments.research import relation_scope as scope
from experiments.research.relation_inventory import source_inventory
from myproject.inputs import encode_response, encode_source
from .test_relation_intervention import WordTokenizer


class ClockTokenizer(WordTokenizer):
    def __init__(self, mode='normal'):
        super().__init__()
        self.vocabulary = {'00': 410, '30': 966}
        self.mode = mode

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False and return_offsets_mapping is True
        prefix = r"',|"  # An explicit quote/comma boundary-overlap carrier.
        if "'Tuesday': '17:30" in text and self.mode in {'length', 'positions'}:
            prefix += r'Tue(?=sday)|sday|'
            if self.mode == 'positions':
                prefix += r"'Wednesday|"  # Compensate the one extra earlier token.
        pattern = prefix + r'[A-Za-z_]+|\d+|[^A-Za-z_\d]'
        ids, offsets = [], []
        for match in re.finditer(pattern, text):
            self.vocabulary.setdefault(match[0], len(self.vocabulary))
            ids.append(self.vocabulary[match[0]])
            offsets.append(match.span())
        return {'input_ids': ids, 'offset_mapping': offsets}


@pytest.fixture
def prepared(monkeypatch):
    values = {d: scope.A if d == 'Monday' else scope.C if d == 'Sunday' else scope.B
              for d in scope.DAYS}
    source = repr({'name': 'Finch & Fork', 'note': 'keep café / 原样 7:05', 'hours': values,
                   'unrelated': {'number': 123, 'text': 'never normalize me'}})
    response = 'Open Monday to Saturday from 17:00-21:00, Sunday from 9:00-14:00.'
    sample = Sample(id='5670', source_id='13601', task='Data2txt', split='test',
                    prompt='Keep this instruction.\nSource:' + source, source=source, response=response)
    tokenizer = ClockTokenizer()
    prompt = encode_source(tokenizer, sample.prompt, source)
    answer = encode_response(tokenizer, response)
    def at(start, text):
        return next(i for i, span in enumerate(answer['offsets'])
                    if list(span) == [start, start + len(text)])
    hour = at(response.index('17:'), '17')
    minute = at(response.index('17:') + 3, '00')
    end_hour = at(response.index('-21') + 1, '21')
    phases = ((at(response.index('Saturday'), 'Saturday'), 'range_end'),
              (hour - 1, 'pre_time'), (minute - 1, 'pre_minute'),
              (minute, 'post_minute'), (end_hour - 1, 'pre_end_hour'),
              (end_hour, 'post_end_hour'))
    ids = prompt['prompt_with_source'] + answer['answer_ids']
    axes = dict(token_ids=ids, answer_ids=answer['answer_ids'],
                source_mask=prompt['source_mask'] + [False] * len(answer['answer_ids']),
                query_positions=list(range(len(prompt['prompt_with_source']) - 1, len(ids))),
                answer_offsets=[list(span) for span in answer['offsets']])
    license_ = dict(case_id='5670', source_id='13601', source_sha256=scope.text_hash(source),
                    source_characters=len(source), licensed_facts=[])
    for field in source_inventory(source, '13601')['fields']:
        if field['owner_path'] != ['hours']:
            continue
        def span(start, stop):
            return dict(start=start, end=stop, text=source[start:stop])
        license_['licensed_facts'].append(dict(M0=dict(path=field['path']),
            time_scope=dict(weekday=field['path'][1]), value=dict(raw=field['value']),
            source_spans=dict(interval_value_literal=span(*field['span']),
                interval_payload=span(field['span'][0] + 1, field['span'][1] - 1),
                weekday_key_literal=span(*field['key_span']))))
    for key, value in dict(SOURCE_SHA256=scope.text_hash(source),
        RESPONSE_SHA256=scope.text_hash(response), SOURCE_CHARACTERS=len(source),
        NATIVE_SHAPE=(len(prompt['prompt_with_source']), len(answer['answer_ids']), len(ids)),
        NATIVE_SOURCE_TOKENS=sum(prompt['source_mask']), MINUTE_TARGET=minute,
        HOUR_TARGET=hour, END_HOUR_TARGET=end_hour, R_PHASES=phases).items():
        monkeypatch.setattr(scope, key, value)
    return sample, tokenizer, axes, license_


def test_five_worlds_keep_exact_unedited_characters_and_complete_response(prepared):
    sample, tokenizer, axes, license_ = prepared
    before = deepcopy(license_)
    result = scope.prepare_worlds(*prepared)
    assert license_ == before
    assert result['world_order'] == list(scope.WORLD_ORDER)
    assert result['budget'] == {'model_calls': 5, 'head_calls': 5}
    assert result['candidate_ids'] == {'00': 410, '30': 966}
    assert not result['candidate30_is_complete_range_correct_answer']
    assert result['direct_rule_baseline'] == dict(zip(scope.WORLD_ORDER, [False, False, True, False, False]))
    assert result['worlds']['native']['source'] == sample.source
    for name, world in result['worlds'].items():
        assert world['answer_ids'] == axes['answer_ids']
        assert world['answer_offsets'] == axes['answer_offsets']
        assert world['whole_response_support'] == 'not_assessed_other_claims_unknown'
        assert ast.literal_eval(world['source'])['note'] == 'keep café / 原样 7:05'
        assert ast.literal_eval(world['source'])['unrelated']['text'] == 'never normalize me'
        for region in world['unchanged_source_regions']:
            assert sample.source[slice(*region['native_span'])] == world['source'][slice(*region['world_span'])]
        if name != 'native':
            assert world['complete_interval_counts'] == {scope.A: 1, scope.B: 6}
            assert not world['later_Sunday_C_baseline']
            assert len(world['source']) == len(sample.source) + 1
    assert result['worlds']['S_plus']['hours_assignment']['Sunday'] == scope.A
    assert result['worlds']['Tuesday_exception']['hours_assignment']['Tuesday'] == scope.A
    assert result['worlds']['Saturday_exception']['hours_assignment']['Saturday'] == scope.A


def test_six_response_and_seven_source_receivers_keep_pre_post_meaning(prepared):
    result = scope.prepare_worlds(*prepared)
    for world in result['worlds'].values():
        assert len(world['capture_positions']) == len(set(world['capture_positions'])) == 13
        assert world['capture_positions'] == sorted(world['capture_positions'])
        response = {r['name']: r for r in world['capture_receivers'] if r['kind'] == 'response'}
        assert response['pre_minute']['target_answer_index'] == scope.MINUTE_TARGET
        assert response['post_minute']['target_answer_index'] == scope.MINUTE_TARGET + 1
        assert response['pre_end_hour']['target_answer_index'] == scope.END_HOUR_TARGET
        assert response['post_end_hour']['target_answer_index'] == scope.END_HOUR_TARGET + 1
        assert world['primary_query_position'] == response['pre_minute']['query_position']
        for day, closure in world['source_closures'].items():
            assert world['source_mask'][closure['query_position']]
            assert closure['path'] == ['hours', day]
            assert closure['semantic_encoding_or_use'] == 'unknown_not_inferred'
        assert any(c['status'] == 'boundary_overlap' for c in world['source_closures'].values())


def test_controlled_total_length_mismatch_stops_without_string_rescue(prepared):
    sample, original, axes, license_ = prepared
    tokenizer = ClockTokenizer('length')
    tokenizer.vocabulary = dict(original.vocabulary)
    with pytest.raises(scope.EligibilityError, match='matched counts/total axes'):
        scope.prepare_worlds(sample, tokenizer, axes, license_)


def test_equal_total_length_with_internal_position_changes_is_explicit_audit(prepared):
    sample, original, axes, license_ = prepared
    tokenizer = ClockTokenizer('positions')
    tokenizer.vocabulary = dict(original.vocabulary)
    result = scope.prepare_worlds(sample, tokenizer, axes, license_)
    audit = result['eligibility_audit']['controlled_pairwise_vs_S_minus']['Tuesday_exception']
    assert audit['source_closure_position_changes']
    assert audit['outside_all_licensed_hours_payload_difference_positions']
    assert not audit['internal_position_identity_claimed']
    assert not audit['activation_interpatch_performed']


@pytest.mark.parametrize('fault', ['missing_day', 'unknown_span', 'response', 'mask', 'nonboolean_mask', 'offsets'])
def test_unknown_or_changed_license_and_original_axes_are_rejected(prepared, fault):
    sample, tokenizer, axes, license_ = deepcopy(prepared)
    if fault == 'missing_day':
        license_['licensed_facts'].pop()
    elif fault == 'unknown_span':
        license_['licensed_facts'][0]['source_spans']['interval_payload']['start'] = None
    elif fault == 'response':
        sample = replace(sample, response=sample.response.replace('00', '01', 1))
    elif fault == 'mask':
        axes['source_mask'][0] = True
    elif fault == 'nonboolean_mask':
        axes['source_mask'][0] = 0
    else:
        axes['answer_offsets'][scope.MINUTE_TARGET][1] -= 1
    with pytest.raises((scope.EligibilityError, ValueError)):
        scope.prepare_worlds(sample, tokenizer, axes, license_)


def test_complete_primary_numeral_cannot_be_a_crossing_or_split_token():
    with pytest.raises(scope.EligibilityError, match='one complete'):
        scope._complete_token([[0, 1], [1, 2]], 0, 2)
    with pytest.raises(scope.EligibilityError, match='one complete'):
        scope._complete_token([[0, 3]], 1, 3)


def _observations():
    margins = dict(native=20., S_minus=0., S_plus=1., Tuesday_exception=.2, Saturday_exception=.3)
    return {name: dict(candidates={'00': {'logit': value / 2}, '30': {'logit': -value / 2}})
            for name, value in margins.items()}


def test_primary_requires_all_three_frozen_contrasts_and_reports_control_spread():
    observations = _observations()
    result = scope.scientific_summary(observations)
    assert result['primary_all_three_pass']
    assert result['margins']['native'] == 20  # Original anchor is excluded.
    assert result['primary_contrasts']['S_minus']['engineering_allowance'] == pytest.approx(.0081)
    assert result['unsupported_world_variation']['spread'] == pytest.approx(.3)
    observations['Saturday_exception']['candidates']['00']['logit'] = .499
    observations['Saturday_exception']['candidates']['30']['logit'] = -.499
    result = scope.scientific_summary(observations)
    assert not result['primary_all_three_pass']
    assert result['primary_contrasts']['S_minus']['passed']
    assert not result['primary_contrasts']['Saturday_exception']['passed']
    assert result['auroc'] is None and result['new_detector_scores'] == 0


def test_secondary_all_seven_targets_cannot_replace_failed_primary(prepared):
    observations = _observations()
    for name, row in observations.items():
        row['original_token_endpoints'] = [dict(answer_index=i, logp=-.25)
                                          for i in range(scope.NATIVE_SHAPE[1])]
    observations['S_plus']['candidates']['00']['logit'] = -5.
    result = scope.scientific_summary(observations)
    assert not result['primary_all_three_pass']
    assert not result['secondary_replaces_primary']
    assert result['secondary']['S_plus']['complete_interval_logp_sum'] == -1.75
    del observations['S_minus']['original_token_endpoints']
    with pytest.raises(ValueError, match='every world'):
        scope.scientific_summary(observations)
    observations = _observations()
    observations['S_plus']['candidates']['00']['logit'] = float('nan')
    with pytest.raises(ValueError, match='finite'):
        scope.scientific_summary(observations)

"""Synthetic isolation, malformed-output and same-content format controls."""

from dataclasses import asdict, replace
import json
import sys

import pytest

from experiments.research.semantic_probe import (
    CONTRACT_VERSION, LIMITATIONS, MASK, ClaimFrame, ClaimSlot, ProbeValidationError,
    build_blind, build_direct, build_leaky, build_proposer, leading_whitespace_variants,
    literal_value_leaks, map_variant_span, masked_claim, parse_observation,
    parse_proposal, parse_proposal_items, proposal_coverage, unknown_observation,
)


def proposed(response, value, role='coverage', aliases=()):
    start = response.index(value)
    raw = {'claims': [{'start': 0, 'end': len(response), 'text': response,
                       'slots': [{'role': role, 'start': start, 'end': start + len(value),
                                  'text': value, 'aliases': list(aliases)}]}]}
    frame, = parse_proposal(raw, response)
    return raw, frame, frame.slots[0]


def data(request):
    return json.loads(request['messages'][1]['content'])


def test_response_only_proposer_retains_the_whole_unhighlighted_text():
    response = 'Courtesy first. Account X has an annual period. Final claim remains.'
    request = build_proposer(response)
    assert request['contract_version'] == CONTRACT_VERSION
    assert request['mode'] == 'proposal'
    assert data(request) == {'response': response}
    assert 'source' not in data(request) and 'gold' not in data(request)
    request['messages'][1]['content'] = 'mutated by caller'
    assert data(build_proposer(response)) == {'response': response}


def test_value_only_change_has_identical_blind_query_but_changes_controls():
    a = 'Invoices are sent annually for each monthly coverage period.'
    b = 'Invoices are sent annually for each annual coverage period.'
    _, first, fs = proposed(a, 'monthly')
    # Locate the coverage value, not the unchanged billing-frequency substring.
    start = b.index('annual coverage')
    bs = ClaimSlot('coverage', start, start + 6, 'annual')
    second = ClaimFrame(0, len(b), b, (bs,))
    source = 'Annual and monthly plans exist; the contract identifies annual coverage.'
    blind_a, blind_b = build_blind(source, a, first, fs), build_blind(source, b, second, bs)
    assert blind_a == blind_b
    assert data(blind_a) == {'source': source, 'role': 'coverage',
                             'masked_claim': 'Invoices are sent annually for each ' + MASK + ' coverage period.'}
    assert set(data(blind_a)) == {'source', 'masked_claim', 'role'}
    assert build_direct(source, a, first, fs) != build_direct(source, b, second, bs)
    leaky_a, leaky_b = build_leaky(source, a, first, fs), build_leaky(source, b, second, bs)
    assert leaky_a != leaky_b
    assert data(leaky_a)['masked_claim'] == data(blind_a)['masked_claim']
    assert data(leaky_a)['claimed_value'] == 'monthly'


def test_blind_builder_does_not_send_the_rest_of_the_answer():
    claim = 'The office is in Alpha.'
    response = claim + ' Unrelated asserted value MUST_STAY_OUTSIDE.'
    slot = ClaimSlot('actor', claim.index('Alpha'), claim.index('Alpha') + 5, 'Alpha')
    frame = ClaimFrame(0, len(claim), claim, (slot,))
    request = build_blind('The office is in Beta.', response, frame, slot)
    assert 'MUST_STAY_OUTSIDE' not in json.dumps(request)
    assert data(build_direct('source', response, frame, slot))['claim'] == claim


def test_alias_repetitions_casefold_and_whitespace_are_masked():
    response = 'The New\tYork office is called NEW  YORK and NY.'
    _, frame, slot = proposed(response, 'New\tYork', 'actor', ['NY'])
    query = masked_claim(response, frame, slot)
    assert query.count(MASK) == 3
    assert not literal_value_leaks(query, slot.text, slot.aliases)
    # Matching is literal, not a semantic synonym detector.
    assert literal_value_leaks('The Big Apple office', 'New York') == ()
    assert any('semantic' in limitation for limitation in LIMITATIONS)


def test_casefold_expansion_preserves_original_character_ranges():
    response = 'Straße is also written STRASSE.'
    _, frame, slot = proposed(response, 'Straße', 'actor')
    assert masked_claim(response, frame, slot) == MASK + ' is also written ' + MASK + '.'


@pytest.mark.parametrize('where', ['claim', 'slot'])
def test_proposer_generated_questions_are_not_a_blind_input(where):
    response = 'The period is annual.'
    raw, _, _ = proposed(response, 'annual')
    item = raw['claims'][0] if where == 'claim' else raw['claims'][0]['slots'][0]
    item['question'] = 'Is annual correct?'
    with pytest.raises(ProbeValidationError, match='member set'):
        parse_proposal(raw, response)


@pytest.mark.parametrize('change', ['claim_text', 'slot_text', 'outside', 'bool_offset', 'role'])
def test_invalid_original_spans_or_roles_are_rejected(change):
    response = 'The period is annual.'
    raw, _, _ = proposed(response, 'annual')
    frame, slot = raw['claims'][0], raw['claims'][0]['slots'][0]
    if change == 'claim_text':
        frame['text'] = 'normalized replacement'
    elif change == 'slot_text':
        slot['text'] = 'monthly'
    elif change == 'outside':
        slot['end'] = len(response) + 1
    elif change == 'bool_offset':
        slot['start'] = True
    else:
        slot['role'] = 'invented_truth_role'
    with pytest.raises(ProbeValidationError):
        parse_proposal(raw, response)


def test_unicode_offsets_are_python_characters_not_bytes():
    response = 'α company names 北京.'
    _, frame, slot = proposed(response, '北京', 'actor')
    assert slot.start == response.index('北京')
    assert frame.text[slot.start:slot.end] == '北京'
    assert data(build_blind('The entity is 北京.', response, frame, slot))['source'].endswith('北京.')


def test_default_text_only_proposal_aligns_exact_original_characters():
    response = 'Courtesy. The period is annual. Final assertion.'
    raw = {'claims': [{'text': 'The period is annual.',
                       'slots': [{'role': 'coverage', 'text': 'annual'}]}]}
    frame, = parse_proposal(raw, response)
    slot, = frame.slots
    assert frame.start == response.index(frame.text)
    assert slot.start == response.index('annual')
    assert frame.text == response[frame.start:frame.end]
    assert slot.text == response[slot.start:slot.end]
    assert frame.alignment == slot.alignment == 'unique_literal'
    schema = build_proposer(response)['messages'][0]['content']
    assert 'omit start/end' in schema


def test_slot_uniqueness_is_within_the_claim_not_the_whole_response():
    response = 'Alpha exists. Alpha is an actor.'
    raw = {'claims': [{'text': 'Alpha is an actor.',
                       'slots': [{'role': 'actor', 'text': 'Alpha'}]}]}
    frame, = parse_proposal(raw, response)
    assert frame.slots[0].start == response.rindex('Alpha')


@pytest.mark.parametrize('kind', ['claim', 'slot'])
def test_repeated_literal_needs_explicit_offsets_and_is_never_guessed(kind):
    response = 'Alpha and Alpha.' if kind == 'slot' else 'Alpha. Alpha.'
    raw = {'claims': [{'text': response if kind == 'slot' else 'Alpha.',
                       'slots': [{'role': 'actor', 'text': 'Alpha'}]}]}
    with pytest.raises(ProbeValidationError) as error:
        parse_proposal(raw, response)
    assert error.value.code == 'ambiguous_' + kind
    if kind == 'claim':
        raw['claims'][0].update(start=7, end=13)
    else:
        raw['claims'][0]['slots'][0].update(start=10, end=15)
    frame, = parse_proposal(raw, response)
    selected = frame if kind == 'claim' else frame.slots[0]
    assert selected.alignment == 'explicit_offsets'


@pytest.mark.parametrize('kind', ['claim', 'slot'])
@pytest.mark.parametrize('unpaired', [False, True])
def test_explicit_offsets_are_paired_and_wrong_positions_never_fall_back(kind, unpaired):
    response = 'The period is annual.'
    raw = {'claims': [{'text': response,
                       'slots': [{'role': 'coverage', 'text': 'annual'}]}]}
    target = raw['claims'][0] if kind == 'claim' else raw['claims'][0]['slots'][0]
    target['start'] = 1
    if not unpaired:
        target['end'] = 3
    with pytest.raises(ProbeValidationError) as error:
        parse_proposal(raw, response)
    assert error.value.code == ('unpaired_offsets' if unpaired else kind + '_text_mismatch')


def test_nested_slots_require_explicit_permission_and_crossing_never_passes():
    response = 'Alpha Beta'
    wide = {'role': 'actor', 'start': 0, 'end': 10, 'text': response}
    inner = {'role': 'account', 'start': 6, 'end': 10, 'text': 'Beta'}
    raw = {'claims': [{'start': 0, 'end': 10, 'text': response, 'slots': [wide, inner]}]}
    with pytest.raises(ProbeValidationError, match='explicit nesting'):
        parse_proposal(raw, response)
    raw['claims'][0]['allow_nested_slots'] = True
    assert len(parse_proposal(raw, response)[0].slots) == 2
    raw['claims'][0]['slots'][0].update(end=8, text=response[:8])
    with pytest.raises(ProbeValidationError, match='crossing'):
        parse_proposal(raw, response)


def test_partial_parse_retains_valid_frames_and_raw_invalid_items():
    response = 'Alpha. Beta.'
    valid = {'start': 0, 'end': 6, 'text': 'Alpha.',
             'slots': [{'role': 'actor', 'start': 0, 'end': 5, 'text': 'Alpha'}]}
    invalid = {'start': 7, 'end': 12, 'text': 'Beta.',
               'slots': [{'role': 'actor', 'start': 7, 'end': 11, 'text': 'WRONG'}]}
    result = parse_proposal_items({'claims': [valid, invalid]}, response)
    assert len(result['frames']) == 1 and result['raw_claim_count'] == 2
    assert result['envelope_error'] is None
    assert result['invalid'][0]['index'] == 1
    assert result['invalid'][0]['raw'] == invalid
    assert result['invalid'][0]['policy'] == 'entire_frame_rejected'
    invalid['slots'][0]['text'] = 'caller changed raw object'
    assert result['invalid'][0]['raw']['slots'][0]['text'] == 'WRONG'


def test_duplicate_spans_are_all_rejected_not_chosen_for_favorable_slots():
    response = 'Alpha.'
    raw, _, _ = proposed(response, 'Alpha', 'actor')
    result = parse_proposal_items({'claims': [raw['claims'][0], raw['claims'][0]]}, response)
    assert result['frames'] == () and len(result['invalid']) == 2
    assert {row['code'] for row in result['invalid']} == {'duplicate_claim'}
    with pytest.raises(ProbeValidationError, match='Duplicate original claim'):
        parse_proposal({'claims': [raw['claims'][0], raw['claims'][0]]}, response)


@pytest.mark.parametrize('payload', ['```json\n{}\n```', '{"claims":[],"claims":[]}',
                                    '{"claims":NaN}', '{"claims":{}}'])
def test_malformed_envelopes_are_explicit_not_repaired(payload):
    result = parse_proposal_items(payload, 'A statement.')
    assert result['frames'] == () and result['envelope_error'] is not None
    assert result['raw_claim_count'] is None


def test_decoder_limits_are_retained_as_invalid_json_not_uncaught_exceptions():
    payloads = ['{"claims":' + '[' * 1500 + '0' + ']' * 1500 + '}',
                '{"claims":' + '[' * 40 + '0' + ']' * 40 + '}']
    # CPython's configurable integer limit should remain enabled; do not mutate it.
    if hasattr(sys, 'get_int_max_str_digits') and sys.get_int_max_str_digits():
        payloads.append('{"claims":' + '9' * (sys.get_int_max_str_digits() + 1) + '}')
    for raw in payloads:
        result = parse_proposal_items(raw, 'A factual statement.')
        assert result['frames'] == () and result['invalid'] == ()
        assert result['envelope_error']['code'] == 'invalid_json'
        with pytest.raises(ProbeValidationError) as error:
            parse_observation(raw, 'A source.')
        assert error.value.code == 'invalid_json'
    result = parse_proposal_items('{"claims":[],"claims":[]}', 'A factual statement.')
    assert result['envelope_error']['code'] == 'duplicate_json_key'


def test_syntactic_coverage_does_not_claim_all_facts_were_found():
    response = 'First assertion. Another assertion.'
    text = 'First assertion.'
    frame = ClaimFrame(0, len(text), text, ())
    result = proposal_coverage(response, (frame,))
    assert result['claims_without_slots'] == 1 and result['slots'] == 0
    assert result['covered_claim_characters'] == len(text)
    assert result['semantic_exhaustiveness_verified'] is False
    assert parse_proposal({'claims': []}, response) == ()


def observed(answer='value', quotes=None, status='answerable', reason=''):
    return {'status': status, 'answer': answer, 'quotes': quotes or [], 'reason': reason}


def test_literal_quotes_can_support_an_observation_without_proving_entailment():
    result = parse_observation(observed('DTA', [{'text': 'Deferred tax assets'}]),
                               'The label is Deferred tax assets.')
    assert result.answer == 'DTA'
    assert result.quote_membership_proves_entailment is False
    assert result.qualification_only is True
    assert result.quotes[0].start == 13
    # No automatic contradiction is inferred from the different answer spelling.
    assert 'score' not in asdict(result) and 'probabilities' not in asdict(result)


def test_ambiguous_quotes_need_validated_offsets_including_overlapping_occurrences():
    with pytest.raises(ProbeValidationError) as error:
        parse_observation(observed('red', [{'text': 'red'}]), 'red; red.')
    assert error.value.code == 'ambiguous_quote'
    result = parse_observation(observed('red', [{'text': 'red', 'start': 5, 'end': 8}]),
                               'red; red.')
    assert result.quotes[0].start == 5
    with pytest.raises(ProbeValidationError, match='exactly once'):
        parse_observation(observed('aaa', [{'text': 'aaa'}]), 'aaaa')


@pytest.mark.parametrize('quote', [
    {'text': 'missing'}, {'text': 'Source', 'start': 1, 'end': 7},
    {'text': 'Source', 'start': 0}, {'text': 'source'},
    {'text': 'Source', 'start': False, 'end': 6},
])
def test_quote_membership_is_literal_and_bounds_are_strict(quote):
    with pytest.raises(ProbeValidationError):
        parse_observation(observed('answer', [quote]), 'Source evidence.')


def test_unknown_does_not_invent_an_answer_and_requires_a_reason():
    result = parse_observation(observed(None, status='unknown', reason='missing premise'), '')
    assert result.status == 'unknown' and result.answer is None and not result.quotes
    assert unknown_observation('invalid model JSON').status == 'unknown'
    with pytest.raises(ProbeValidationError):
        parse_observation(observed('guessed value', status='unknown', reason='uncertain'), '')
    with pytest.raises(ProbeValidationError):
        parse_observation(observed(None, status='unknown'), '')
    with pytest.raises(ProbeValidationError, match='requires source evidence'):
        parse_observation(observed('answer'), '')


def test_direct_verdicts_are_model_observations_not_string_comparison_rules():
    result = parse_observation(observed('supported', [{'text': 'A fact'}]),
                               'A fact is supplied.', mode='direct')
    assert result.answer == 'supported' and result.quote_membership_proves_entailment is False
    with pytest.raises(ProbeValidationError, match='supported or conflict'):
        parse_observation(observed('different entity', [{'text': 'A fact'}]),
                           'A fact is supplied.', mode='direct')


def test_source_quote_entailment_flag_cannot_be_overridden_by_model_json():
    raw = observed('answer', [{'text': 'Source'}])
    raw['quote_membership_proves_entailment'] = True
    with pytest.raises(ProbeValidationError, match='member set'):
        parse_observation(raw, 'Source')


def test_single_ascii_whitespace_controls_map_original_characters():
    response = '  Alpha  beta.\n'
    panel = leading_whitespace_variants(response)
    assert panel.original == response and panel.removed_prefix == ' '
    assert panel.original_population_modified is False
    variants = {v.name: v for v in panel.variants}
    trimmed = variants['remove_one_leading_ascii_space']
    assert trimmed.text == ' Alpha  beta.\n' and trimmed.removed_original == (0,)
    i = trimmed.text.index('beta')
    mapped = map_variant_span(trimmed, i, i + 4)
    j = response.index('beta')
    assert mapped.original_spans == ((j, j + 4),)
    assert mapped.inserted_characters == 0 and mapped.order_preserved
    assert len(variants['add_leading_ascii_space'].text) == len(response) + 1
    assert len(variants['add_interior_ascii_space'].text) == len(response) + 1
    assert len(variants['remove_one_interior_ascii_space'].text) == len(response) - 1
    assert variants['remove_one_interior_ascii_space'].removed_original == (response.index('  ', 2),)
    for variant in panel.variants:
        assert ''.join(variant.text.split()) == ''.join(response.split())
        assert variant.qualification_only is True
        assert variant.token_count_matched is None and variant.unit_format_matched is None
        assert variant.semantic_equivalence_verified is False
        assert map_variant_span(variant, 0, len(variant.text)).order_preserved
        for k, original in enumerate(variant.char_to_original):
            if original is not None:
                assert variant.text[k] == response[original]
    assert panel.token_count_matched is None and panel.unit_format_matched is None
    assert panel.semantic_equivalence_verified is False


def test_no_leading_space_uses_matched_insertions_and_reports_unaligned_event():
    response = 'Alpha beta.'
    panel = leading_whitespace_variants(response)
    variants = {v.name: v for v in panel.variants}
    removed = variants['remove_one_leading_ascii_space']
    assert removed.text == response and removed.removed_original == ()
    assert not removed.operation_applied and removed.unavailable_reason == 'no_leading_ascii_space'
    a, b = variants['add_leading_ascii_space'], variants['add_interior_ascii_space']
    assert len(a.text) == len(b.text) == len(response) + 1
    mapped = map_variant_span(a, 0, 1)
    assert mapped.original_spans == () and mapped.inserted_characters == 1
    assert map_variant_span(a, 1, 6).original_spans == ((0, 5),)


def test_unavailable_interior_control_is_explicit_not_a_content_edit():
    panel = leading_whitespace_variants(' Solo')
    assert panel.control_unavailable_reason == 'no_interior_ascii_gap'
    variants = {v.name: v for v in panel.variants}
    assert len(variants) == 5
    assert variants['add_leading_ascii_space'].operation_applied
    assert variants['remove_one_leading_ascii_space'].operation_applied
    for name in ['add_interior_ascii_space', 'remove_one_interior_ascii_space']:
        assert variants[name].text == panel.original
        assert variants[name].removed_original == ()
        assert not variants[name].operation_applied and variants[name].unavailable_reason
    with pytest.raises(ProbeValidationError):
        map_variant_span(panel.variants[1], 0, 99)


def test_nonascii_or_newline_prefix_is_never_stripped_by_ascii_intervention():
    for response in ['\u00a0\nAlpha beta.', '\nAlpha beta.', '\tAlpha beta.']:
        panel = leading_whitespace_variants(response)
        variants = {v.name: v for v in panel.variants}
        assert panel.removed_prefix == ''
        assert variants['remove_one_leading_ascii_space'].text == response
        assert variants['remove_one_leading_ascii_space'].removed_original == ()
        assert variants['add_leading_ascii_space'].text == ' ' + response


@pytest.mark.parametrize('corruption', ['bounds', 'order', 'duplicate', 'unhashable',
    'text', 'length', 'removed', 'operation', 'inserted_nonspace'])
def test_mapping_boundary_revalidates_forged_dataclasses(corruption):
    original, added = leading_whitespace_variants('Alpha beta.').variants[:2]
    if corruption == 'bounds':
        forged = replace(original, char_to_original=(99, *original.char_to_original[1:]))
    elif corruption == 'order':
        forged = replace(original, char_to_original=(1, 0, *original.char_to_original[2:]))
    elif corruption == 'duplicate':
        forged = replace(original, char_to_original=(0, 0, *original.char_to_original[2:]))
    elif corruption == 'unhashable':
        forged = replace(original, char_to_original=([], *original.char_to_original[1:]))
    elif corruption == 'text':
        forged = replace(original, text='X' + original.text[1:])
    elif corruption == 'length':
        forged = replace(original, original_length=0)
    elif corruption == 'removed':
        forged = replace(original, removed_original=(0,))
    elif corruption == 'operation':
        forged = replace(original, operation_applied=True)
    else:
        forged = replace(added, text='\n' + added.text[1:])
    with pytest.raises(ProbeValidationError):
        map_variant_span(forged, 0, 1)


def test_in_memory_contract_does_not_open_files_or_call_network(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Pure contract performed I/O')

    monkeypatch.setattr('builtins.open', forbidden)
    response = 'The period is annual.'
    _, frame, slot = proposed(response, 'annual')
    assert data(build_blind('The period is annual.', response, frame, slot))['role'] == 'coverage'
    assert leading_whitespace_variants(' Example response.').original_population_modified is False

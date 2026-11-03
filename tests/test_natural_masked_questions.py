"""Synthetic H012 string/provenance tests; no NLP, model, tokenizer or data."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from experiments.research import natural_masked_questions as kernel


def field(text, spans):
    return dict(text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(), characters=len(text),
                offset_unit=kernel.OFFSET_UNIT, status='processed', entities=[
                    dict(occurrence_id=i, text=text[a:b], type=label, start_char=a, end_char=b)
                    for i, (a, b, label) in enumerate(spans)])


def prepare(source='Bob Ada Ada', response='Ada met Bob. Ada left.', source_spans=None, response_spans=None):
    source_spans = source_spans if source_spans is not None else [(0, 3, 'PERSON'), (4, 7, 'PERSON'), (8, 11, 'PERSON')]
    response_spans = response_spans if response_spans is not None else [(0, 3, 'PERSON'), (8, 11, 'PERSON')]
    sample = dict(id='case', source=source, response=response)
    inventory = dict(id='case', source=field(source, source_spans), response=field(response, response_spans))
    return sample, inventory


def run(*args, **kwargs):
    sample, inventory = prepare(*args, **kwargs)
    return kernel.prepare_natural_questions([sample], [inventory])


def test_source_duplicates_canonical_raw_candidates_and_all_response_occurrences_are_retained():
    result = run()
    pool = next(iter(result['source_pools'].values()))
    assert pool['candidates'] == ['Ada', 'Bob']
    assert [e['occurrence_id'] for e in pool['occurrences']] == [0, 1, 2]
    assert pool['raw_occurrence_count'] == 3 and pool['unique_candidate_count'] == 2
    assert result['counts']['response_occurrences'] == len(result['proposals']) == 2
    target = result['proposals'][0]
    assert target['cloze_artifact']['masked_response'] == '[TARGET] met [OTHER_0001]. [TARGET] left.'
    assert target['target_literal_occurrences'] == 2
    assert target['target_readout']['original_value_exactly_in_source_pool'] is True
    assert result['originals'][0]['proposal_ids'] == [p['proposal_id'] for p in result['proposals']]


def test_source_pools_are_byte_identical_when_response_values_or_order_change():
    sample, inventory = prepare()
    first = kernel.prepare_natural_questions([sample], [inventory])
    sample['response'] = 'Zoe met Bob. Zoe left.'
    inventory['response'] = field(sample['response'], [(0, 3, 'PERSON'), (8, 11, 'PERSON')])
    changed_value = kernel.prepare_natural_questions([sample], [inventory])
    sample['response'] = 'Bob met Zoe. Zoe left.'
    inventory['response'] = field(sample['response'], [(0, 3, 'PERSON'), (8, 11, 'PERSON')])
    changed_order = kernel.prepare_natural_questions([sample], [inventory])
    assert json.dumps(first['source_pools'], sort_keys=True) == json.dumps(changed_value['source_pools'], sort_keys=True)
    assert first['source_pools'] == changed_order['source_pools']
    assert changed_value['proposals'][0]['target_readout']['original_value_exactly_in_source_pool'] is False
    assert changed_value['proposals'][0]['compiler_status'] == first['proposals'][0]['compiler_status'] == 'compiled'


def test_all_cases_source_pools_precede_any_response_entity_value_access():
    accesses = []
    class Traced(dict):
        def get(self, key, default=None):
            if key == 'text': accesses.append(self['side'])
            return super().get(key, default)
    samples, inventories = [], []
    for identifier in ('first', 'second'):
        sample, inventory = prepare()
        sample['id'] = inventory['id'] = identifier
        for side in ('source', 'response'):
            inventory[side]['entities'] = [Traced(e, side=side) for e in inventory[side]['entities']]
        samples.append(sample); inventories.append(inventory)
    result = kernel.prepare_natural_questions(samples, inventories)
    assert result['counts']['originals'] == 2
    assert max(i for i, side in enumerate(accesses) if side == 'source') < min(
        i for i, side in enumerate(accesses) if side == 'response')


def test_same_label_erased_other_type_context_retained_and_semantics_unknown():
    source = 'Ada Bob 2030'
    response = 'Ada met Bob in 2030.'
    result = run(source, response, [(0, 3, 'PERSON'), (4, 7, 'PERSON'), (8, 12, 'DATE')],
                 [(0, 3, 'PERSON'), (8, 11, 'PERSON'), (15, 19, 'DATE')])
    assert result['proposals'][0]['cloze_artifact']['masked_response'] == '[TARGET] met [OTHER_0001] in 2030.'
    assert result['proposals'][2]['cloze_artifact']['masked_response'] == 'Ada met Bob in [TARGET].'
    assert all(set(p['semantic_status'].values()) == {kernel.UNKNOWN} for p in result['proposals'])


def test_unicode_combining_marks_emoji_and_whitespace_exact_offsets():
    name = 'E\u0301ve'
    source, response = name + ' Bob', '😀 ' + name + '\tmet Bob.\n' + name + ' left.  '
    result = run(source, response, [(0, 4, 'PERSON'), (5, 8, 'PERSON')],
                 [(2, 6, 'PERSON'), (11, 14, 'PERSON')])
    proposal = result['proposals'][0]
    assert proposal['compiler_status'] == 'compiled'
    assert proposal['cloze_artifact']['masked_response'] == '😀 [TARGET]\tmet [OTHER_0001].\n[TARGET] left.  '
    assert all(response[m['start_char']:m['end_char']] == m['text'] for m in proposal['mask_spans'])


def test_raw_unicode_candidates_are_neither_normalized_nor_case_folded():
    source = 'Éve E\u0301ve eve'
    result = run(source, 'Éve.', [(0, 3, 'PERSON'), (4, 8, 'PERSON'), (9, 12, 'PERSON')], [(0, 3, 'PERSON')])
    assert next(iter(result['source_pools'].values()))['candidates'] == sorted(['Éve', 'E\u0301ve', 'eve'])


def test_duplicate_coordinates_merge_mask_provenance_without_target_dedup():
    result = run(response_spans=[(0, 3, 'PERSON'), (0, 3, 'PERSON'), (8, 11, 'PERSON')])
    assert len(result['proposals']) == 3
    a, b = result['proposals'][:2]
    assert a['proposal_id'] != b['proposal_id'] and a['cloze_artifact'] == b['cloze_artifact']
    assert len(a['mask_spans'][0]['provenance']) == 3
    assert a['mask_spans'][0]['replacement'] == '[TARGET]'


@pytest.mark.parametrize('kind', ['partial_ner', 'overlapping_literal', 'bad_bounds', 'bad_slice', 'empty', 'placeholder'])
def test_unsafe_masks_stay_explicit_unknown_and_occurrences_are_not_dropped(kind):
    sample, inventory = prepare()
    if kind == 'partial_ner': inventory['response'] = field(sample['response'], [(0, 3, 'PERSON'), (0, 7, 'PERSON')])
    elif kind == 'overlapping_literal':
        sample['response'] = 'aaaa'; inventory['response'] = field('aaaa', [(0, 3, 'PERSON')])
    elif kind == 'bad_bounds': inventory['response']['entities'][0]['end_char'] = 999
    elif kind == 'bad_slice': inventory['response']['entities'][0]['text'] = 'Zoe'
    elif kind == 'empty': inventory['response']['entities'][0]['text'] = ''
    else:
        sample['response'] += ' [OTHER_0001]'
        inventory['response'] = field(sample['response'], [(0, 3, 'PERSON'), (8, 11, 'PERSON')])
    result = kernel.prepare_natural_questions([sample], [inventory])
    assert result['counts']['response_occurrences'] == len(inventory['response']['entities'])
    assert result['proposals'][0]['compiler_status'] == 'unknown'
    assert result['proposals'][0]['cloze_artifact'] is None


def test_bad_source_inventory_never_builds_a_partial_candidate_pool():
    sample, inventory = prepare()
    inventory['source']['entities'][1]['end_char'] = 99
    result = kernel.prepare_natural_questions([sample], [inventory])
    pool = next(iter(result['source_pools'].values()))
    assert pool['compiler_status'] == 'unknown' and pool['candidates'] == []
    assert len(pool['occurrences']) == 3
    assert all(p['compiler_status'] == 'unknown' and p['target_readout']['original_value_exactly_in_source_pool'] is None
               for p in result['proposals'])


@pytest.mark.parametrize('source,spans,cardinality', [('', [], 'empty'), ('Bob', [(0, 3, 'PERSON')], 'single'),
    ('Bob Zoe', [(0, 3, 'PERSON'), (4, 7, 'PERSON')], 'multiple')])
def test_zero_single_multiple_source_pool_does_not_admit_qa_or_drop_target(source, spans, cardinality):
    result = run(source, 'Ada.', spans, [(0, 3, 'PERSON')])
    pool = next(iter(result['source_pools'].values()))
    assert pool['cardinality'] == cardinality
    assert result['proposals'][0]['compiler_status'] == 'compiled'
    assert result['proposals'][0]['qa_admitted'] is False and result['proposals'][0]['risk'] is None


def test_numeric_literal_in_fixed_marker_is_only_diagnostic_not_response_value_leakage():
    result = run('1 2', '1 + 2', [(0, 1, 'CARDINAL'), (2, 3, 'CARDINAL')],
                 [(0, 1, 'CARDINAL'), (4, 5, 'CARDINAL')])
    proposal = result['proposals'][0]
    assert proposal['compiler_status'] == 'compiled'
    assert proposal['cloze_artifact']['masked_response'] == '[TARGET] + [OTHER_0001]'
    assert proposal['diagnostics']['placeholder_constant_contains_target_literal'] is True
    assert kernel._mask_integrity('1 + 2', '1', proposal['cloze_artifact']['masked_response'], proposal['mask_spans']) is None


def test_unerased_original_literal_is_rejected_by_provenance_audit():
    mapping = dict(start_char=0, end_char=3, text='Ada', replacement='[TARGET]')
    assert kernel._mask_integrity('Ada + Ada', 'Ada', '[TARGET] + Ada', [mapping]) == 'unmasked_original_literal_occurrence'
    assert kernel._mask_integrity('Ada', 'Ada', '[TARGET] injected', [mapping]) == 'invalid_mask_provenance'


def test_preserves_case_order_empty_entities_attributes_and_no_input_mutation_or_qa_keys():
    sample, inventory = prepare()
    other, other_inventory = prepare('', '', [], [])
    other['id'] = other_inventory['id'] = 'empty'
    original = deepcopy((sample, inventory, other, other_inventory))
    result = kernel.prepare_natural_questions([SimpleNamespace(**sample), SimpleNamespace(**other)],
                                               {'case': inventory, 'empty': other_inventory})
    assert (sample, inventory, other, other_inventory) == original
    assert [row['id'] for row in result['originals']] == ['case', 'empty']
    assert result['originals'][1]['proposal_ids'] == []
    def audit(value):
        if isinstance(value, dict):
            assert not set(value) & {'question', 'system', 'messages', 'prompt', 'confidence', 'entailment'}
            if 'risk' in value: assert value['risk'] is None
            if 'qa_admitted' in value and type(value['qa_admitted']) is bool: assert value['qa_admitted'] is False
            for child in value.values(): audit(child)
        elif isinstance(value, list):
            for child in value: audit(child)
    audit(result)
    assert all(result['counts'][key] == 0 for key in ('model_calls', 'tokenizer_calls', 'nlp_calls'))
    json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False)


def test_population_or_duplicate_ids_are_not_silently_repaired():
    sample, inventory = prepare()
    with pytest.raises(ValueError, match='Unique'):
        kernel.prepare_natural_questions([sample, sample], [inventory, inventory])
    with pytest.raises(ValueError, match='population'):
        kernel.prepare_natural_questions([sample], [])

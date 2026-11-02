"""Logic-only H007 toy checks; no target data, predictions, or model loading."""
from collections import Counter
from copy import deepcopy
import importlib.util
from itertools import product
import json
from pathlib import Path
import re

import pytest

from experiments import data
from experiments.datasets import Sample
from experiments.research import focused_verdict as focus
from experiments.semantic_qualification import _context

_path = Path(__file__).with_name('generate_controls.py')
_spec = importlib.util.spec_from_file_location('h007_controls_generator', _path)
generator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(generator)


def packet_map():
    rows, records, truth = generator.build()
    return {(record['factor_cell']['relation_family'], record['factor_cell']['vocabulary'],
             record['factor_cell']['source_direction'], record['factor_cell']['response_direction'],
             record['factor_cell']['distractor_count']): (row, record, label)
            for row, record, label in zip(rows, records, truth)}


def test_complete_48_cell_factorial_and_balanced_truth():
    rows, records, truth = generator.build()
    expected = set(product(('winner', 'sole_signer'), ('lex1', 'lex2'), (0, 1), (0, 1), (0, 4, 12)))
    assert len(rows) == len(records) == len(truth) == 48 and set(packet_map()) == expected
    assert len({row['id'] for row in rows}) == 48
    assert Counter(label['target_outcome'] for label in truth) == {'supported': 24, 'contradicted': 24}


def test_truth_matches_independently_parsed_unique_source_bindings():
    for cell, (row, record, label) in packet_map().items():
        family = cell[0]
        suffix = 'winner of the (.+?) final' if family == 'winner' else 'signer of the (.+?)'
        facts = dict((event, person) for person, event in re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+) was the sole ' + suffix + r'\.', row['source']))
        assert len(facts) == 2
        target = record['target_text'].strip()
        pattern = (r'(.+?) won the (.+?) final, and (.+?) won the (.+?) final\.' if family == 'winner' else
                   r'(.+?) was the sole signer of the (.+?), and (.+?) was the sole signer of the (.+?)\.')
        a, x, b, y = re.fullmatch(pattern, target).groups()
        claims = {x: a, y: b}
        outcomes = [facts[event] == person for event, person in claims.items()]
        assert outcomes == [True, True] if label['target_outcome'] == 'supported' else outcomes == [False, False]
        assert label['whole_outcome'] == label['target_outcome']


@pytest.mark.parametrize('family', ['winner', 'sole_signer'])
def test_source_flip_preserves_every_word_count(family):
    cases = packet_map()
    for vocabulary, response_direction, burden in product(('lex1', 'lex2'), (0, 1), (0, 4, 12)):
        a = cases[family, vocabulary, 0, response_direction, burden][0]
        b = cases[family, vocabulary, 1, response_direction, burden][0]
        assert a['source'] != b['source'] and generator.bag(a['source']) == generator.bag(b['source'])
        assert a['response'] == b['response']


@pytest.mark.parametrize('family', ['winner', 'sole_signer'])
def test_response_flip_preserves_every_word_count(family):
    cases = packet_map()
    for vocabulary, source_direction, burden in product(('lex1', 'lex2'), (0, 1), (0, 4, 12)):
        a = cases[family, vocabulary, source_direction, 0, burden][0]
        b = cases[family, vocabulary, source_direction, 1, burden][0]
        assert a['source'] == b['source'] and a['response'] != b['response']
        assert generator.bag(a['response']) == generator.bag(b['response'])


def test_increased_answer_burden_never_changes_source_prompt_or_target_bytes():
    cases = packet_map()
    for family, vocabulary, source_direction, response_direction in product(
            generator.FAMILIES, generator.VOCABULARIES, (0, 1), (0, 1)):
        packets = [cases[family, vocabulary, source_direction, response_direction, burden] for burden in (0, 4, 12)]
        assert len({row['source'] for row, _, _ in packets}) == 1
        assert len({row['prompt'] for row, _, _ in packets}) == 1
        assert len({record['target_text'] for _, record, _ in packets}) == 1
        assert len({tuple(record['target_span']) for _, record, _ in packets}) == 1


def test_every_added_filler_is_explicitly_supported_and_target_unrelated():
    assert len(set(generator.FILLERS)) == 12
    for cell, (row, record, label) in packet_map().items():
        case = generator.SCENARIOS[cell[:2]]
        assert all(row['source'].count(filler) == 1 for filler in generator.FILLERS)
        assert all(name not in filler for name in (*case['people'], *case['events']) for filler in generator.FILLERS)
        assert row['response'][record['target_span'][1]:] == ' '.join(generator.FILLERS[:cell[-1]])
        assert label['filler_outcomes'] == ['supported'] * cell[-1]


def test_target_span_is_exactly_existing_first_focus_unit_at_all_burdens():
    for cell, (row, record, _) in packet_map().items():
        units = focus.response_units(row['response'])
        assert len(units) == cell[-1] + 1
        assert units[0] == dict(start=0, end=record['target_span'][1], text=record['target_text'])
        assert record['target_text'].endswith('. ') and row['response'][:record['target_span'][1]] == record['target_text']
        assert ''.join(unit['text'] for unit in units) == row['response']


def test_all_rows_are_inference_only_sample_inputs_with_eight_source_ids():
    rows, records, _ = generator.build()
    assert len({row['source_id'] for row in rows}) == 8
    allowed = {'id', 'source_id', 'split', 'task', 'prompt', 'source', 'response', 'generator', 'evidence_path'}
    for row in rows:
        assert set(row) == allowed and Sample(**row).inputs['source'] == row['source']
        assert 'factor_cell' not in row and 'outcome' not in row and 'label' not in row
    assert all(record['metadata_sent_to_model'] is False for record in records)


def test_factual_text_has_no_pronouns_digits_or_arithmetic_symbols():
    forbidden = re.compile(r'\b(he|she|it|they|him|her|them|his|their|this|that|these|those|who|which)\b', re.I)
    for row in generator.build()[0]:
        assert not forbidden.search(row['source'] + ' ' + row['response'])
        assert not re.search(r'[0-9+*/=]', row['source'] + row['response'])


def test_every_winner_match_has_two_distinct_full_named_contestants():
    for cell, (row, _, _) in packet_map().items():
        if cell[0] != 'winner':
            continue
        matches = re.findall(r'The (.+?) final was a match between (.+?) and (.+?)\.', row['source'])
        assert len(matches) == 2
        assert all(a != b and a.count(' ') == b.count(' ') == 1 for _, a, b in matches)
        assert len({event for event, _, _ in matches}) == 2


def test_existing_focus_builder_gives_576_logical_and_480_distinct_requests():
    unique, logical = set(), 0
    for cell, (row, record, _) in packet_map().items():
        context = _context(Sample(**row).inputs)[0]
        for assignment in focus.label_assignments():
            whole = focus.build_request(row['source'], row['response'], context, record['whole_span'], assignment)
            target = focus.build_request(row['source'], row['response'], context, record['target_span'], assignment)
            assert (whole == target) is (cell[-1] == 0)
            for request in (whole, target):
                logical += 1
                unique.add(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(',', ':')))
    assert logical == 576 and len(unique) == 480


def test_generation_seals_inference_and_truth_in_separate_directories(tmp_path):
    root = tmp_path / 'outputs' / 'h007-toy'
    receipt = generator.generate(root)
    data.verify_seal(root / 'inference', expected={'inputs.jsonl', 'plan.json'})
    data.verify_seal(root / 'construction', expected={'construction_truth.jsonl'})
    assert receipt['checks']['original_cases'] == 48 and receipt['model_calls'] == 0
    assert receipt['natural_datasets_read'] is receipt['natural_predictions_read'] is False
    assert 'construction_truth' not in (root / 'inference' / 'seal.json').read_text(encoding='utf-8')


def test_generation_refuses_existing_material_without_overwrite(tmp_path):
    root = tmp_path / 'outputs' / 'h007-toy'
    generator.generate(root)
    before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}
    with pytest.raises(FileExistsError):
        generator.generate(root)
    assert before == {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}


def test_validation_rejects_an_individually_mismatched_plan_identity():
    rows, records, truth = deepcopy(generator.build())
    records[0]['id'] = 'wrong-original'
    with pytest.raises(ValueError, match='identity'):
        generator.validate(rows, records, truth)

"""Deterministic H007 construction only. No datasets, models, or scoring imports."""
from collections import Counter
from datetime import datetime, timezone
from itertools import product
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
FAMILIES = ('winner', 'sole_signer')
VOCABULARIES = ('lex1', 'lex2')
BURDENS = (0, 4, 12)
SCENARIOS = {
    ('winner', 'lex1'): dict(people=('Mira Tanswick', 'Oren Pellford'), events=('Virelan Cup', 'Nerowan Cup')),
    ('winner', 'lex2'): dict(people=('Sana Whitmere', 'Evan Calderby'), events=('Raventon Cup', 'Dalespar Cup')),
    ('sole_signer', 'lex1'): dict(people=('Lena Cresswell', 'Tomas Brindle'), events=('Quorren Accord', 'Meliston Accord')),
    ('sole_signer', 'lex2'): dict(people=('Nadia Fennbrook', 'Hugo Valecrest'), events=('Corvellan Accord', 'Tremorin Accord')),
}
FILLERS = (
    'The Lenwick Gallery displayed a bronze compass.',
    'The Haverill Library stored a green atlas.',
    'The Torsen Workshop repaired a copper lantern.',
    'The Belmonte Cafe served a barley biscuit.',
    'The Darswick Museum exhibited a violet vase.',
    'The Elloway Theatre presented a chamber recital.',
    'The Farnell Bakery offered a sesame loaf.',
    'The Grensley Observatory housed a silver telescope.',
    'The Halbrook Conservatory grew a cedar sapling.',
    'The Iverton Archive catalogued a linen banner.',
    'The Jasmere Studio painted a saffron mural.',
    'The Kelston Depot stored a wicker basket.',
)


def sha_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bag(text):
    return Counter(re.findall(r"[A-Za-z]+", text.lower()))


def source_facts(family, vocabulary, direction):
    case = SCENARIOS[family, vocabulary]
    people, events = case['people'], case['events']
    assigned = people if direction == 0 else people[::-1]
    if family == 'winner':
        return [sentence for event, person in zip(events, assigned) for sentence in (
            f'The {event} final was a match between {people[0]} and {people[1]}.',
            f'{person} was the sole winner of the {event} final.')]
    return [f'{person} was the sole signer of the {event}.' for event, person in zip(events, assigned)]


def target_sentence(family, vocabulary, direction):
    case = SCENARIOS[family, vocabulary]
    a, b = case['people'] if direction == 0 else case['people'][::-1]
    x, y = case['events']
    if family == 'winner':
        return f'{a} won the {x} final, and {b} won the {y} final.'
    return f'{a} was the sole signer of the {x}, and {b} was the sole signer of the {y}.'


def build():
    rows, records, truth = [], [], []
    for family, vocabulary, source_direction, response_direction, burden in product(
            FAMILIES, VOCABULARIES, (0, 1), (0, 1), BURDENS):
        scenario = f'{family}-{vocabulary}'
        source_id = f'h007-{scenario}-s{source_direction}'
        sample_id = f'{source_id}-r{response_direction}-b{burden:02d}'
        facts = source_facts(family, vocabulary, source_direction)
        source = ' '.join([*facts, *FILLERS])
        target = target_sentence(family, vocabulary, response_direction) + ' '
        response = target + ' '.join(FILLERS[:burden])
        prompt = 'Use the source passage as the only factual evidence.\n\n' + source + '\n\nWrite a factual response.'
        rows.append(dict(id=sample_id, source_id=source_id, split='test', task='QA',
            prompt=prompt, source=source, response=response, generator='rule-h007-fixed-controls-v1', evidence_path=''))
        records.append(dict(id=sample_id, source_id=source_id, scenario=scenario,
            factor_cell=dict(relation_family=family, vocabulary=vocabulary,
                source_direction=source_direction, response_direction=response_direction, distractor_count=burden),
            target_span=[0, len(target)], target_text=target, whole_span=[0, len(response)],
            input_text_sha256={key: sha_text(value) for key, value in
                               dict(prompt=prompt, source=source, response=response).items()},
            source_target_fact_count=len(facts), source_filler_count=12,
            response_filler_count=burden, target_position='first', metadata_sent_to_model=False))
        outcome = 'supported' if source_direction == response_direction else 'contradicted'
        truth.append(dict(id=sample_id, source_id=source_id,
            source_sha256=sha_text(source), response_sha256=sha_text(response),
            target_span=[0, len(target)], target_outcome=outcome, whole_outcome=outcome,
            filler_outcomes=['supported'] * burden,
            construction_rule='entity_event_bijections_equal_iff_source_direction_equals_response_direction',
            target_benchmark_labels_used=False, intended_use='post_observation_qualification_only'))
    return rows, records, truth


def validate(rows, records, truth):
    expected = set(product(FAMILIES, VOCABULARIES, (0, 1), (0, 1), BURDENS))
    cells = {(r['factor_cell']['relation_family'], r['factor_cell']['vocabulary'],
              r['factor_cell']['source_direction'], r['factor_cell']['response_direction'],
              r['factor_cell']['distractor_count']) for r in records}
    if len(rows) != 48 or len(records) != 48 or len(truth) != 48 or cells != expected:
        raise ValueError('The complete 48-cell factorial must be retained')
    if (len({r['id'] for r in rows}) != 48 or [r['id'] for r in rows] != [r['id'] for r in records]
            or [r['id'] for r in rows] != [r['id'] for r in truth]):
        raise ValueError('Unique original row/order identity required')
    grouped = {}
    for row, record, label in zip(rows, records, truth):
        cell = record['factor_cell']
        family, vocabulary = cell['relation_family'], cell['vocabulary']
        direction, burden = cell['response_direction'], cell['distractor_count']
        target = target_sentence(family, vocabulary, direction) + ' '
        expected_source = ' '.join([*source_facts(family, vocabulary, cell['source_direction']), *FILLERS])
        if (row['source'] != expected_source or row['response'] != target + ' '.join(FILLERS[:burden])
                or record['target_span'] != [0, len(target)] or record['target_text'] != target
                or row['prompt'].count(row['source']) != 1 or row['response'][:len(target)] != target
                or label['id'] != row['id'] or label['source_sha256'] != sha_text(row['source'])
                or label['response_sha256'] != sha_text(row['response'])):
            raise ValueError('Exact source/response/target/input identity changed')
        people, events = SCENARIOS[family, vocabulary]['people'], SCENARIOS[family, vocabulary]['events']
        if len(set(people)) != 2 or len(set(events)) != 2 or any(name in sentence for name in (*people, *events) for sentence in FILLERS):
            raise ValueError('Distinct target entities/events and unrelated filler required')
        key = family, vocabulary, cell['source_direction'], direction
        grouped.setdefault(key, []).append(row)
        expected_outcome = 'supported' if cell['source_direction'] == direction else 'contradicted'
        if (label['target_outcome'] != expected_outcome or label['whole_outcome'] != expected_outcome
                or label['filler_outcomes'] != ['supported'] * burden):
            raise ValueError('Construction outcome changed')
    if any(len({row['source'] for row in group}) != 1 for group in grouped.values()):
        raise ValueError('Answer burden must never change the source')
    return dict(original_cases=48, scenarios=4, unique_sources=8, source_facts_always_include_all_fillers=12,
                burdens=list(BURDENS), target_position='first', model_calls=0, target_benchmark_labels_used=False)


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def write_jsonl(path, rows):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':'), allow_nan=False) + '\n')


def generate(root=HERE):
    root = Path(root).resolve()
    if 'outputs' not in root.parts:
        raise ValueError('Retain H007 artifacts under outputs/')
    if any((root / name).exists() for name in ('inference', 'construction', 'generation-receipt.json')):
        raise FileExistsError('Existing generated material is immutable; do not retry or overwrite')
    rows, records, truth = build()
    checks = validate(rows, records, truth)
    root.mkdir(parents=True, exist_ok=True)
    inference, construction = root / 'inference', root / 'construction'
    inference.mkdir(); construction.mkdir()
    write_jsonl(inference / 'inputs.jsonl', rows)
    plan = dict(schema='h007-burden-controls-plan-v1', state='frozen_before_model_observation',
        factors=dict(relation_family=list(FAMILIES), vocabulary=list(VOCABULARIES), source_direction=[0, 1],
                     response_direction=[0, 1], distractor_count=list(BURDENS)),
        expected_original_cases=48, logical_observation_cases=96, logical_label_requests=576,
        expected_distinct_requests_before_tokenization=480, modes=['whole', 'target'], permutations=6,
        construction_truth_not_for_observer=True, metadata_ids_or_factors_sent_to_model=False,
        source_policy='all_twelve_fillers_fixed_within_scenario_and_source_direction',
        target_boundary_policy='first_sentence_plus_one_fixed_trailing_space', records=records)
    write_json(inference / 'plan.json', plan)
    write_json(inference / 'seal.json', {name: file_hash(inference / name) for name in ('inputs.jsonl', 'plan.json')})
    write_jsonl(construction / 'construction_truth.jsonl', truth)
    write_json(construction / 'seal.json', {'construction_truth.jsonl': file_hash(construction / 'construction_truth.jsonl')})
    source_files = [Path(__file__).resolve(), HERE / 'DESIGN.md', HERE / 'test_generate_controls.py']
    receipt = dict(schema='h007-construction-generation-receipt-v1', generated_utc=datetime.now(timezone.utc).isoformat(),
        model_calls=0, natural_datasets_read=False, natural_predictions_read=False, target_benchmark_labels_used=False,
        checks=checks, generator_sources={path.name: file_hash(path) for path in source_files},
        inference_seal_sha256=file_hash(inference / 'seal.json'),
        construction_seal_sha256=file_hash(construction / 'seal.json'),
        input_rows_sha256=file_hash(inference / 'inputs.jsonl'),
        next_step='Root freezes existing shared verdict collector + focus whole/target requests; no scoring was performed.')
    write_json(root / 'generation-receipt.json', receipt)
    return receipt


if __name__ == '__main__':
    print(json.dumps(generate(), ensure_ascii=False, indent=2))

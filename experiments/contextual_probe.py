"""Frozen crossed-role qualification, with separate input, score and evaluation stages."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

import numpy as np

from experiments import data
from experiments.provenance import completed_identity, research_identity
from experiments.research.contextual_alignment import (
    WORDS, anchored_residual, qualify_family, word_states,
)


TEMPLATES = [
    ('budget', ['8', '2'], ['The event budget totals {x} million dollars',
      'The competition prizes total {y} million dollars'],
     ['The event budget amounts to {x} million dollars',
      'The competition prizes amount to {y} million dollars'], 'The banner is green.', ['banner', 'green']),
    ('parcel', ['violet', 'amber'], ['Package Orion carries {x}', 'Package Vega carries {y}'],
     ['Package Orion contains {x}', 'Package Vega contains {y}'], 'The seal is square.', ['seal', 'square']),
    ('mode', ['7', '4'], ['Winter mode consumes {x} watts', 'Summer mode consumes {y} watts'],
     ['Winter mode requires {x} watts', 'Summer mode requires {y} watts'],
     'The casing is silver.', ['casing', 'silver']),
    ('reservoir', ['12', '30'], ['The north reservoir holds {x} litres',
      'The south reservoir holds {y} litres'],
     ['The north reservoir stores {x} litres', 'The south reservoir stores {y} litres'],
     'The marker is round.', ['marker', 'round']),
    ('launch', ['Tuesday', 'Friday'], ['The Atlas mission launches {x}',
      'The Boreal mission launches {y}'],
     ['The Atlas launch occurs on {x}', 'The Boreal launch occurs on {y}'],
     'The flag is orange.', ['flag', 'orange']),
    ('office', ['Oslo', 'Lima'], ['The analyst works in {x}', 'The designer works in {y}'],
     ['The analyst is stationed in {x}', 'The designer is stationed in {y}'],
     'The notice is yellow.', ['notice', 'yellow']),
    ('entry', ['seven', 'nine'], ['Morning entry opens at {x}', 'Evening entry opens at {y}'],
     ['Morning entry starts at {x}', 'Evening entry starts at {y}'],
     'The sign is blue.', ['sign', 'blue']),
    ('author', ['Nora', 'Zane'], ['The novel was written by {x}', 'The essay was written by {y}'],
     ['The novel credits {x} as its author', 'The essay credits {y} as its author'],
     'The bookmark is red.', ['bookmark', 'red']),
]


def read_rows(path):
    return [json.loads(line) for line in Path(path).open()]


def write_rows(path, rows):
    with Path(path).open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def sentences(clauses, values, reverse, combined):
    x, y = values
    clauses = [c.format(x=x, y=y) for c in clauses]
    if reverse:
        clauses.reverse()
    if combined:
        second = clauses[1]
        if second.startswith('The '):
            second = 'the ' + second[4:]
        return clauses[0] + ', whereas ' + second + '.'
    return '. '.join(clauses) + '.'


def prepare(output, cache=None):
    output.mkdir(parents=True, exist_ok=False)
    inputs, annotations = [], []
    for family, values, sources, answers, constant, collateral in TEMPLATES:
        for rendering in ['primary', 'source_reordered', 'answer_reordered']:
            source_texts, answer_texts = [], []
            for assignment in [0, 1]:
                pair = values if assignment == 0 else values[::-1]
                source_texts.append(sentences(sources, pair, rendering == 'source_reordered', False) + ' ' + constant)
                answer_texts.append(sentences(answers, pair, rendering != 'answer_reordered', True) + ' ' + constant)
            for texts in [source_texts, answer_texts]:
                bags = [Counter(m.group().casefold() for m in WORDS.finditer(t)) for t in texts]
                if bags[0] != bags[1]:
                    raise ValueError('Crossed assignments changed the lexical multiset')
            for a in [0, 1]:
                for b in [0, 1]:
                    identity = f'{family}-{rendering}-{a}{b}'
                    inputs.append({'id': identity, 'source': source_texts[a], 'answer': answer_texts[b]})
                    annotations.append({'id': identity, 'family': family, 'rendering': rendering,
                                        'source_assignment': a, 'answer_assignment': b,
                                        'supported': a == b, 'targets': [v.casefold() for v in values],
                                        'collateral': collateral})
    # Predeclared causal-scope stress: role follows each monetary value.
    for a in [0, 1]:
        x, y = ['8', '2'] if a == 0 else ['2', '8']
        source = f'The organiser allocated {x} million dollars to the budget and {y} million dollars to prizes. The banner is green.'
        for b in [0, 1]:
            identity = f'budget-role_after-{a}{b}'
            answer = next(r['answer'] for r in inputs if r['id'] == f'budget-primary-0{b}')
            inputs.append({'id': identity, 'source': source, 'answer': answer})
            annotations.append({'id': identity, 'family': 'budget', 'rendering': 'role_after',
                                'source_assignment': a, 'answer_assignment': b, 'supported': a == b,
                                'targets': ['8', '2'], 'collateral': ['banner', 'green']})
    for index, answer in enumerate(['No opening date is provided.', 'I do not know the opening date.']):
        identity = f'refusal-{index}'
        inputs.append({'id': identity, 'source': 'The venue is closed. Its banner is green.', 'answer': answer})
        annotations.append({'id': identity, 'family': 'refusal', 'rendering': 'coverage_only',
                            'supported': True, 'targets': [], 'collateral': []})
    if cache is not None:
        manifest = data.read_json(cache / 'manifest.json')
        sources = {str(r['source_id']): r for r in read_rows(Path(manifest['dataset']) / 'source_info.jsonl')}
        for identifier in ['3180', '3182']:
            record = next(r for r in manifest['records'] if r['id'] == identifier)
            response = data.read_json(cache / record['directory'] / 'response.json')
            source = sources[record['source_id']]
            inputs.append({'id': f'natural-{identifier}', 'source': str(source['source_info']),
                           'answer': response['text'], 'prompt': source['prompt']})
            annotations.append({'id': f'natural-{identifier}', 'family': 'known_natural',
                                'rendering': 'coverage_only', 'targets': ['prizes'], 'collateral': []})
    write_rows(output / 'inputs.jsonl', inputs)
    write_rows(output / 'annotations.jsonl', annotations)
    data.write_json(output / 'plan.json', {
        'scope': 'frozen contextual observation qualification; not a complete detector',
        'cases': len(inputs), 'semantic_families': 8, 'minimum_other_anchors': 4,
        'primary': 'leave-one-out residual; wrong assignment should have higher cost',
        'layers': 'final post-norm only; no search',
        'baselines': ['unpooled cosine mismatch', 'matched full-history source gap'],
        'gate': 'all 8 family contrasts positive; all wrong target costs exceed all supported '
                'renderings; unchanged-fact crossed contrasts smaller than target contrasts',
        'inputs_sha256': data.file_hash(output / 'inputs.jsonl'),
        'annotations_sha256': data.file_hash(output / 'annotations.jsonl'),
        'provenance': research_identity(),
    })
    print('PREPARED', len(inputs), flush=True)


def collect(root, model_path):
    import torch
    from transformers import AutoTokenizer
    from myproject.model import load_model
    from experiments.hidden import hidden_observation

    plan = data.read_json(root / 'plan.json')
    if data.file_hash(root / 'inputs.jsonl') != plan['inputs_sha256']:
        raise ValueError('Prepared inputs changed')
    identity = research_identity()
    output = root / 'states'
    output.mkdir(exist_ok=False)
    model_files = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                   'special_tokens_map.json', 'model.safetensors.index.json']
    model_hashes = {name: data.file_hash(model_path / name) for name in model_files}
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = load_model(model_path, 'cuda:0')
    runtime = {'dtype': str(model.dtype), 'attention': model.config._attn_implementation,
               'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
               'chat_template_sha256': data.text_hash(tokenizer.chat_template)}
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    records = []
    for sample in read_rows(root / 'inputs.jsonl'):
        observed = hidden_observation(model, tokenizer, sample['source'], sample['answer'], sample.get('prompt'))
        file = output / f'{sample["id"]}.npz'
        np.savez_compressed(file, **observed)
        records.append({'id': sample['id'], 'file': str(file.relative_to(root)),
                        'sha256': data.file_hash(file), 'answer_tokens': len(observed['answer_ids']),
                        'source_tokens': len(observed['source_ids']),
                        'processed_tokens': (observed['prompt_tokens'] + observed['absent_prompt_tokens']
                                             + 2 * len(observed['answer_ids']))})
        print('COLLECTED', sample['id'], len(records), flush=True)
    torch.cuda.synchronize()
    data.write_json(root / 'collection.json', {
        'records': records, 'backbone_forwards': 2 * len(records),
        'seconds_after_model_load_including_io': time.monotonic() - start,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
        'model': str(model_path.resolve()),
        'model_files_sha256': model_hashes, 'runtime': runtime,
        'weight_hash_scope': 'index only; not a full weight-content audit',
        'inputs_sha256': plan['inputs_sha256'], 'provenance': completed_identity(identity),
        'annotation_reads': 0,
    })


def score(root):
    identity = research_identity()
    inputs = {r['id']: r for r in read_rows(root / 'inputs.jsonl')}
    collection = data.read_json(root / 'collection.json')
    if data.file_hash(root / 'inputs.jsonl') != collection['inputs_sha256']:
        raise ValueError('Collection inputs changed')
    results = []
    for record in collection['records']:
        file = root / record['file']
        if data.file_hash(file) != record['sha256']:
            raise ValueError('Observed states changed')
        sample = inputs[record['id']]
        with np.load(file, allow_pickle=False) as observed:
            source = word_states(sample['source'], observed['source_offsets'], observed['source_hidden'])
            answer = word_states(sample['answer'], observed['answer_offsets'], observed['answer_hidden'])
            costs = anchored_residual(source, answer, [(0, len(sample['answer']))])
            for word in costs:
                word['gap'] = float(observed['source_gap'][word['tokens']].mean())
        results.append({'id': record['id'], 'answer_words': len(answer), 'anchors': costs})
    data.write_json(root / 'scores.json', {'cases': results, 'annotation_reads': 0,
                    'provenance': completed_identity(identity)})
    data.write_json(root / 'scores.seal.json', {
        name: data.file_hash(root / name) for name in ['scores.json', 'inputs.jsonl', 'collection.json']})


def evaluate(root):
    seal = data.read_json(root / 'scores.seal.json')
    for name, expected in seal.items():
        if data.file_hash(root / name) != expected:
            raise ValueError('Score seal mismatch')
    if data.file_hash(root / 'annotations.jsonl') != data.read_json(root / 'plan.json')['annotations_sha256']:
        raise ValueError('Construction annotations changed')
    annotations = {r['id']: r for r in read_rows(root / 'annotations.jsonl')}
    cells, coverage = defaultdict(list), []
    for case in data.read_json(root / 'scores.json')['cases']:
        annotation = annotations[case['id']]
        anchors = {r['word']: r for r in case['anchors'] if r['available']}
        targets = [anchors[t] for t in annotation['targets'] if t in anchors]
        collateral = [anchors[t] for t in annotation['collateral'] if t in anchors]
        coverage.append({'id': case['id'], 'words': case['answer_words'], 'available_anchors': len(anchors),
                         'target_coverage': len(targets), 'targets_requested': len(annotation['targets'])})
        if annotation['rendering'] == 'coverage_only':
            continue
        if len(targets) != 2 or not collateral:
            raise ValueError(f'Controlled target coverage failed: {case["id"]}')
        cell = {'id': case['id'], 'supported': annotation['supported'],
                'a': annotation['source_assignment'], 'b': annotation['answer_assignment']}
        for readout in ['residual', 'cosine', 'gap']:
            cell[readout] = float(np.mean([r[readout] for r in targets]))
            cell[f'{readout}_collateral'] = float(np.mean([r[readout] for r in collateral]))
        cell['target_costs'] = {r['word']: {k: r[k] for k in ['residual', 'cosine', 'gap']}
                                for r in targets}
        cells[(annotation['family'], annotation['rendering'])].append(cell)
    families = {}
    for family in [t[0] for t in TEMPLATES]:
        families[family] = qualify_family({rendering: cells[(family, rendering)]
            for rendering in ['primary', 'source_reordered', 'answer_reordered']})
    result = {'scope': 'controlled lexical-anchor observation qualification, not natural token AUROC',
              'families': families, 'coverage': coverage,
              'cells': {f'{f}/{r}': rows for (f, r), rows in cells.items()},
              'all_families_qualified': {k: all(v[k]['qualified'] for v in families.values())
                                         for k in ['residual', 'cosine', 'gap']}}
    data.write_json(root / 'evaluation.json', result)
    print('QUALIFICATION', result['all_families_qualified'], flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'collect', 'score', 'evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    parser.add_argument('--cache', type=Path)
    args = parser.parse_args()
    from experiments.research.run import output_path
    args.output = output_path(args.output, Path.cwd())
    if args.stage == 'prepare':
        prepare(args.output, args.cache)
    elif args.stage == 'collect':
        if args.model is None:
            parser.error('collect requires --model')
        collect(args.output, args.model)
    elif args.stage == 'score':
        score(args.output)
    else:
        evaluate(args.output)


if __name__ == '__main__':
    main()

"""Bounded lexical-identity qualification of within-passage deletion fingerprints."""

import argparse
from collections import Counter
from pathlib import Path
import time

import numpy as np

from experiments import data
from experiments.provenance import completed_identity, research_identity
from experiments.research.contextual_alignment import WORDS
from experiments.research.fingerprints import mask_preserving_permutation, signed_profile_distance


# Frozen before extraction, independent of the previous contextual probe's families.
FAMILIES = [
    ('frequency', ['oboe', 'flute'], ['440', '523'],
     'The instruments are {r1} and {r2}. Their frequencies, in that order, are {v1} and {v2} hertz.',
     'For {r1} followed by {r2}, the frequency assignments are respectively {v1} and {v2} hertz.',
     'The enclosure is ceramic.', ['enclosure', 'ceramic']),
    ('decision', ['copper', 'silver'], ['approved', 'rejected'],
     'The proposals are {r1} and {r2}. Their decisions, in that order, are {v1} and {v2}.',
     'For {r1} followed by {r2}, the recorded decisions are respectively {v1} and {v2}.',
     'The folder is striped.', ['folder', 'striped']),
    ('phase', ['preflight', 'postflight'], ['active', 'idle'],
     'The phases are {r1} and {r2}. Their states, in that order, are {v1} and {v2}.',
     'For {r1} followed by {r2}, the logged states are respectively {v1} and {v2}.',
     'The panel is wooden.', ['panel', 'wooden']),
    ('transfer', ['sender', 'receiver'], ['Mira', 'Theo'],
     'The roles are {r1} and {r2}. Their names, in that order, are {v1} and {v2}.',
     'For {r1} followed by {r2}, the participant names are respectively {v1} and {v2}.',
     'The receipt is triangular.', ['receipt', 'triangular']),
]
ABSENCE = [
    ('crate', 'A sealed crate rests in the vestibule. No mass is listed.',
     ['The crate mass is not listed.', 'The crate mass is 14 kilograms.']),
    ('rehearsal', 'A rehearsal took place in the atrium. No duration is recorded.',
     ['The rehearsal duration is not recorded.', 'The rehearsal duration is 19 minutes.']),
]


def prepare(root, model_path):
    from transformers import AutoTokenizer
    from experiments.conditional_dependencies import lexical_anchors
    from myproject.inputs import encode_response

    root.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    passages, groups, annotations = {}, [], []

    def register(text):
        identity = data.text_hash(text)
        encoded = encode_response(tokenizer, text)
        passages[identity] = {'id': identity, 'text': text,
            'anchors': lexical_anchors(text, encoded['offsets']),
            'token_ids': encoded['answer_ids']}
        return identity

    def group(name, sources, answers):
        records = [passages[p] for p in sources + answers]
        words = sorted(set.intersection(*[{w['word'] for w in r['anchors']} for r in records]))
        masks = []
        for record in records:
            anchors = {w['word']: w for w in record['anchors']}
            masks.append([[anchors[u]['tokens'][-1] < anchors[v]['tokens'][0]
                           for v in words] for u in words])
        mask = np.logical_and.reduce(masks)
        permutation = mask_preserving_permutation(mask)
        groups.append({'name': name, 'sources': sources, 'answers': answers,
                       'words': words, 'mask': mask.tolist(), 'permutation': permutation.tolist()})
        return groups[-1]

    for name, roles, values, source_template, answer_template, constant, collateral in FAMILIES:
        variants = []
        for template in [source_template, answer_template]:
            records = []
            for assignment in [0, 1]:
                for order in [0, 1]:
                    assigned = values if assignment == 0 else values[::-1]
                    r, v = (roles, assigned) if order == 0 else (roles[::-1], assigned[::-1])
                    text = template.format(r1=r[0], r2=r[1], v1=v[0], v2=v[1])
                    records.append((register(text + ' ' + constant + ' This text is a brief note.'),
                                    assignment, order))
            bags = [Counter(m.group().casefold() for m in WORDS.finditer(passages[p]['text']))
                    for p, _, _ in records]
            if any(b != bags[0] for b in bags):
                raise ValueError('Assignment/order worlds must retain the lexical bag')
            variants.append(records)
        sources, answers = variants
        frozen = group(name, [r[0] for r in sources], [r[0] for r in answers])
        for value in values:
            index = frozen['words'].index(value.casefold())
            if np.asarray(frozen['mask'])[:, index].sum() < 2:
                raise ValueError('Constructed target lacks two common predecessors')
        for s, a, so in sources:
            for y, b, ao in answers:
                annotations.append({'source': s, 'answer': y, 'family': name,
                    'a': a, 'b': b, 'source_order': so, 'answer_order': ao, 'supported': a == b,
                    'targets': [v.casefold() for v in values],
                    'collateral': collateral + ['text', 'brief', 'note']})
    for name, source, answers in ABSENCE:
        group(name, [register(source)], [register(y) for y in answers])
    data.write_json(root / 'passages.json', {'records': list(passages.values())})
    data.write_json(root / 'groups.json', {'records': groups})
    data.write_json(root / 'annotations.json', {'records': annotations})
    data.write_json(root / 'plan.json', {'scope': 'within-passage dependency qualification only',
        'hashes': {name: data.file_hash(root / name)
                   for name in ['passages.json', 'groups.json', 'annotations.json']},
        'model': str(model_path.resolve()), 'source': research_identity(),
        'passages': len(passages), 'main_comparisons': 64,
        'prompt': 'Continue the passage.', 'primary': 'signed normalized squared Hellinger',
        'baseline': 'raw-edge MSE', 'minimum_predecessors': 2,
        'status': 'prepared; masks/permutations frozen before model extraction'})
    print('PREPARED', len(passages), flush=True)


def collect(root):
    import torch
    from transformers import AutoTokenizer
    from myproject.model import load_model
    from experiments.conditional_dependencies import deletion_dependencies

    plan = data.read_json(root / 'plan.json')
    for name in ['passages.json', 'groups.json']:
        if data.file_hash(root / name) != plan['hashes'][name]:
            raise ValueError('Frozen inputs/masks changed')
    identity = research_identity()
    destination = root / 'observations'
    destination.mkdir(exist_ok=False)
    model_path = Path(plan['model'])
    hashes = {name: data.file_hash(model_path / name) for name in
        ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'model.safetensors.index.json']}
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
    model = load_model(model_path, 'cuda:0')
    counts = {'backbone_calls': 0, 'processed_tokens': 0}

    def count(_module, _args, kwargs):
        counts['backbone_calls'] += 1
        counts['processed_tokens'] += kwargs['input_ids'].numel()

    hook = model.model.register_forward_pre_hook(count, with_kwargs=True)
    torch.cuda.reset_peak_memory_stats()
    start, records = time.monotonic(), []
    try:
        for passage in data.read_json(root / 'passages.json')['records']:
            result = deletion_dependencies(model, tokenizer, passage['text'])
            if result['anchors'] != passage['anchors'] or result['token_ids'].tolist() != passage['token_ids']:
                raise ValueError('Prepared lexical/token axes changed')
            file = destination / f'{passage["id"]}.npz'
            np.savez_compressed(file, **{k: v for k, v in result.items() if k != 'anchors'})
            records.append({'id': passage['id'], 'file': str(file.relative_to(root)),
                            'sha256': data.file_hash(file), 'worlds': result['worlds'],
                            'noop_max_abs_logp': result['noop_max_abs_logp']})
            print('COLLECTED', len(records), result['worlds'], flush=True)
    finally:
        hook.remove()
    torch.cuda.synchronize()
    data.write_json(root / 'collection.json', {'records': records, 'counts': counts,
        'seconds_after_model_load_including_io': time.monotonic() - start,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'model_files_sha256': hashes,
        'weight_hash_scope': 'index/config/tokenizer only, not full shards',
        'runtime': {'dtype': str(model.dtype), 'attention': model.config._attn_implementation,
                    'device': torch.cuda.get_device_name(), 'cuda': torch.version.cuda,
                    'chat_template_sha256': data.text_hash(tokenizer.chat_template)},
        'input_hashes': {n: plan['hashes'][n] for n in ['passages.json', 'groups.json']},
        'source': completed_identity(identity), 'annotation_reads': 0})


def score(root):
    identity = research_identity()
    collection = data.read_json(root / 'collection.json')
    for name, expected in collection['input_hashes'].items():
        if data.file_hash(root / name) != expected:
            raise ValueError('Frozen inputs/masks changed')
    passages = {r['id']: r for r in data.read_json(root / 'passages.json')['records']}
    matrices = {}
    for record in collection['records']:
        file = root / record['file']
        if data.file_hash(file) != record['sha256']:
            raise ValueError('Raw dependency arrays changed')
        with np.load(file, allow_pickle=False) as archive:
            matrices[record['id']] = archive['matrix']
    results = []
    for group in data.read_json(root / 'groups.json')['records']:
        words, mask = group['words'], np.asarray(group['mask'], bool)
        aligned = {}
        for identifier in group['sources'] + group['answers']:
            lookup = {w['word']: i for i, w in enumerate(passages[identifier]['anchors'])}
            rows = [lookup[w] for w in words]
            aligned[identifier] = matrices[identifier][np.ix_(rows, rows)]
        for source in group['sources']:
            for answer in group['answers']:
                costs = {}
                for v, word in enumerate(words):
                    senders = np.flatnonzero(mask[:, v])
                    if len(senders) < 2:
                        continue
                    original = aligned[source][senders, v]
                    observed = aligned[answer][senders, v]
                    permuted = aligned[source][np.asarray(group['permutation'])[senders], v]
                    primary = signed_profile_distance(original, observed)
                    null = signed_profile_distance(permuted, observed)
                    source_anchor = next(w for w in passages[source]['anchors'] if w['word'] == word)
                    answer_anchor = next(w for w in passages[answer]['anchors'] if w['word'] == word)
                    costs[word] = primary | {'null_hellinger': null['hellinger'],
                        'null_mse': null['mse'], 'senders': [words[u] for u in senders],
                        'subword_ids_equal': ([passages[source]['token_ids'][i] for i in source_anchor['tokens']]
                            == [passages[answer]['token_ids'][i] for i in answer_anchor['tokens']])}
                results.append({'family': group['name'], 'source': source, 'answer': answer,
                                'words': words, 'costs': costs})
    data.write_json(root / 'scores.json', {'cases': results, 'source': completed_identity(identity),
                                         'annotation_reads': 0})
    data.write_json(root / 'seal.json', {name: data.file_hash(root / name) for name in
        ['scores.json', 'collection.json', 'passages.json', 'groups.json']})


def evaluate(root):
    for name, expected in data.read_json(root / 'seal.json').items():
        if data.file_hash(root / name) != expected:
            raise ValueError('Score seal mismatch')
    plan = data.read_json(root / 'plan.json')
    if data.file_hash(root / 'annotations.json') != plan['hashes']['annotations.json']:
        raise ValueError('Construction labels changed')
    annotations = {(r['source'], r['answer']): r for r in data.read_json(root / 'annotations.json')['records']}
    cases = data.read_json(root / 'scores.json')['cases']
    expected_pairs = {(g['name'], s, y) for g in data.read_json(root / 'groups.json')['records']
                      for s in g['sources'] for y in g['answers']}
    if len(cases) != len(expected_pairs) or {
            (r['family'], r['source'], r['answer']) for r in cases} != expected_pairs:
        raise ValueError('Sealed scores must include every frozen comparison exactly once')
    by_family, coverage = {}, []
    for case in cases:
        annotation = annotations.get((case['source'], case['answer']))
        coverage.append({k: case[k] for k in ['family', 'source', 'answer', 'words']} | {
            'available': [w for w, r in case['costs'].items() if r['hellinger'] is not None]})
        if annotation:
            by_family.setdefault(case['family'], []).append(case | annotation)
    from experiments.research.fingerprints import qualify_dependency_family
    if set(by_family) != {f[0] for f in FAMILIES}:
        raise ValueError('Every frozen main family is required')
    summary = {family: qualify_dependency_family(rows) for family, rows in by_family.items()}
    noop = max(r['noop_max_abs_logp'] for r in data.read_json(root / 'collection.json')['records'])
    data.write_json(root / 'evaluation.json', {'families': summary, 'coverage': coverage,
        'noop_max_abs_logp': noop,
        'all_qualified': {k: noop == 0 and all(v[k]['qualified'] for v in summary.values())
                          for k in ['hellinger', 'mse']},
        'scope': 'constructed dependency qualification, not natural token AUROC'})
    print('QUALIFICATION', {f: v['hellinger']['qualified'] for f, v in summary.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'collect', 'score', 'evaluate'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    from experiments.research.run import output_path
    root = output_path(args.output, Path.cwd())
    if args.stage == 'prepare':
        if args.model is None:
            parser.error('prepare requires --model')
        prepare(root, args.model)
    else:
        {'collect': collect, 'score': score, 'evaluate': evaluate}[args.stage](root)


if __name__ == '__main__':
    main()

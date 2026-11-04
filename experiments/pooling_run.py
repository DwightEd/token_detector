"""Source-disjoint collection and exact shared-evidence/FIT evaluation."""

import argparse
from pathlib import Path
import time

import numpy as np

from experiments import data
from experiments.fit_ablation import evaluate, verify_seal
from experiments.provenance import completed_identity, research_identity
from experiments.research.pooling import grouped_scores
from experiments.research.prior import fit_prior, population_spans
from experiments.span_pilot import load_batches, select_records, validated_inputs
from experiments.research.r4 import Detector
from experiments.research.segmentation import legal_atoms, partition_scores, predictive_gain, segment

SYSTEMS = {
    'raw_full': ('raw', False), 'whole_fixed': ('whole', False),
    'punct_fixed': ('punct', False), 'dependency_fixed': ('direct_mdl', False),
    'predictive_fixed': ('predictive', False), 'predictive_fit': ('predictive', True),
    'punct_fit': ('punct', True), 'r4_punct': ('baseline', False),
    'frozen_r4': ('baseline', False),
}
CONTRASTS = {
    'fit_increment': ('predictive_fixed', 'predictive_fit'),
    'sharing_increment': ('raw_full', 'predictive_fixed'),
    'segmentation_increment': ('punct_fixed', 'predictive_fixed'),
    'versus_r4': ('r4_punct', 'predictive_fit'),
    'punct_fit_increment': ('punct_fixed', 'punct_fit'),
}


def prepare(root, cache, exclude):
    manifest = data.read_json(cache / 'manifest.json')
    old = data.read_json(exclude / 'inputs.json')['records']
    records = select_records(manifest['records'], sources=12,
                             exclude_sources=[r['source_id'] for r in old])
    for record in records:
        files = [record['source_file'], f'{record["directory"]}/response.json',
                 f'{record["directory"]}/with_source.npz', f'{record["directory"]}/without_source.npz']
        record['input_hashes'] = {f: data.file_hash(cache / f) for f in files}
    root.mkdir(parents=True, exist_ok=False)
    data.write_json(root / 'inputs.json', {'records': records, 'cache': str(cache.resolve())})
    data.write_json(root / 'plan.json', {
        'dataset': manifest['dataset'], 'model': manifest['model'], 'source': research_identity(),
        'inputs_sha256': data.file_hash(root / 'inputs.json'), 'seed': 2710,
        'sources_per_task_partition': 12, 'answers_per_source': 2,
        'exclude_source_run': str(exclude.resolve()),
        'exclude_inputs_sha256': data.file_hash(exclude / 'inputs.json'),
        'scope': 'additional144 source-disjoint from pilot48; historical train-FIT/DEV, not untouched TEST',
        'label_reads': 0, 'measurements': 'new attention; verified historical full/local native logp and route',
    })
    print('PREPARED', len(records), 'answers', flush=True)


def collect(root):
    import torch
    from experiments.research.attention import attention_profiles
    from myproject.model import load_model

    plan, inputs = validated_inputs(root)
    identity = research_identity()
    cache = Path(inputs['cache'])
    destination = root / 'observations'
    destination.mkdir(exist_ok=False)
    model_path = Path(plan['model'])
    model_hashes = {name: data.file_hash(model_path / name) for name in
                    ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                     'special_tokens_map.json', 'model.safetensors.index.json']}
    model = load_model(model_path, 'cuda:0')
    counts = {'backbone_calls': 0, 'processed_tokens': 0}

    def count(_module, _args, kwargs):
        counts['backbone_calls'] += 1
        counts['processed_tokens'] += kwargs['input_ids'].numel()

    handle = model.model.register_forward_pre_hook(count, with_kwargs=True)
    torch.cuda.reset_peak_memory_stats()
    start, records = time.monotonic(), []
    try:
        for record in inputs['records']:
            response = data.read_json(cache / record['directory'] / 'response.json')
            source = data.read_json(cache / record['source_file'])
            graph = attention_profiles(model, source['prompt_with_source'], response['answer_ids'])
            atoms, _ = legal_atoms(response)
            fitted = segment(graph['fit'], graph['available'], atoms)
            fine, _ = partition_scores(fitted['interval_gains'], atoms)
            cuts = {'punct': np.array([u['start'] for u in response['units']] + [len(graph['fit'])]),
                    'direct_mdl': fitted['cuts'], 'direct_zero_cut_cost': fine}
            with np.load(cache / record['directory'] / 'with_source.npz') as present, \
                    np.load(cache / record['directory'] / 'without_source.npz') as absent:
                if any(not np.array_equal(z['token_id'], response['answer_ids']) for z in [present, absent]):
                    raise ValueError('Cached primitive token identity changed')
                native = {'source_full_logp': present['full'], 'absent_full_logp': absent['full'],
                          'source_local_punct': present['local'], 'absent_local_punct': absent['local'],
                          'route': present['raw_route'], 'token_id': present['token_id']}
            native.update({'fit_profiles': graph['fit'], 'hold_profiles': graph['hold'],
                           'available': graph['available'], 'raw_attention_mass': graph['raw_mass'],
                           **{f'cuts_{m}': c for m, c in cuts.items()}})
            file = destination / f'{record["id"]}.npz'
            with file.open('xb') as stream:
                np.savez_compressed(stream, **native)
            records.append({'id': record['id'], 'file': str(file.relative_to(root)),
                'sha256': data.file_hash(file), 'prompt_tokens': graph['prompt_tokens'],
                'answer_tokens': len(graph['fit']), 'cuts': {m: c.tolist() for m, c in cuts.items()},
                'held_gain': {m: predictive_gain(graph['fit'], graph['hold'], graph['available'], c)
                              for m, c in cuts.items()}})
            print('COLLECTED', len(records), record['task'], record['partition'], record['id'], flush=True)
    finally:
        handle.remove()
    torch.cuda.synchronize()
    validated_inputs(root)
    data.write_json(root / 'collection.json', {'records': records, 'counts': counts,
        'input_sha256': plan['inputs_sha256'], 'model_files_sha256': model_hashes,
        'weight_hash_scope': 'config/tokenizer/index, not weight shards',
        'seconds_attention_and_segmentation_including_io': time.monotonic()-start,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'source': completed_identity(identity),
        'label_reads': 0})
    data.write_json(root / 'seal.json', {name: data.file_hash(root / name)
        for name in ['plan.json', 'inputs.json', 'collection.json']})


def _unit_index(method, packed, collection):
    if method == 'whole':
        return np.zeros(len(packed['target']), dtype=int)
    result = np.empty(len(packed['target']), dtype=int)
    for answer in np.unique(packed['answer_index']):
        record = collection['records'][answer]
        selected = (max(['direct_mdl', 'direct_zero_cut_cost'], key=lambda m: record['held_gain'][m])
                    if method == 'predictive' else method)
        rows = packed['answer_index'] == answer
        result[rows] = np.searchsorted(record['cuts'][selected][1:], packed['target'][rows], side='right')
    return result


def score(source, output):
    verify_seal(source)
    plan, inputs = validated_inputs(source)
    collection = data.read_json(source / 'collection.json')
    identity = research_identity()
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output / 'plan.json', {
        'source_run': str(source.resolve()), 'source_seal_sha256': data.file_hash(source / 'seal.json'),
        'source': identity, 'scope': plan['scope'], 'systems': SYSTEMS, 'contrasts': CONTRASTS,
        'fit_prior': 'Gamma(2,1) MAP; source/answer/span-balanced composite evidence', 'label_reads': 0})
    batches = load_batches(source, methods=['punct'])
    files, priors = [], {}
    for task in data.TASKS:
        packed = batches[(task, 'punct')]
        fit_mask, dev_mask = ~packed['development'], packed['development']
        original_dev = {k: v[dev_mask] for k, v in packed.items()}
        source_index = np.asarray([inputs['records'][a]['source_id'] for a in packed['answer_index']])
        r4 = Detector(task).fit({k: v[fit_mask] for k, v in packed.items()})
        for name, (method, use_fit) in SYSTEMS.items():
            prior = {'alpha': 1., 'beta': 1., 'fit': False}
            extra = {}
            if method == 'baseline':
                readout = (r4 if name == 'r4_punct' else Detector.load(
                    Path(__file__).parents[1] / 'myproject' / 'references' / f'{task}.npz'))
                result = readout.score(original_dev)
            elif method == 'raw':
                result = packed['full_contrast'][dev_mask]
            else:
                units = _unit_index(method, packed, collection)
                if use_fit:
                    spans, weights = population_spans(packed['full_contrast'][fit_mask],
                        packed['answer_index'][fit_mask], units[fit_mask], source_index[fit_mask])
                    prior = fit_prior(spans, weights) | {'fit': True}
                extra = grouped_scores(packed['full_contrast'][dev_mask], packed['answer_index'][dev_mask],
                                       units[dev_mask], prior['alpha'], prior['beta'])
                result = extra.pop('score')
            if not np.isfinite(result).all():
                raise ValueError('Nonfinite posterior score')
            priors[f'{task}/{name}'] = prior
            file = output / f'{task}_{name}.npz'
            with file.open('xb') as stream:
                np.savez_compressed(stream, score=result, **extra,
                    **{k: original_dev[k] for k in ['target', 'token_id', 'answer_index']})
            files.append(file)
        print('SCORED', task, flush=True)
    data.write_json(output / 'priors.json', priors)
    verify_seal(source)
    validated_inputs(source)
    data.write_json(output / 'complete.json', {'source': completed_identity(identity), 'label_reads': 0})
    data.write_json(output / 'seal.json', {str(f.relative_to(output)): data.file_hash(f)
        for f in [*files, output/'priors.json', output/'plan.json', output/'complete.json']})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['prepare', 'collect', 'score', 'evaluate'])
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--exclude', type=Path)
    args = parser.parse_args()
    if args.stage == 'prepare':
        if args.cache is None or args.exclude is None:
            parser.error('prepare needs --cache and --exclude')
        prepare(args.output, args.cache, args.exclude)
    elif args.stage == 'collect':
        collect(args.output)
    elif args.stage == 'score':
        if args.input is None:
            parser.error('score needs --input')
        score(args.input, args.output)
    else:
        evaluate(args.output)


if __name__ == '__main__':
    main()

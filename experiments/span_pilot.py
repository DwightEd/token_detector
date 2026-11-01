"""Source-disjoint natural-data pilot: directed spans through the complete R4 scorer."""

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

from experiments import data
from experiments.provenance import completed_identity, research_identity
from experiments.research.spans import constrained_partition, contexts
from myproject.segmentation import legal_atoms, partition_scores, predictive_gain, segment
from myproject.detector import Detector
from myproject.measure import answer_rows

METHODS = ['punct', 'walk1', 'direct_mdl', 'direct_zero_cut_cost']


def select_records(records, sources=4, answers=2, seed=2710, exclude_sources=()):
    """Label-blind source/answer selection shared by native experiment runners."""
    excluded = set(map(str, exclude_sources))
    grouped = defaultdict(list)
    for record in data.partition_records(records):
        if record['partition'] in {'fit', 'dev'} and str(record['source_id']) not in excluded:
            grouped[(record['task'], record['partition'], record['source_id'])].append(record)
    selected = []
    for task in data.TASKS:
        for partition in ['fit', 'dev']:
            keys = [k for k, v in grouped.items() if k[:2] == (task, partition) and len(v) >= answers]
            keys.sort(key=lambda k: data.text_hash(f'{seed}:{task}:{partition}:{k[2]}'))
            if len(keys) < sources:
                raise ValueError('Insufficient complete sources')
            for key in keys[:sources]:
                rows = sorted(grouped[key], key=lambda r: data.text_hash(f'{seed}:{r["id"]}'))[:answers]
                selected.extend(rows)
    return selected


def prepare(root, cache, sources=4, answers=2, seed=2710):
    manifest = data.read_json(cache / 'manifest.json')
    selected = []
    for record in select_records(manifest['records'], sources, answers, seed):
        files = [record['source_file'], f'{record["directory"]}/response.json']
        selected.append(record | {'input_hashes': {f: data.file_hash(cache / f) for f in files}})
    root.mkdir(parents=True, exist_ok=False)
    data.write_json(root / 'inputs.json', {'records': selected, 'cache': str(cache.resolve())})
    data.write_json(root / 'plan.json', {'dataset': manifest['dataset'], 'model': manifest['model'],
        'seed': seed, 'sources_per_task_partition': sources, 'answers_per_source': answers,
        'inputs_sha256': data.file_hash(root / 'inputs.json'), 'source': research_identity(),
        'methods': METHODS, 'scope': 'exposed FIT/DEV pilot; not full-dataset or untouched TEST',
        'parameters': {'prior_mass': 1, 'boundary_cost': 'log(2 * legal_interior_cuts)',
                       'head_views': 'even/odd physical heads averaged over all layers'},
        'label_reads': 0})
    print('PREPARED', len(selected), flush=True)


def units(cuts):
    return [{'start': int(a), 'stop': int(b)} for a, b in zip(cuts[:-1], cuts[1:])]


def validated_inputs(root):
    plan, inputs = data.read_json(root / 'plan.json'), data.read_json(root / 'inputs.json')
    if data.file_hash(root / 'inputs.json') != plan['inputs_sha256']:
        raise ValueError('Frozen selection changed')
    cache = Path(inputs['cache'])
    bindings = {name: value for record in inputs['records']
                for name, value in record['input_hashes'].items()}
    if any(data.file_hash(cache / name) != expected for name, expected in bindings.items()):
        raise ValueError('Original input changed')
    return plan, inputs


def collect(root):
    import torch
    from myproject.attention import attention_profiles
    from myproject.model import load_model, prefill, score_continuation
    from myproject.routing import projection_grams
    from myproject.source import source_condition

    plan, inputs = validated_inputs(root)
    cache = Path(inputs['cache'])
    identity = research_identity()
    destination = root / 'observations'
    destination.mkdir(exist_ok=False)
    model_path = Path(plan['model'])
    model_hashes = {name: data.file_hash(model_path / name) for name in
        ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
         'model.safetensors.index.json']}
    model = load_model(model_path, 'cuda:0')
    grams = projection_grams(model)
    counts = {'backbone_calls': 0, 'processed_tokens': 0}

    def count(_module, _args, kwargs):
        counts['backbone_calls'] += 1
        counts['processed_tokens'] += kwargs['input_ids'].numel()

    handle = model.model.register_forward_pre_hook(count, with_kwargs=True)
    start, records = time.monotonic(), []
    torch.cuda.reset_peak_memory_stats()
    try:
        for record in inputs['records']:
            source = data.read_json(cache / record['source_file'])
            response = data.read_json(cache / record['directory'] / 'response.json')
            graph = attention_profiles(model, source['prompt_with_source'], response['answer_ids'])
            atoms, _ = legal_atoms(response)
            directed = segment(graph['fit'], graph['available'], atoms)
            unpenalized, _ = partition_scores(directed['interval_gains'], atoms)
            p = graph['prompt_tokens']
            history = graph['fit'][:, p:p+len(response['answer_ids'])]
            _, views, _ = contexts(history)
            cuts = {'punct': np.array([u['start'] for u in response['units']] + [len(history)]),
                    'walk1': constrained_partition(views[1][1], atoms),
                    'direct_mdl': directed['cuts'], 'direct_zero_cut_cost': unpenalized}
            local, full, route = source_condition(model, source['prompt_with_source'],
                response['answer_ids'], response['units'], source['source_mask'], grams)
            absent_local, absent_full, _ = source_condition(model, source['prompt_without_source'],
                response['answer_ids'], response['units'])
            native = {'fit_profiles': graph['fit'], 'hold_profiles': graph['hold'],
                      'available': graph['available'], 'raw_attention_mass': graph['raw_mass'],
                      'source_full_logp': full, 'absent_full_logp': absent_full,
                      'route': route, 'token_id': np.asarray(response['answer_ids']),
                      'source_local_punct': local, 'absent_local_punct': absent_local,
                      'direct_predicted': directed['predicted'], 'direct_residual': directed['residual'],
                      'direct_shared': directed['shared'], 'direct_background': directed['background']}
            native['direct_interval_gains'] = directed['interval_gains']
            for present in [True, False]:
                prefix = source['prompt_with_source' if present else 'prompt_without_source']
                prefix_cache = prefill(model, prefix[:-1])
                reuse = {tuple(cuts['punct']): local if present else absent_local}
                for method in METHODS:
                    signature = tuple(cuts[method])
                    if signature not in reuse:
                        probabilities = np.empty(len(history))
                        for a, b in zip(signature[:-1], signature[1:]):
                            prefix_cache.crop(len(prefix)-1)
                            probabilities[a:b] = score_continuation(model, prefix_cache, prefix[-1],
                                                                    response['answer_ids'][a:b])
                        reuse[signature] = probabilities
                    native[f'{"source" if present else "absent"}_local_{method}'] = reuse[signature]
            native.update({f'cuts_{method}': value for method, value in cuts.items()})
            file = destination / f'{record["id"]}.npz'
            with file.open('xb') as stream:
                np.savez_compressed(stream, **native)
            records.append({'id': record['id'], 'file': str(file.relative_to(root)),
                'sha256': data.file_hash(file), 'prompt_tokens': p, 'answer_tokens': len(history),
                'cuts': {m: c.tolist() for m, c in cuts.items()},
                'held_gain': {m: predictive_gain(graph['fit'], graph['hold'], graph['available'], c)
                              for m, c in cuts.items()}})
            print('COLLECTED', len(records), record['task'], record['partition'], record['id'],
                  {m: len(c)-1 for m, c in cuts.items()}, flush=True)
    finally:
        handle.remove()
    torch.cuda.synchronize()
    validated_inputs(root)
    data.write_json(root / 'collection.json', {'records': records, 'counts': counts,
        'seconds_measurement_loop_including_io': time.monotonic()-start,
        'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'input_sha256': plan['inputs_sha256'],
        'model_files_sha256': model_hashes, 'weight_hash_scope': 'index/config/tokenizer; not shards',
        'runtime': {'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
                    'dtype': str(model.dtype)}, 'source': completed_identity(identity), 'label_reads': 0})


def load_batches(root, methods=METHODS):
    """Rebuild original-axis packs from verified native arrays for any readout."""
    plan, inputs = validated_inputs(root)
    collection = data.read_json(root / 'collection.json')
    if data.file_hash(root / 'inputs.json') != collection['input_sha256']:
        raise ValueError('Collection inputs changed')
    cache = Path(inputs['cache'])
    batches = defaultdict(list)
    for index, (record, observed) in enumerate(zip(inputs['records'], collection['records'], strict=True)):
        if record['id'] != observed['id'] or data.file_hash(root / observed['file']) != observed['sha256']:
            raise ValueError('Native identity/seal mismatch')
        response = data.read_json(cache / record['directory'] / 'response.json')
        with np.load(root / observed['file'], allow_pickle=False) as raw:
            for method in methods:
                selected = method
                if method == 'predictive':
                    selected = max(['direct_mdl', 'direct_zero_cut_cost'],
                                   key=lambda m: observed['held_gain'][m])
                adapted = response | {'units': units(raw[f'cuts_{selected}'])}
                values = {'delta': np.column_stack((raw[f'absent_local_{selected}'].astype(np.float64)-raw[f'source_local_{selected}'],
                                                   raw['absent_full_logp'].astype(np.float64)-raw['source_full_logp'])),
                          'route': raw['route'], 'token_id': raw['token_id']}
                batch = answer_rows(values, adapted, index)
                batch['full_contrast'] = values['delta'][batch['target'], 1]
                batch['development'] = np.full(len(batch['target']), record['partition']=='dev', bool)
                batches[(record['task'], method)].append(batch)
    return {key: {field: np.concatenate([row[field] for row in rows])
                  for field in (*data.FIELDS, 'full_contrast')}
            for key, rows in batches.items()}


def score(root):
    identity = research_identity()
    batches = load_batches(root)
    destination = root / 'predictions'
    destination.mkdir(exist_ok=False)
    files = []
    for (task, method), packed in batches.items():
        fit = {k: v[~packed['development']] for k, v in packed.items()}
        dev = {k: v[packed['development']] for k, v in packed.items()}
        segmentation = {'punct': 'punct', 'direct_mdl': 'dependency'}.get(method, 'precomputed')
        detector = Detector(task, segmentation=segmentation).fit(fit)
        reference = destination / f'{task}_{method}_reference.npz'
        detector.save(reference)
        prediction = {'score': detector.score(dev), **{k: dev[k] for k in ['target','token_id','answer_index']}}
        file = destination / f'{task}_{method}.npz'
        with file.open('xb') as stream:
            np.savez_compressed(stream, **prediction)
        files.extend([file, reference])
        if method == 'punct':
            reference = Path(__file__).parents[1] / 'myproject' / 'references' / f'{task}.npz'
            prediction['score'] = Detector.load(reference).score(dev)
            file = destination / f'{task}_frozen_R4.npz'
            with file.open('xb') as stream:
                np.savez_compressed(stream, **prediction)
            files.append(file)
    data.write_json(root / 'score_identity.json', {'source': completed_identity(identity), 'label_reads':0})
    validated_inputs(root)
    data.write_json(root / 'seal.json', {str(f.relative_to(root)): data.file_hash(f) for f in
        [*files, root/'inputs.json', root/'plan.json', root/'collection.json', root/'score_identity.json']})
    print('SEALED',len(files),'files',flush=True)


def evaluate(root):
    from experiments.evaluate import official_labels, ranking
    from experiments.evaluate import paired_source_differences

    for name, expected in data.read_json(root/'seal.json').items():
        if data.file_hash(root/name) != expected:
            raise ValueError('Score seal mismatch')
    plan, inputs = validated_inputs(root)
    metadata = {'records': inputs['records'], 'dataset': plan['dataset'], 'response_root': inputs['cache']}
    results = {}
    for task in data.TASKS:
        methods = {}
        for method in [*METHODS,'frozen_R4']:
            with np.load(root/'predictions'/f'{task}_{method}.npz',allow_pickle=False) as raw:
                methods[method] = {k: raw[k] for k in raw.files}
        labels = official_labels(methods['punct'],metadata)
        groups = np.array([inputs['records'][i]['source_id'] for i in methods['punct']['answer_index']])
        metrics = {}
        for method, packed in methods.items():
            for field in ['token_id','target','answer_index']:
                if not np.array_equal(packed[field],methods['punct'][field]):
                    raise ValueError('Variants must score identical original tokens')
            metrics[method] = ranking(labels,packed['score'])
        paired = {m: paired_source_differences(labels,methods['punct']['score'],methods[m]['score'],groups)
                  for m in METHODS if m!='punct'}
        results[task] = {'metrics':metrics,'paired_vs_matched_punct':paired,
                        'dev_sources':len(set(groups)), 'dev_answers':len(set(methods['punct']['answer_index']))}
    data.write_json(root/'evaluation.json',{'scope':plan['scope'],'tasks':results,
        'annotation_sha256': data.file_hash(Path(plan['dataset'])/'response.jsonl')})
    for task, result in results.items():
        print(task,{m:(v['auroc'],v['ap']) for m,v in result['metrics'].items()},flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['prepare','collect','score','evaluate'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--sources',type=int,default=4)
    parser.add_argument('--answers',type=int,default=2)
    args=parser.parse_args()
    from experiments.research.run import output_path
    root=output_path(args.output,Path.cwd())
    if args.stage=='prepare':
        if args.cache is None or args.sources<1 or args.answers<1:
            parser.error('prepare needs a cache and positive source/answer counts')
        prepare(root,args.cache,args.sources,args.answers)
    else:
        {'collect':collect,'score':score,'evaluate':evaluate}[args.stage](root)


if __name__=='__main__':
    main()

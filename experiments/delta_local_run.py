"""Collect paired post-states and test fixed local-direction readouts without labels."""

import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from experiments import data
from experiments.fit_ablation import evaluate, verify_seal
from experiments.pooling_run import _unit_index
from experiments.provenance import completed_identity, research_identity
from experiments.research.delta_local import EPSILON, direction_scores, fit_whitening, whiten
from myproject.pooling import grouped_scores
from experiments.span_pilot import load_batches, validated_inputs
from myproject.detector import Detector
from myproject.measure import answer_rows

METHODS = {'punct': 'punct', 'dependency': 'direct_mdl'}
SYSTEMS = ['direction_dependency', 'direction_punct', 'whitened_direction_dependency',
           'whitened_direction_punct', 'source_direction_punct', 'absent_direction_punct',
           'minus_delta_norm', 'native_gap', 'native_pool_dependency', 'native_pool_punct',
           'old_gap', 'nll', 'entropy', 'excess_surprisal', 'r4_punct', 'frozen_r4']
CONTRASTS = {
    'new_observation': ('native_pool_dependency', 'direction_dependency'),
    'matched_punctuation': ('native_pool_punct', 'direction_punct'),
    'fit_increment': ('direction_dependency', 'whitened_direction_dependency'),
    'versus_r4': ('r4_punct', 'direction_dependency'),
    'direction_vs_strength': ('minus_delta_norm', 'direction_dependency'),
    'paired_vs_present': ('source_direction_punct', 'direction_punct'),
    'paired_vs_absent': ('absent_direction_punct', 'direction_punct'),
}


def prepare(root, old, additional, protocol):
    cohorts, indexed, bindings, cache, model = {}, {}, {}, None, None
    for name, scores in [('old48', old), ('additional144', additional)]:
        verify_seal(scores)
        previous = data.read_json(scores/'plan.json')
        source = Path(previous['source_run'])
        verify_seal(source)
        plan, inputs = validated_inputs(source)
        if data.file_hash(source/'seal.json') != previous['source_seal_sha256']:
            raise ValueError('Previous native source binding changed')
        if cache is not None and (cache != inputs['cache'] or model != plan['model']):
            raise ValueError('Both cohorts must use the same model and primitive cache')
        cache, model = inputs['cache'], plan['model']
        cohorts[name] = {'source': str(source.resolve()), 'scores': str(scores.resolve()),
                         'source_seal_sha256': data.file_hash(source/'seal.json'),
                         'score_seal_sha256': data.file_hash(scores/'seal.json')}
        observed = data.read_json(source/'collection.json')['records']
        for record, row in zip(inputs['records'], observed, strict=True):
            if record['id'] in indexed:
                raise ValueError('Original cohorts must not share answers')
            indexed[record['id']] = {'record': record, 'uses': [name], 'cuts': row['cuts']}
    manifest = data.read_json(Path(cache)/'manifest.json')
    for record in manifest['records']:
        if record['id'] in {'3182', '3180', '10988', '2390'}:
            if record['id'] in indexed:
                indexed[record['id']]['uses'].append('case')
            else:
                indexed[record['id']] = {'record': record, 'uses': ['case'], 'cuts': None}
    for entry in indexed.values():
        r = entry['record']
        for file in [r['source_file'], *[f'{r["directory"]}/{n}' for n in
                                       ['response.json', 'with_source.npz', 'without_source.npz']]]:
            bindings[file] = data.file_hash(Path(cache)/file)
    overlap = {}
    for name, cohort in cohorts.items():
        _, original = validated_inputs(Path(cohort['source']))
        overlap[name] = {key: sorted({r['source_id'] for r in original['records'] if r['partition'] == key}
                                     & {e['record']['source_id'] for e in indexed.values() if 'case' in e['uses']})
                         for key in ['fit', 'dev']}
    root.mkdir(parents=True, exist_ok=False)
    data.write_json(root/'inputs.json', {'cache': cache, 'records': list(indexed.values()),
                                       'input_hashes': bindings})
    data.write_json(root/'plan.json', {'cohorts': cohorts, 'model': model, 'dataset': manifest['dataset'],
        'protocol': str(protocol.resolve()), 'protocol_sha256': data.file_hash(protocol),
        'inputs_sha256': data.file_hash(root/'inputs.json'), 'source': research_identity(),
        'systems': SYSTEMS, 'contrasts': CONTRASTS, 'label_reads': 0,
        'scope': 'two historical exposed FIT/DEV cohorts and four posthoc cases; not untouched TEST',
        'case_source_overlap': overlap, 'backend': 'sdpa both worlds; native BF16, FP32 readout',
        'epsilon': EPSILON})
    print('PREPARED', len(indexed), 'distinct answers', flush=True)


def validate(root):
    plan, inputs = data.read_json(root/'plan.json'), data.read_json(root/'inputs.json')
    if (data.file_hash(root/'inputs.json') != plan['inputs_sha256']
            or data.file_hash(Path(plan['protocol'])) != plan['protocol_sha256']):
        raise ValueError('Frozen inputs/protocol changed')
    if any(data.file_hash(Path(inputs['cache'])/f) != h for f,h in inputs['input_hashes'].items()):
        raise ValueError('Primitive input changed')
    for cohort in plan['cohorts'].values():
        for key, expected in [('source', 'source_seal_sha256'), ('scores', 'score_seal_sha256')]:
            if data.file_hash(Path(cohort[key])/'seal.json') != cohort[expected]:
                raise ValueError('Cohort seal changed')
    return plan, inputs


def collect(root):
    import torch
    from experiments.delta_observe import hidden_trace
    from myproject.attention import attention_profiles
    from myproject.model import load_model
    from myproject.segmentation import legal_atoms, segment

    plan, inputs = validate(root)
    identity = research_identity()
    destination = root/'observations'
    destination.mkdir(exist_ok=False)
    model_path = Path(plan['model'])
    hashes = {n:data.file_hash(model_path/n) for n in ['config.json', 'tokenizer.json',
        'tokenizer_config.json', 'special_tokens_map.json', 'model.safetensors.index.json']}
    torch.manual_seed(42)
    model = load_model(model_path, 'cuda:0')
    counts = {'backbone_calls': 0, 'processed_tokens': 0}

    def count(_module, _args, kwargs):
        counts['backbone_calls'] += 1
        counts['processed_tokens'] += kwargs['input_ids'].numel()

    handle = model.model.register_forward_pre_hook(count, with_kwargs=True)
    torch.cuda.reset_peak_memory_stats()
    start, observed = time.monotonic(), []
    try:
        for entry in inputs['records']:
            r = entry['record']; cache = Path(inputs['cache'])
            response = data.read_json(cache/r['directory']/'response.json')
            source = data.read_json(cache/r['source_file'])
            traces = {key:hidden_trace(model, source[prompt], response['answer_ids']) for key,prompt in
                      [('present','prompt_with_source'), ('absent','prompt_without_source')]}
            cuts = entry['cuts']
            if cuts is None:
                graph = attention_profiles(model, source['prompt_with_source'], response['answer_ids'])
                cuts = {'punct':[u['start'] for u in response['units']]+[len(response['answer_ids'])],
                        'direct_mdl':segment(graph['fit'], graph['available'], legal_atoms(response)[0])['cuts'].tolist()}
            arrays = {f'{world}_{key}':value for world,trace in traces.items() for key,value in trace.items()}
            arrays['token_id'] = np.asarray(response['answer_ids'])
            if any(not np.isfinite(a).all() for a in arrays.values()):
                raise ValueError('Nonfinite native trace')
            comparison = {}
            for world, file in [('present','with_source.npz'), ('absent','without_source.npz')]:
                with np.load(cache/r['directory']/file) as old:
                    if not np.array_equal(old['token_id'], arrays['token_id']):
                        raise ValueError('Original token axes changed')
                    diff = traces[world]['logp'].astype(float)-old['full']
                    comparison[world] = {'mean_abs':float(np.mean(abs(diff))), 'max_abs':float(np.max(abs(diff)))}
            file = destination/f'{r["id"]}.npz'
            with file.open('xb') as stream: np.savez_compressed(stream, **arrays)
            observed.append({'id':r['id'], 'file':str(file.relative_to(root)), 'sha256':data.file_hash(file),
                             'cuts':cuts, 'tokens':len(response['answer_ids']), 'cached_logp_difference':comparison})
            print('COLLECTED', len(observed), r['id'], flush=True)
    finally:
        handle.remove()
    torch.cuda.synchronize()
    validate(root)
    data.write_json(root/'collection.json', {'records':observed, 'counts':counts,
        'seconds_including_io':time.monotonic()-start, 'peak_cuda_bytes':torch.cuda.max_memory_allocated(),
        'source':completed_identity(identity), 'model_files_sha256':hashes,
        'weight_hash_scope':'config/tokenizer/index, not weight shards', 'label_reads':0})
    data.write_json(root/'seal.json', {n:data.file_hash(root/n) for n in ['inputs.json','plan.json','collection.json']})


def _readouts(present, absent, gap, nll, entropy, packed, units, fit, sources, output, task):
    delta = present.astype(float)-absent
    raw_nonzero = np.linalg.norm(delta, axis=1) > EPSILON
    result = {'native_gap': {'score':gap}, 'old_gap':{'score':packed['full_contrast']},
        'minus_delta_norm':{'score':-np.linalg.norm(delta, axis=1)}, 'nll':{'score':nll},
        'entropy':{'score':entropy}, 'excess_surprisal':{'score':nll-entropy},
        'source_direction_punct':direction_scores(present, packed['answer_index'], units['punct']),
        'absent_direction_punct':direction_scores(absent, packed['answer_index'], units['punct'])}
    for method in METHODS:
        indices = units[method]
        result[f'direction_{method}'] = direction_scores(delta, packed['answer_index'], indices)
        parameters = fit_whitening(delta[fit], packed['answer_index'][fit], indices[fit], sources[fit])
        with (output/f'{task}_{method}_whitening.npz').open('xb') as stream:
            np.savez_compressed(stream, **parameters)
        result[f'whitened_direction_{method}'] = direction_scores(
            whiten(delta, parameters), packed['answer_index'], indices, raw_nonzero)
        result[f'native_pool_{method}'] = {'score':grouped_scores(gap, packed['answer_index'], indices)['score']}
    return result


def score(root):
    verify_seal(root)
    plan, inputs = validate(root)
    identity = research_identity()
    observed = {r['id']:r for r in data.read_json(root/'collection.json')['records']}
    for row in observed.values():
        if data.file_hash(root/row['file']) != row['sha256']: raise ValueError('Native state seal changed')
    for name, cohort in plan['cohorts'].items():
        source, previous = Path(cohort['source']), Path(cohort['scores'])
        verify_seal(source); verify_seal(previous)
        original_plan, original = validated_inputs(source)
        batches = load_batches(source, methods=['punct'])
        collection = data.read_json(source/'collection.json')
        output = root/f'scores_{name}'; output.mkdir(exist_ok=False)
        data.write_json(output/'plan.json', {'source_run':str(source),
            'source_seal_sha256':cohort['source_seal_sha256'], 'new_observation_run':str(root.resolve()),
            'new_observation_seal_sha256':data.file_hash(root/'seal.json'), 'source':identity,
            'systems':SYSTEMS, 'contrasts':CONTRASTS, 'scope':plan['scope'], 'label_reads':0})
        for task in data.TASKS:
            packed = batches[(task,'punct')]
            arrays = {key:[] for key in ['present_hidden','absent_hidden','present_logp','absent_logp','present_entropy']}
            for answer in np.unique(packed['answer_index']):
                r = original['records'][answer]; targets=packed['target'][packed['answer_index']==answer]
                with np.load(root/observed[r['id']]['file']) as raw:
                    if not np.array_equal(raw['token_id'][targets], packed['token_id'][packed['answer_index']==answer]):
                        raise ValueError('State and scored token identities differ')
                    for key in arrays: arrays[key].append(raw[key][targets])
            arrays={k:np.concatenate(v) for k,v in arrays.items()}
            units={m:_unit_index(key,packed,collection) for m,key in METHODS.items()}
            fit,dev = ~packed['development'],packed['development']
            sources=np.asarray([original['records'][a]['source_id'] for a in packed['answer_index']])
            readouts=_readouts(arrays['present_hidden'],arrays['absent_hidden'],
                arrays['absent_logp'].astype(float)-arrays['present_logp'], -arrays['present_logp'].astype(float),
                arrays['present_entropy'].astype(float), packed,units,fit,sources,output,task)
            for system in SYSTEMS:
                file=output/f'{task}_{system}.npz'
                if system in {'r4_punct','frozen_r4'}:
                    with (previous/file.name).open('rb') as src,file.open('xb') as dst: shutil.copyfileobj(src,dst)
                else:
                    value={k:v[dev] for k,v in readouts[system].items()}
                    method = 'dependency' if system.endswith('dependency') else 'punct'
                    value['unit_index'] = units[method][dev]
                    if not np.isfinite(value['score']).all(): raise ValueError('Nonfinite readout')
                    with file.open('xb') as stream:
                        np.savez_compressed(stream, **value,
                            **{k:packed[k][dev] for k in ['answer_index','target','token_id']})
            print('SCORED',name,task,flush=True)
        validated_inputs(source);verify_seal(previous)
        data.write_json(output/'complete.json', {'source':completed_identity(identity),'label_reads':0})
        data.write_json(output/'seal.json', {f.name:data.file_hash(f) for f in output.iterdir()})
    score_cases(root, inputs, observed)
    for row in observed.values():
        if data.file_hash(root/row['file']) != row['sha256']:
            raise ValueError('Native state changed during scoring')
    data.write_json(root/'score_complete.json', {
        'state_seal_sha256': data.file_hash(root/'seal.json'),
        'prediction_seals': {name:data.file_hash(root/f'scores_{name}'/'seal.json')
                             for name in plan['cohorts']},
        'case_seal_sha256': data.file_hash(root/'case_readouts'/'seal.json'),
        'source': completed_identity(identity), 'label_reads': 0})


def score_cases(root, inputs, observed):
    """Same readouts on the prior cases, without reading their official labels."""
    identity=research_identity(); cases=[]
    for entry in inputs['records']:
        if 'case' not in entry['uses']: continue
        r=entry['record']; response=data.read_json(Path(inputs['cache'])/r['directory']/'response.json')
        with np.load(root/observed[r['id']]['file']) as raw, \
                np.load(Path(inputs['cache'])/r['directory']/'with_source.npz') as old_p, \
                np.load(Path(inputs['cache'])/r['directory']/'without_source.npz') as old_a:
            delta=np.column_stack([old_a[k].astype(float)-old_p[k] for k in ['local','full']])
            packed=answer_rows({'delta':delta,'route':old_p['raw_route'],'token_id':old_p['token_id']},response,0)
            target=packed['target']; states=raw['present_hidden'][target].astype(float)-raw['absent_hidden'][target]
            values={'native_gap':raw['absent_logp'][target].astype(float)-raw['present_logp'][target],
                    'minus_delta_norm':-np.linalg.norm(states,axis=1)}
            for method,key in METHODS.items():
                units=np.searchsorted(observed[r['id']]['cuts'][key][1:],target,side='right')
                values[f'direction_{method}']=direction_scores(states,packed['answer_index'],units)['score']
            ref=Path(__file__).parents[1]/'myproject/references'/f'{r["task"]}.npz'
            values['r4_punct']=Detector.load(ref).score(packed)
        cases.append({'id':r['id'],'source_id':r['source_id'],'response':response['text'],
            'offsets':[response['offsets'][i] for i in target], 'target':target.tolist(),
            'token_id':packed['token_id'].tolist(),'scores':{k:v.tolist() for k,v in values.items()}})
    destination=root/'case_readouts';destination.mkdir(exist_ok=False)
    data.write_json(destination/'scores.json',{'cases':cases,'label_reads':0,'source':completed_identity(identity)})
    data.write_json(destination/'seal.json',{'scores.json':data.file_hash(destination/'scores.json')})
    validate(root);verify_seal(root)


def evaluation_gate(root):
    """Every cohort and case score must be sealed before the first label join."""
    complete = data.read_json(root/'score_complete.json')
    verify_seal(root)
    plan, _ = validate(root)
    if data.file_hash(root/'seal.json') != complete['state_seal_sha256']:
        raise ValueError('Completed native seal changed')
    if set(complete['prediction_seals']) != set(plan['cohorts']):
        raise ValueError('Not all cohorts completed scoring')
    for name, expected in complete['prediction_seals'].items():
        output = root/f'scores_{name}'
        verify_seal(output)
        if data.file_hash(output/'seal.json') != expected:
            raise ValueError('Completed predictions changed')
        if data.read_json(output/'plan.json')['new_observation_seal_sha256'] != complete['state_seal_sha256']:
            raise ValueError('Prediction/native binding changed')
    verify_seal(root/'case_readouts')
    if data.file_hash(root/'case_readouts'/'seal.json') != complete['case_seal_sha256']:
        raise ValueError('Completed case scores changed')
    return plan


def evaluate_cohort(root, name):
    from experiments.evaluate import official_labels, ranking
    from myproject.detector import unit_ranges

    plan = evaluation_gate(root)
    output = root/f'scores_{name}'
    evaluate(output)
    source = Path(plan['cohorts'][name]['source'])
    original_plan, inputs = validated_inputs(source)
    metadata = {'records':inputs['records'], 'dataset':original_plan['dataset'],
                'response_root':inputs['cache']}
    tasks = {}
    for task in data.TASKS:
        with np.load(output/f'{task}_r4_punct.npz') as raw:
            baseline = dict(raw)
        labels = official_labels(baseline, metadata)
        answers = [baseline['answer_index'] == i for i in np.unique(baseline['answer_index'])]
        mixed = [m for m in answers if 0 < labels[m].sum() < m.sum()]
        systems = {}
        for system in SYSTEMS:
            with np.load(output/f'{task}_{system}.npz') as raw:
                available = raw['available'] if 'available' in raw.files else np.ones(len(labels),bool)
                metrics = [ranking(labels[m], raw['score'][m]) for m in mixed]
                row = {'tokens':len(labels), 'available':int(available.sum()),
                       'unavailable_positive':int(labels[~available].sum()),
                       'macro_within_answer':{k:float(np.mean([v[k] for v in metrics])) if metrics else None
                                              for k in ['auroc','ap']}}
                if 'unit_index' in raw.files:
                    first,last = unit_ranges(raw['answer_index'],raw['unit_index'])
                    size = last-first
                    row['span_sizes'] = {'one':int(np.sum(size==1)), 'two':int(np.sum(size==2)),
                                         'above_two':int(np.sum(size>2))}
                systems[system] = row
        tasks[task] = {'mixed_answers':len(mixed), 'total_answers':len(answers), 'systems':systems}
    data.write_json(output/'diagnostics.json', {'scope':'coverage and conditional mixed-answer diagnostics; '
        'not the all-token benchmark or selection criterion', 'tasks':tasks})


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('stage',choices=['prepare','collect','score','evaluate'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--old',type=Path)
    parser.add_argument('--additional',type=Path)
    parser.add_argument('--protocol',type=Path)
    parser.add_argument('--cohort',choices=['old48','additional144'])
    args=parser.parse_args()
    if args.stage=='prepare':
        if any(v is None for v in [args.old,args.additional,args.protocol]):
            parser.error('prepare needs --old, --additional and --protocol')
        prepare(args.output,args.old,args.additional,args.protocol)
    elif args.stage=='collect':collect(args.output)
    elif args.stage=='score':score(args.output)
    else:
        if args.cohort is None:parser.error('evaluate needs --cohort')
        evaluate_cohort(args.output,args.cohort)


if __name__=='__main__':main()

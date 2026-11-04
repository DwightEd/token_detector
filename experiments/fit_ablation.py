"""Frozen readout/FIT ablations reusing sealed native span measurements."""

import argparse
from pathlib import Path

import numpy as np

from experiments import data
from experiments.provenance import completed_identity, research_identity
from experiments.span_pilot import load_batches, validated_inputs
from experiments.research.r4 import Detector

# Context, individual observation, calibration, reference population. No label search.
SYSTEMS = {
    'r4_punct': ('punct', 'window', 'quantile', 'task'),
    'r4_dependency': ('direct_mdl', 'window', 'quantile', 'task'),
    'point_punct': ('punct', 'point', 'quantile', 'task'),
    'point_dependency': ('direct_mdl', 'point', 'quantile', 'task'),
    'predictive_point': ('predictive', 'point', 'quantile', 'task'),
    'predictive_no_fit': ('predictive', 'point', 'none', 'none'),
    'predictive_pooled_fit': ('predictive', 'point', 'quantile', 'pooled'),
}
CONTRASTS = {
    'remove_window': ('r4_dependency', 'point_dependency'),
    'segmentation_with_point': ('point_punct', 'predictive_point'),
    'routing_selection': ('point_dependency', 'predictive_point'),
    'add_task_fit': ('predictive_no_fit', 'predictive_point'),
    'pooled_vs_task_fit': ('predictive_point', 'predictive_pooled_fit'),
    'main_vs_r4': ('r4_punct', 'predictive_point'),
}


def verify_seal(root):
    for name, expected in data.read_json(root / 'seal.json').items():
        if data.file_hash(root / name) != expected:
            raise ValueError(f'Sealed artifact changed: {name}')


def score(source, output):
    verify_seal(source)
    plan, inputs = validated_inputs(source)
    identity = research_identity()
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output / 'plan.json', {
        'source_run': str(source.resolve()), 'source_seal_sha256': data.file_hash(source / 'seal.json'),
        'systems': SYSTEMS, 'contrasts': CONTRASTS, 'source': identity, 'label_reads': 0,
        'scope': 'post-result development on the exposed 48-answer pilot; not independent confirmation',
        'no_fit_meaning': 'fixed bounded contrasts, not calibrated hallucination/world posterior',
        'predictive_selection': 'choose coarse or zero-cost cuts by correlated other-head routing gain',
        'fit_population': 'same source-disjoint FIT records; pooled includes FIT only from all tasks',
    })
    batches = load_batches(source, ['punct', 'direct_mdl', 'predictive'])
    files = []
    for task in data.TASKS:
        for name, (method, innovation, calibration, population) in SYSTEMS.items():
            packed = batches[(task, method)]
            dev = data.select_rows(packed, 'dev')
            segmentation = {'punct': 'punct', 'direct_mdl': 'dependency',
                            'predictive': 'dependency_predictive'}[method]
            detector = Detector(task, segmentation=segmentation,
                                innovation=innovation, calibration=calibration)
            if population != 'none':
                fit_tasks = data.TASKS if population == 'pooled' else [task]
                rows = [data.select_rows(batches[(t, method)], 'fit') for t in fit_tasks]
                fit = {field: np.concatenate([r[field] for r in rows]) for field in data.FIELDS}
                detector.fit(fit)
            reference = output / f'{task}_{name}_reference.npz'
            detector.save(reference)
            stages = detector.score_components(dev)
            if not np.array_equal(stages['final'], detector.score(dev)):
                raise ValueError('Diagnostic and callable scorer disagree')
            if name in {'r4_punct', 'r4_dependency'}:
                original = 'punct' if name == 'r4_punct' else 'direct_mdl'
                with np.load(source / 'predictions' / f'{task}_{original}.npz') as old:
                    if not np.array_equal(old['score'], stages['final']):
                        raise ValueError('Default R4 arithmetic changed')
            file = output / f'{task}_{name}.npz'
            with file.open('xb') as stream:
                np.savez_compressed(stream, score=stages.pop('final'), **stages,
                                    **{key: dev[key] for key in ['answer_index', 'target', 'token_id']})
            files.extend([file, reference])
    verify_seal(source)
    validated_inputs(source)
    data.write_json(output / 'complete.json', {'source': completed_identity(identity), 'label_reads': 0,
        'input_records': len(inputs['records']), 'dataset': plan['dataset']})
    data.write_json(output / 'seal.json', {str(f.relative_to(output)): data.file_hash(f)
        for f in [*files, output / 'plan.json', output / 'complete.json']})
    print('SEALED', len(files), 'files; no model calls or label reads', flush=True)


def evaluate(output):
    from experiments.evaluate import official_labels, paired_source_differences, ranking

    verify_seal(output)
    run = data.read_json(output / 'plan.json')
    source = Path(run['source_run'])
    if data.file_hash(source / 'seal.json') != run['source_seal_sha256']:
        raise ValueError('Source run seal changed')
    verify_seal(source)
    plan, inputs = validated_inputs(source)
    metadata = {'records': inputs['records'], 'dataset': plan['dataset'], 'response_root': inputs['cache']}
    results = {}
    for task in data.TASKS:
        scores = {}
        for name in run['systems']:
            with np.load(output / f'{task}_{name}.npz', allow_pickle=False) as raw:
                scores[name] = {key: raw[key] for key in raw.files}
        baseline = scores['r4_punct']
        labels = official_labels(baseline, metadata)
        groups = np.asarray([inputs['records'][i]['source_id'] for i in baseline['answer_index']])
        metrics, diagnostics = {}, {}
        for name, packed in scores.items():
            for key in ['token_id', 'target', 'answer_index']:
                if not np.array_equal(packed[key], baseline[key]):
                    raise ValueError('Original token axes differ')
            metrics[name] = ranking(labels, packed['score'])
            if {'prior', 'source', 'innovation'} <= packed.keys():
                diagnostics[name] = {
                    'raw_prior': ranking(labels, packed['prior']),
                    'calibrated_prior': ranking(labels, packed['source']),
                    'innovation_zero_fraction': float(np.mean(packed['innovation'] == 0)),
                    'source_zero_fraction': float(np.mean(packed['source'] == 0)),
                    'source_one_fraction': float(np.mean(packed['source'] == 1)),
                }
        contrasts = {name: paired_source_differences(labels, scores[a]['score'], scores[b]['score'], groups)
                     for name, (a, b) in run['contrasts'].items()}
        results[task] = {'metrics': metrics, 'diagnostics': diagnostics,
                         'paired_differences': contrasts, 'dev_sources': len(set(groups))}
        print(task, {name: (m['auroc'], m['ap']) for name, m in metrics.items()}, flush=True)
    data.write_json(output / 'evaluation.json', {'scope': run['scope'], 'tasks': results,
        'annotation_sha256': data.file_hash(Path(plan['dataset']) / 'response.jsonl')})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['score', 'evaluate'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.stage == 'score':
        if args.source is None:
            parser.error('score needs --source')
        score(args.source.resolve(), args.output.resolve())
    else:
        evaluate(args.output.resolve())


if __name__ == '__main__':
    main()

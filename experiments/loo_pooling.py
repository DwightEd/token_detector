"""One boundary-objective iteration, reusing the sealed full-evidence pooler."""

import argparse
from pathlib import Path
import shutil
import time

import numpy as np

from experiments import data
from experiments.fit_ablation import evaluate, verify_seal
from experiments.pooling_run import _unit_index
from experiments.provenance import completed_identity, research_identity
from myproject.pooling import fit_prior, grouped_scores, population_spans
from experiments.research.state_partition import routing_partition
from experiments.span_pilot import load_batches, validated_inputs
from myproject.segmentation import legal_atoms

BASELINES = ['raw_full', 'predictive_fixed', 'predictive_fit', 'punct_fixed',
             'punct_fit', 'r4_punct', 'frozen_r4']
CONTRASTS = {
    'boundary_iteration': ('predictive_fixed', 'loo_fixed'),
    'versus_punctuation': ('punct_fixed', 'loo_fixed'),
    'sharing_increment': ('raw_full', 'loo_fixed'),
    'fit_increment': ('loo_fixed', 'loo_fit'),
    'versus_r4': ('r4_punct', 'loo_fit'),
}


def score(previous, output, protocol):
    verify_seal(previous)
    previous_plan = data.read_json(previous / 'plan.json')
    source = Path(previous_plan['source_run'])
    verify_seal(source)
    if data.file_hash(source/'seal.json') != previous_plan['source_seal_sha256']:
        raise ValueError('Previous source binding changed')
    _, inputs = validated_inputs(source)
    identity, start = research_identity(), time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output / 'plan.json', {
        'source_run': str(source.resolve()), 'source_seal_sha256': data.file_hash(source/'seal.json'),
        'previous_scores': str(previous.resolve()), 'previous_seal_sha256': data.file_hash(previous/'seal.json'),
        'source': identity, 'protocol': str(protocol.resolve()), 'protocol_sha256': data.file_hash(protocol),
        'systems': {name: 'retained sealed comparator' for name in BASELINES}
                   | {'loo_fixed': 'LOO routing / Beta(1,1)', 'loo_fit': 'LOO routing / unlabeled Beta FIT'},
        'contrasts': CONTRASTS, 'scope': previous_plan['scope']+'; post-result LOO boundary development',
        'label_reads': 0, 'new_model_calls': 0})
    batches = load_batches(source, methods=['punct'])
    collection = data.read_json(source/'collection.json')
    for record, observed in zip(inputs['records'], collection['records'], strict=True):
        response = data.read_json(Path(inputs['cache'])/record['directory']/'response.json')
        with np.load(source/observed['file'], allow_pickle=False) as raw:
            result = routing_partition(raw['fit_profiles'], raw['hold_profiles'],
                                       raw['available'], legal_atoms(response)[0])
        observed['cuts']['loo'] = result['cuts'].tolist()
        observed['loo_objective'] = result['objective']
    data.write_json(output/'boundaries.json', collection['records'])
    priors = {name: value for name, value in data.read_json(previous/'priors.json').items()
              if name.split('/')[1] in BASELINES}
    files = []
    for task in data.TASKS:
        packed = batches[(task, 'punct')]
        fit, dev = ~packed['development'], packed['development']
        units = _unit_index('loo', packed, collection)
        sources = np.asarray([inputs['records'][a]['source_id'] for a in packed['answer_index']])
        for name in BASELINES:
            file = output/f'{task}_{name}.npz'
            with (previous/file.name).open('rb') as src, file.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            files.append(file)
        for name in ['loo_fixed', 'loo_fit']:
            prior = {'alpha': 1., 'beta': 1., 'fit': False}
            if name == 'loo_fit':
                spans, weights = population_spans(packed['full_contrast'][fit],
                    packed['answer_index'][fit], units[fit], sources[fit])
                prior = fit_prior(spans, weights) | {'fit': True}
            result = grouped_scores(packed['full_contrast'][dev], packed['answer_index'][dev],
                                    units[dev], prior['alpha'], prior['beta'])
            file = output/f'{task}_{name}.npz'
            with file.open('xb') as stream:
                np.savez_compressed(stream, **result,
                    **{k: packed[k][dev] for k in ['answer_index', 'target', 'token_id']})
            files.append(file)
            priors[f'{task}/{name}'] = prior
        print('SCORED', task, flush=True)
    verify_seal(previous)
    verify_seal(source)
    validated_inputs(source)
    data.write_json(output/'priors.json', priors)
    data.write_json(output/'complete.json', {'source': completed_identity(identity),
        'seconds': time.monotonic()-start, 'label_reads': 0, 'new_model_calls': 0})
    data.write_json(output/'seal.json', {str(f.relative_to(output)): data.file_hash(f) for f in
        [*files, output/'plan.json', output/'boundaries.json', output/'priors.json', output/'complete.json']})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('stage', choices=['score', 'evaluate'])
    parser.add_argument('--input', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path)
    args = parser.parse_args()
    if args.stage == 'score':
        if args.input is None or args.protocol is None:
            parser.error('score needs --input and --protocol')
        score(args.input, args.output, args.protocol)
    else:
        evaluate(args.output)


if __name__ == '__main__':
    main()

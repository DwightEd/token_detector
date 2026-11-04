"""Reconstruct archived cases from surviving primitives and official annotations."""

import argparse
from pathlib import Path

import numpy as np

from experiments import data
from experiments.datasets import read_jsonl
from experiments.evaluate import token_labels
from experiments.provenance import completed_identity, research_identity
from experiments.research.pooling import grouped_scores
from experiments.research.r4 import Detector
from experiments.research.measurement import answer_rows


def audit(cache, output, ids):
    identity = research_identity()
    manifest = data.read_json(cache / 'manifest.json')
    records = [r for r in manifest['records'] if str(r['id']) in ids]
    if {str(r['id']) for r in records} != set(ids):
        raise ValueError('Every requested answer must exist in the primitive cache')
    dataset = Path(manifest['dataset'])
    annotations = {str(r['id']): r for r in read_jsonl(dataset/'response.jsonl')
                   if str(r['id']) in ids}
    source_ids = {str(r['source_id']) for r in records}
    sources = {str(r['source_id']): r for r in read_jsonl(dataset/'source_info.jsonl')
               if str(r['source_id']) in source_ids}
    bindings = {str(f.resolve()): data.file_hash(f) for f in
                [cache/'manifest.json', dataset/'response.jsonl', dataset/'source_info.jsonl']}
    cases = []
    for index, record in enumerate(records):
        directory = cache / record['directory']
        response = data.read_json(directory/'response.json')
        annotation = annotations[str(record['id'])]
        if response['text'] != annotation['response']:
            raise ValueError('Prepared and official response text differ')
        files = [directory/f for f in ['response.json', 'with_source.npz', 'without_source.npz']]
        files.append(cache/record['source_file'])
        reference = Path(__file__).parents[1]/'myproject/references'/f'{record["task"]}.npz'
        files.append(reference)
        bindings.update({str(f.resolve()): data.file_hash(f) for f in files})
        with np.load(directory/'with_source.npz') as present, np.load(directory/'without_source.npz') as absent:
            token_id = np.asarray(response['answer_ids'])
            if any(not np.array_equal(z['token_id'], token_id) for z in [present, absent]):
                raise ValueError('Primitive and response token identities differ')
            delta = np.column_stack([absent[k].astype(float)-present[k] for k in ['local', 'full']])
            packed = answer_rows({'delta': delta, 'route': present['raw_route'],
                                  'token_id': token_id}, response, index)
            probabilities = {f'{world}_{view}': z[view].astype(float) for world, z in
                             [('present', present), ('absent', absent)] for view in ['local', 'full']}
        detector = Detector.load(reference)
        stages = detector.score_components(packed)
        if not np.array_equal(stages['final'], detector.score(packed)):
            raise ValueError('Diagnostic and default scorer differ')
        raw_full = delta[packed['target'], 1]
        pooled = grouped_scores(raw_full, packed['answer_index'], packed['unit_index'])
        labels = token_labels(annotation, response)
        rows = []
        for row, target in enumerate(packed['target']):
            start, stop = response['offsets'][target]
            rows.append({'target': int(target), 'start': start, 'end': stop,
                'text': response['text'][start:stop], 'token_id': int(token_id[target]),
                'label': bool(labels[target]), 'unit': int(packed['unit_index'][row]),
                'local_gap': float(delta[target, 0]), 'full_gap': float(raw_full[row]),
                **{k: float(v[target]) for k, v in probabilities.items()},
                **{k: float(v[row]) for k, v in stages.items()},
                'fixed_pool_score': float(pooled['score'][row]),
                'fixed_pool_cavity': float(pooled['cavity_log_odds'][row])})
        cases.append({'record': record, 'source': sources[str(record['source_id'])],
                      'annotation': annotation, 'tokens': rows})
    if any(data.file_hash(Path(name)) != value for name, value in bindings.items()):
        raise ValueError('An input changed during reconstruction')
    result = {'scope': 'post-result selected-case reconstruction; no new detector, population metric or threshold',
              'fn_fp_definition': 'Historical cohort top10% review budget; not inferred from final sign here',
              'comparison': 'existing full-gap Beta(1,1) pooler on the same punctuation spans; no FIT',
              'new_model_calls': 0, 'inputs_sha256': bindings,
              'source': completed_identity(identity), 'cases': cases}
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output/'cases.json', result)
    data.write_json(output/'seal.json', {'cases.json': data.file_hash(output/'cases.json')})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ids', nargs='+', required=True)
    args = parser.parse_args()
    audit(args.cache, args.output, args.ids)


if __name__ == '__main__':
    main()

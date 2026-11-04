"""Inspect sealed transfer errors and test explicit table-binding witnesses on original tokens."""

from pathlib import Path

import numpy as np

from experiments import data
from experiments.datasets import read_jsonl
from experiments.evaluate import ranking, token_labels
from experiments.fit_ablation import verify_seal
from experiments.native_dataset import read_native
from experiments.provenance import completed_identity, research_identity
from experiments.research.table_bindings import scores as binding_scores
from experiments.research.readouts import SharedDetector


def alarm(values):
    mask = np.zeros(len(values), bool)
    mask[np.argsort(-values, kind='stable')[:int(np.ceil(.05*len(values)))]] = True
    return mask


def check_dependencies(native, scores, protocol, expected):
    read_native(native)
    verify_seal(scores)
    if (data.file_hash(native/'seal.json') != expected['native_seal_sha256']
            or data.file_hash(scores/'seal.json') != expected['score_seal_sha256']
            or data.file_hash(protocol) != expected['protocol_sha256']):
        raise ValueError('Diagnostic dependencies changed during execution')
    if expected.get('worlds'):
        from experiments.relational_transfer import read_worlds

        root = Path(expected['worlds'])
        read_worlds(root, native)
        if data.file_hash(root/'seal.json') != expected['worlds_seal_sha256']:
            raise ValueError('Relational source seal changed')


def check_derived(output, records):
    verify_seal(output)
    if any(data.file_hash(output/r['file']) != r['sha256'] for r in records):
        raise ValueError('Derived prediction changed after sealing')


def fit_shared_prior(native, plan, samples, records):
    """Fit one source-balanced prior from the declared unlabeled FIT observations."""
    from experiments.research.prior import fit_prior, population_spans

    blocks = {key: [] for key in ['contrast', 'answer', 'unit', 'source']}
    population = []
    for (group, sample), record in zip(samples, records, strict=True):
        if plan['groups'][group]['role'] != 'fit':
            continue
        with np.load(native/record['file']) as raw:
            contrast = raw['delta'][:, 1]
            for key, value in [('contrast', contrast), ('answer', raw['answer_index']),
                               ('unit', raw['unit_index']),
                               ('source', np.full(len(contrast), sample.source_id))]:
                blocks[key].append(value)
        population.append({'id': sample.id, 'source_id': sample.source_id,
                           'source_sha256': data.text_hash(sample.source),
                           'native_sha256': record['sha256']})
    if not population:
        raise ValueError('Shared prior FIT requires declared training observations')
    spans, weights = population_spans(*(np.concatenate(blocks[key]) for key in blocks))
    return fit_prior(spans, weights) | {'population': population,
        'answers': len(population), 'sources': len({r['source_id'] for r in population}),
        'spans': len(spans), 'tokens': sum(map(len, spans)), 'annotation_reads': 0}


def diagnose(native, scores, output, protocol, constraint_chain=False, method='bindings', worlds=None):
    if method not in {'bindings', 'pooling', 'hierarchy', 'prior_fit', 'relational'} or (method != 'bindings' and constraint_chain):
        raise ValueError('Choose one declared diagnostic method')
    extra_systems = (['shared_score', 'world_shared', 'matched_empty_shared', 'raw_empty_gap', 'matched_native_nll'] if method == 'relational' else
        ['shared_score', 'shared_prior_fit'] if method == 'prior_fit' else
        ['shared_score', 'hierarchical_score', 'atomic_shared'] if method == 'hierarchy' else
        ['cavity_full', 'cavity_pair'] if method == 'pooling' else
        ['binding_conflict', 'atom_absence', 'year_absence']+(
            ['binding_chain', 'binding_no_chain'] if constraint_chain else []))
    plan, samples, observed = read_native(native)
    verify_seal(scores)
    score_plan = data.read_json(scores/'plan.json')
    if constraint_chain and not {'frozen_unary', 'frozen_score'} <= set(score_plan['systems']):
        raise ValueError('The historical constraint-chain diagnostic requires explicit legacy R4 stages')
    if data.file_hash(native/'seal.json') != score_plan['native_seal_sha256']:
        raise ValueError('Score/native binding changed')
    predictions = data.read_json(scores/'complete.json')['records']
    test_samples = {s.id: (g, s) for g, s in samples if plan['groups'][g]['role'] == 'test'}
    if len(predictions) != len(test_samples) or {r['id'] for r in predictions} != set(test_samples):
        raise ValueError('Require complete sealed test predictions')
    by_id = {r['id']: r for r in observed}
    world_records = {}
    if method == 'relational':
        from experiments.relational_transfer import read_worlds

        world_records = {r['id']: r for r in read_worlds(worlds, native)}
        if set(world_records) != set(test_samples):
            raise ValueError('Require complete relational test observations')
    identity = research_identity()
    output.mkdir(parents=True, exist_ok=False)
    prediction_root = output/'predictions'; prediction_root.mkdir()
    metadata = {'native': str(native.resolve()), 'scores': str(scores.resolve()),
        'native_seal_sha256': data.file_hash(native/'seal.json'),
        'score_seal_sha256': data.file_hash(scores/'seal.json'),
        'observation_source': plan['source'], 'analysis_source': identity,
        'protocol': str(protocol.resolve()), 'protocol_sha256': data.file_hash(protocol),
        'constraint_chain': constraint_chain,
        'method': method, 'extra_systems': extra_systems,
        'scope': 'post-result analysis on the complete sealed token cohort; no new method selection'}
    if method == 'relational':
        metadata.update(worlds=str(worlds.resolve()), worlds_seal_sha256=data.file_hash(worlds/'seal.json'))
    data.write_json(output/'plan.json', metadata)
    prior = None
    if method == 'prior_fit':
        prior = fit_shared_prior(native, plan, samples, observed)
        data.write_json(output/'fitted_prior.json', prior)
    derived = []
    for record in predictions:
        group, sample = test_samples[record['id']]
        if record['group'] != group or data.file_hash(scores/record['file']) != record['sha256']:
            raise ValueError('Score record identity changed')
        with np.load(scores/record['file']) as raw, np.load(native/by_id[sample.id]['file']) as original:
            for axis in ['target', 'token_id', 'answer_index', 'unit_index', 'offsets']:
                if not np.array_equal(raw[axis], original[axis]):
                    raise ValueError('Native/predicted original axis changed')
            if method == 'relational':
                from experiments.research.relational import multiworld_scores

                with np.load(worlds/world_records[sample.id]['file']) as world:
                    for key in ['target', 'token_id', 'answer_index', 'unit_index']:
                        if not np.array_equal(world[key], original[key]):
                            raise ValueError('Relational token axis changed')
                    contrast = world['logp'][:, 1:]-world['logp'][:, :1]
                    values = multiworld_scores(contrast, original['answer_index'], original['unit_index'])
                    values.update(shared_score=SharedDetector(sample.task).score(original),
                                  raw_empty_gap=contrast[:, 0], matched_native_nll=-world['logp'][:, 0])
                witnesses = []
            elif method == 'prior_fit':
                from experiments.research.pooling import grouped_scores

                pack = {key: original[key] for key in ['delta', 'target', 'answer_index', 'unit_index']}
                fitted = grouped_scores(pack['delta'][:, 1], pack['answer_index'],
                                        pack['unit_index'], prior['alpha'], prior['beta'])
                values = {'shared_score': SharedDetector(sample.task).score(pack),
                          'shared_prior_fit': fitted['score']}
                witnesses = []
            elif method == 'hierarchy':
                from experiments.research.hierarchy import shared_exploration

                pack = {key: original[key] for key in ['delta', 'target', 'answer_index', 'unit_index']}
                values = shared_exploration(pack, sample.response, original['offsets'], sample.task)
                witnesses = []
            elif method == 'pooling':
                from experiments.research.pooling import grouped_scores
                from myproject.measure import unit_ranges

                starts, stops = unit_ranges(original['answer_index'], original['unit_index'])
                limits = np.repeat(np.log(stops-starts), stops-starts)
                values, witnesses = {}, []
                from experiments.research.r4 import RECIPES, source_values

                pack = {key: original[key] for key in ['delta', 'target', 'answer_index', 'unit_index']}
                point = (raw['point'] if 'point' in raw else
                         source_values(pack['delta'], RECIPES[sample.task].view))
                for key, contrast in [('cavity_full', pack['delta'][:, 1]), ('cavity_pair', point)]:
                    pooled = (SharedDetector(sample.task).score(pack) if key == 'cavity_full' else
                              grouped_scores(contrast, pack['answer_index'], pack['unit_index'])['score'])
                    if np.any(np.abs(pooled-contrast) > limits+1e-10):
                        raise ValueError('Fixed-prior cavity correction violated its preservation bound')
                    values[key] = pooled
            else:
                values, witnesses = binding_scores(sample.source, sample.response, raw['offsets'][raw['target']])
            if constraint_chain:
                from experiments.research.constraints import constraint_log_odds

                offsets = raw['offsets'][raw['target']]
                clauses = []
                for pair in witnesses:
                    if pair['conflict']:
                        endpoints = np.zeros(len(offsets), bool)
                        for a, b in [pair['value_span'], pair['year_span']]:
                            endpoints |= (offsets[:, 0] < b) & (offsets[:, 1] > a)
                        clauses.append(tuple(np.flatnonzero(endpoints)))
                emission = 2*(raw['frozen_unary']-.5)
                penalty = .5/np.diff(raw['target'])
                baseline = constraint_log_odds(emission, penalty, [])
                if not np.array_equal(baseline, raw['frozen_score']):
                    raise ValueError('Constrained model baseline differs from the saved R4 chain')
                values['binding_chain'] = constraint_log_odds(emission, penalty, clauses)
                values['binding_no_chain'] = constraint_log_odds(emission, np.zeros_like(penalty), clauses)
            file = prediction_root/record['file']
            with file.open('xb') as stream:
                np.savez_compressed(stream, **values, target=raw['target'], token_id=raw['token_id'])
        derived.append({'id': sample.id, 'file': str(file.relative_to(output)),
                        'sha256': data.file_hash(file), 'pairs': witnesses})
    check_dependencies(native, scores, protocol, metadata)
    data.write_json(output/'prediction_complete.json', {'records': derived,
        'source': completed_identity(identity), 'annotation_reads': 0})
    data.write_json(output/'seal.json', {p.name: data.file_hash(p) for p in output.iterdir() if p.is_file()})
    check_derived(output, derived)
    # All declared diagnostic score vectors are fixed and sealed before this label join.
    results, cases = {}, []
    derived_by_id = {r['id']: r for r in derived}
    for group, spec in enumerate(plan['groups']):
        if spec['role'] != 'test':
            continue
        prepared = Path(spec['path'])
        manifest = data.read_json(prepared/'manifest.json')
        if data.file_hash(prepared/'annotations.jsonl') != manifest['annotations_sha256']:
            raise ValueError('Annotation provenance changed')
        rows = list(read_jsonl(prepared/'annotations.jsonl'))
        annotations = {a['id']: a for a in rows}
        if len(rows) != len(annotations) or set(annotations) != {s.id for g, s in test_samples.values() if g == group}:
            raise ValueError('Require exact unique annotations')
        blocks, labels, types, lengths = {}, [], [], []
        selected = [r for r in predictions if r['group'] == group]
        for record in selected:
            _, sample = test_samples[record['id']]
            annotation = annotations[sample.id]
            if (annotation['coverage'] != 'complete' or annotation['granularity'] != 'token'
                    or annotation['response_sha256'] != data.text_hash(sample.response)):
                raise ValueError('Complete original token annotation required')
            with np.load(scores/record['file']) as raw, np.load(output/derived_by_id[sample.id]['file']) as extra:
                offsets = raw['offsets'][raw['target']]
                y = token_labels({'labels': annotation['spans']}, {'offsets': offsets})
                labels.append(y); lengths.append(len(y))
                kinds = {s.get('error_type', 'unspecified') for s in annotation['spans']}
                types.append({kind: token_labels({'labels': [s for s in annotation['spans']
                    if s.get('error_type', 'unspecified') == kind]}, {'offsets': offsets}) for kind in kinds})
                for key in score_plan['systems']:
                    blocks.setdefault(key, []).append(raw[key])
                for key in extra_systems:
                    if key in score_plan['systems']:
                        if not np.array_equal(raw[key], extra[key]):
                            raise ValueError('Recomputed primary baseline differs from sealed parent')
                        continue
                    blocks.setdefault(key, []).append(extra[key])
        y = np.concatenate(labels)
        arrays = {k: np.concatenate(v) for k, v in blocks.items()}
        alarms = {key: alarm(v) for key, v in arrays.items()}
        kinds = sorted({kind for row in types for kind in row})
        typed = {}
        for kind in kinds:
            mask = np.concatenate([row.get(kind, np.zeros_like(part)) for row, part in zip(types, labels, strict=True)])
            typed[kind] = {'positives': int(mask.sum()),
                'source_promoted_full': int((mask & (arrays['full_gap'] < 0)).sum()),
                'top5_true_positives': {k: int((mask & a).sum()) for k, a in alarms.items()}}
            if {'frozen_prior', 'frozen_score'} <= alarms.keys():
                typed[kind].update(
                    lost_after_prior=int((mask & alarms['frozen_prior'] & ~alarms['frozen_score']).sum()),
                    recovered_after_prior=int((mask & ~alarms['frozen_prior'] & alarms['frozen_score']).sum()))
        pairs = [p for r in selected for p in derived_by_id[r['id']]['pairs']]
        results[prepared.name] = {'tokens': len(y), 'positives': int(y.sum()), 'types': typed,
            'metrics': {k: ranking(y, v) for k, v in arrays.items()}}
        if method == 'bindings':
            flagged = arrays['binding_conflict'] > 0
            results[prepared.name]['binding'] = {'checked_pairs': len(pairs), 'conflicting_pairs': sum(p['conflict'] for p in pairs),
                'seen_condition_conflicts': sum(p['conflict'] and p['condition_seen_in_table'] for p in pairs),
                'flagged_tokens': int(flagged.sum()), 'true_positive_tokens': int((flagged & y).sum()),
                'false_positive_tokens': int((flagged & ~y).sum())}
        ordered = {k: np.sort(v) for k, v in arrays.items()}
        first = 0
        for record, n, part in zip(selected, lengths, labels, strict=True):
            _, sample = test_samples[record['id']]
            annotation = annotations[sample.id]
            with np.load(scores/record['file']) as raw:
                offsets = raw['offsets'][raw['target']]
                token_rows = []
                for i, (a, b) in enumerate(offsets):
                    token_rows.append({'text': sample.response[a:b], 'start': int(a), 'end': int(b),
                        'label': int(part[i]),
                        'scores': {k: float(v[first+i]) for k, v in arrays.items()},
                        'risk_percentiles': {k: float((np.searchsorted(ordered[k], v[first+i], side='left')+
                            np.searchsorted(ordered[k], v[first+i], side='right'))/(2*len(y)))
                                             for k, v in arrays.items()},
                        'top5': {k: bool(v[first+i]) for k, v in alarms.items()}})
            cases.append({'id': sample.id, 'component': prepared.name, 'source': sample.source,
                'response': sample.response, 'spans': annotation['spans'], 'tokens': token_rows,
                'binding_pairs': derived_by_id[sample.id]['pairs']})
            first += n
    check_dependencies(native, scores, protocol, metadata)
    check_derived(output, derived)
    data.write_json(output/'cases.json', {'cases': cases, 'scope': 'post-result error inspection, not new test'})
    data.write_json(output/'evaluation.json', {'groups': results, 'source': completed_identity(identity)})
    print({g: {key: r['metrics'][key] for key in extra_systems} for g, r in results.items()})

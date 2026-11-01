"""Independent read-only numerical audit of the frozen 20261028 delta trial.

Does not import experiment scorers, use GPU, or change any experiment artifacts.
Run from token_detector; writes only its unique forensic trace JSON.
"""

from collections import Counter
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp, roots_legendre


ROOT = Path('outputs/delta_local_20261028')
TRACE = Path('.aris/traces/experiment-audit/20261028_delta_local')
COUNTS = Counter()
ERRORS = {}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(key, actual, expected, tolerance=2e-11):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape, (key, actual.shape, expected.shape)
    error = float(np.max(abs(actual.astype(float)-expected.astype(float)))) if actual.size else 0.
    ERRORS[key] = max(ERRORS.get(key, 0.), error)
    assert error <= tolerance, (key, error, tolerance)
    COUNTS[key] += 1


def verify(root):
    for name, wanted in read(root/'seal.json').items():
        assert digest(root/name) == wanted, (root, name)
        COUNTS['seal_entries'] += 1


def boundaries(answer, unit):
    changes = (answer[1:] != answer[:-1]) | (unit[1:] != unit[:-1])
    cuts = np.r_[0, np.flatnonzero(changes)+1, len(answer)]
    return list(zip(cuts[:-1], cuts[1:]))


def geometry(values, answer, unit, raw=None):
    """Independent prefix/suffix sums avoid the author's own-row subtraction."""
    norms = np.linalg.norm(values, axis=1)
    valid = norms > 1e-12
    if raw is not None:
        valid &= raw
    normalized = np.zeros_like(values, dtype=float)
    normalized[valid] = values[valid]/norms[valid, None]
    score, available = np.ones(len(values)), np.zeros(len(values), bool)
    for lo, hi in boundaries(answer, unit):
        part = normalized[lo:hi]
        zero = np.zeros(part.shape[1])
        prefix = np.vstack([zero, np.cumsum(part, axis=0)])
        suffix = np.vstack([np.cumsum(part[::-1], axis=0)[::-1], zero])
        other = prefix[:-1]+suffix[1:]
        length = np.linalg.norm(other, axis=1)
        good = valid[lo:hi] & (length > 1e-12)
        cosine = np.zeros(len(part))
        cosine[good] = np.einsum('ij,ij->i', part[good], other[good])/length[good]
        score[lo:hi][good] = 1-np.clip(cosine[good], -1, 1)
        available[lo:hi] = good
    return score, available


@lru_cache(None)
def quadrature(length):
    nodes, weights = roots_legendre((length+2)//2)
    theta = (nodes+1)/2
    return np.log(theta), np.log1p(-theta), np.log(weights/2)


def beta_scores(gap, answer, unit):
    """Independent Gauss integration of the uniform-Beta cavity polynomial."""
    result = np.empty(len(gap))
    for lo, hi in boundaries(answer, unit):
        values = gap[lo:hi]
        lt, lc, lw = quadrature(len(values))
        factor = np.logaddexp(lc[:, None], lt[:, None]+values)
        others = factor.sum(1)[:, None]-factor
        result[lo:hi] = (values+logsumexp(lw[:, None]+lt[:, None]+others, axis=0)
                         - logsumexp(lw[:, None]+lc[:, None]+others, axis=0))
    return result


def metric(labels, score, weights=None, include_counts=True):
    """Tie-grouped weighted rank statistics, independent of sklearn evaluator."""
    labels, score = np.asarray(labels, bool), np.asarray(score)
    weights = np.ones(len(labels), float) if weights is None else np.asarray(weights, float)
    order = np.lexsort((np.arange(len(score)), -score))
    y, s, weights = labels[order], score[order], weights[order]
    ends = np.r_[np.flatnonzero(s[:-1] != s[1:])+1, len(s)]
    starts = np.r_[0, ends[:-1]]
    positive = np.add.reduceat(weights*y, starts)
    negative = np.add.reduceat(weights*~y, starts)
    npos, nneg = positive.sum(), negative.sum()
    cumulative_pos, cumulative_neg = np.cumsum(positive), np.cumsum(negative)
    auroc = float(np.sum(positive*(nneg-cumulative_neg+.5*negative))/(npos*nneg)) if npos and nneg else None
    total = cumulative_pos+cumulative_neg
    precision = np.divide(cumulative_pos, total, out=np.zeros_like(total, dtype=float), where=total > 0)
    ap = float(np.sum(positive*precision)/npos) if npos else None
    if not include_counts:
        return auroc, ap
    alarms = order[:int(np.ceil(.05*len(labels)))]
    tp = int(labels[alarms].sum())
    return {'tokens': len(labels), 'errors': int(labels.sum()), 'auroc': auroc, 'ap': ap,
            'top5pct': {'tp': tp, 'fp': len(alarms)-tp, 'fn': int(labels.sum())-tp}}


def compare_metric(actual, expected):
    for key in ['tokens', 'errors', 'top5pct']:
        assert actual[key] == expected[key], (key, actual[key], expected[key])
    for key in ['auroc', 'ap']:
        if actual[key] is None:
            assert expected[key] is None
        else:
            compare('metrics_'+key, actual[key], expected[key], 5e-15)


def whitening(delta, answer, unit, sources):
    """Explicit source -> answer -> span -> retained-token weight construction."""
    weights = np.zeros(len(delta))
    unique_sources = np.unique(sources)
    for source in unique_sources:
        answers = np.unique(answer[sources == source])
        for ans in answers:
            spans = np.unique(unit[answer == ans])
            for span in spans:
                mask = (answer == ans) & (unit == span)
                weights[mask] = 1/(len(unique_sources)*len(answers)*len(spans)*mask.sum())
    compare('weight_sum', weights.sum(), 1.)
    mean = np.average(delta, axis=0, weights=weights)
    constant = np.all(delta == delta[0], axis=0)
    mean[constant] = delta[0, constant]
    variance = np.average((delta-mean)**2, axis=0, weights=weights)
    scale = np.where(~constant & (variance > 0), np.sqrt(variance), 1.)
    return mean, scale


def load_native(plan, inputs, collection):
    cache = Path(inputs['cache'])
    for file, wanted in inputs['input_hashes'].items():
        assert digest(cache/file) == wanted
        COUNTS['primitive_hashes'] += 1
    native, responses, gaps = {}, {}, {'present': [], 'absent': [], 'contrast': []}
    observed = {row['id']: row for row in collection['records']}
    assert len(observed) == len(collection['records']) == len(inputs['records']) == 196
    for entry in inputs['records']:
        record = entry['record']
        row = observed[record['id']]
        assert digest(ROOT/row['file']) == row['sha256']
        COUNTS['native_file_hashes'] += 1
        response = read(cache/record['directory']/'response.json')
        responses[record['id']] = response
        with np.load(ROOT/row['file'], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        native[record['id']] = arrays
        assert np.array_equal(arrays['token_id'], response['answer_ids'])
        length = len(arrays['token_id'])
        assert length == row['tokens']
        COUNTS['native_tokens'] += length
        old_logp = {}
        for world, filename in [('present', 'with_source.npz'), ('absent', 'without_source.npz')]:
            hidden = arrays[f'{world}_hidden']
            assert hidden.shape == (length, 4096) and hidden.dtype == np.float32
            assert np.isfinite(hidden).all() and not np.any(hidden.view(np.uint32) & 65535)
            COUNTS['bf16_exact_hidden_elements'] += hidden.size
            for key in ['logp', 'entropy', 'rank']:
                assert arrays[f'{world}_{key}'].shape == (length,)
                assert np.isfinite(arrays[f'{world}_{key}']).all()
            assert np.all(arrays[f'{world}_logp'] <= 0)
            assert np.all(arrays[f'{world}_entropy'] >= 0)
            assert np.all(arrays[f'{world}_rank'] >= 0)
            with np.load(cache/record['directory']/filename) as old:
                assert np.array_equal(old['token_id'], arrays['token_id'])
                old_logp[world] = old['full'].astype(float)
            difference = arrays[f'{world}_logp'].astype(float)-old_logp[world]
            compare('stored_logp_difference', [np.mean(abs(difference)), max(abs(difference))],
                    [row['cached_logp_difference'][world][key] for key in ['mean_abs', 'max_abs']])
            gaps[world].extend(difference.tolist())
        difference = (arrays['absent_logp'].astype(float)-arrays['present_logp']
                      -(old_logp['absent']-old_logp['present']))
        gaps['contrast'].extend(difference.tolist())
    summaries = {key: {'mean_abs': float(np.mean(abs(np.asarray(values)))),
                       'max_abs': float(max(abs(np.asarray(values)))),
                       'nonzero': int(np.count_nonzero(values))}
                 for key, values in gaps.items()}
    return native, responses, summaries


def task_rows(task, records, source, source_collection, responses, native, annotations):
    rows = []
    for answer, record in enumerate(records):
        if record['task'] != task:
            continue
        response = responses[record['id']]
        arrays = native[record['id']]
        offsets = np.asarray(response['offsets'])
        target = np.flatnonzero((offsets[:, 1] > offsets[:, 0])
                               & ~np.isin(arrays['token_id'], response['special_ids']))
        annotation = annotations[record['id']]
        assert annotation['response'] == response['text']
        assert str(annotation['source_id']) == str(record['source_id'])
        labels = np.array([any(offsets[t, 0] < span['end'] and offsets[t, 1] > span['start']
                               for span in annotation['labels']) for t in target], bool)
        with np.load(source/source_collection[answer]['file']) as old:
            old_gap = old['absent_full_logp'].astype(float)-old['source_full_logp']
        length = len(target)
        row = {'answer': np.full(length, answer), 'target': target,
               'token': arrays['token_id'][target], 'fit': np.full(length, record['partition'] == 'fit'),
               'source': np.full(length, record['source_id']), 'label': labels,
               'old_gap': old_gap[target], 'present': arrays['present_hidden'][target].astype(float),
               'absent': arrays['absent_hidden'][target].astype(float),
               'native_gap': arrays['absent_logp'][target].astype(float)-arrays['present_logp'][target],
               'nll': -arrays['present_logp'][target].astype(float),
               'entropy': arrays['present_entropy'][target].astype(float)}
        for method, key in [('punct', 'punct'), ('dependency', 'direct_mdl')]:
            cuts = source_collection[answer]['cuts'][key]
            assert cuts == read(ROOT/'collection.json')['records'][
                next(i for i, entry in enumerate(read(ROOT/'inputs.json')['records'])
                     if entry['record']['id'] == record['id'])]['cuts'][key]
            row[method] = np.searchsorted(cuts[1:], target, side='right')
        rows.append(row)
    return {key: np.concatenate([row[key] for row in rows]) for key in rows[0]}


def diagnostics(labels, packed):
    answer = packed['answer_index']
    answers = [answer == a for a in np.unique(answer)]
    mixed = [mask for mask in answers if 0 < labels[mask].sum() < mask.sum()]
    metrics = [metric(labels[mask], packed['score'][mask]) for mask in mixed]
    available = packed.get('available', np.ones(len(labels), bool))
    result = {'tokens': len(labels), 'available': int(available.sum()),
              'unavailable_positive': int(labels[~available].sum()),
              'macro_within_answer': {key: float(np.mean([value[key] for value in metrics]))
                                     if metrics else None for key in ['auroc', 'ap']}}
    if 'unit_index' in packed:
        sizes = np.array([stop-start for start, stop in boundaries(answer, packed['unit_index'])])
        result['span_sizes'] = {'one': int(sum(sizes == 1)), 'two': int(sum(sizes == 2)),
                                'above_two': int(sum(sizes > 2))}
    return result, len(mixed), len(answers)


def bootstrap(labels, packed, groups, contrasts):
    unique, index = np.unique(groups, return_inverse=True)
    draws = np.random.default_rng(17).integers(0, len(unique), (1000, len(unique)))
    weights = np.array([np.bincount(draw, minlength=len(unique)) for draw in draws])
    needed = {system for pair in contrasts.values() for system in pair}
    values = {system: np.array([metric(labels, packed[system]['score'], row[index], False)
                                for row in weights], dtype=float) for system in needed}
    result = {}
    for name, (baseline, candidate) in contrasts.items():
        row = {'seed': 17, 'repetitions': 1000, 'unit': 'source_id'}
        for column, key in enumerate(['auroc', 'ap']):
            differences = values[candidate][:, column]-values[baseline][:, column]
            differences = differences[np.isfinite(differences)]
            row[key] = {'defined': len(differences), 'degenerate': 1000-len(differences),
                        'interval95': np.quantile(differences, [.025, .975]).tolist()
                        if len(differences) else None}
        result[name] = row
    return result


def audit_task(task, records, source, source_collection, output, previous,
               responses, native, annotations, plan, evaluation, saved_diagnostics):
    rows = task_rows(task, records, source, source_collection, responses, native, annotations)
    fit, answer = rows['fit'], rows['answer']
    dev = ~fit
    delta = rows['present']-rows['absent']
    raw = np.linalg.norm(delta, axis=1) > 1e-12
    readouts = {key: rows[key] for key in ['native_gap', 'old_gap', 'nll', 'entropy']}
    readouts['minus_delta_norm'] = -np.linalg.norm(delta, axis=1)
    readouts['excess_surprisal'] = rows['nll']-rows['entropy']
    available = {}
    for system, values in [('source_direction_punct', rows['present']),
                            ('absent_direction_punct', rows['absent'])]:
        readouts[system], available[system] = geometry(values, answer, rows['punct'])
    for method in ['punct', 'dependency']:
        units = rows[method]
        readouts['direction_'+method], available['direction_'+method] = geometry(delta, answer, units)
        mean, scale = whitening(delta[fit], answer[fit], units[fit], rows['source'][fit])
        with np.load(output/f'{task}_{method}_whitening.npz') as parameters:
            compare('FIT_mean', mean, parameters['mean'])
            compare('FIT_scale', scale, parameters['scale'])
        system = 'whitened_direction_'+method
        readouts[system], available[system] = geometry((delta-mean)/scale, answer, units, raw)
        readouts['native_pool_'+method] = beta_scores(rows['native_gap'], answer, units)
    axis = {'answer_index': answer[dev], 'target': rows['target'][dev], 'token_id': rows['token'][dev]}
    labels, groups = rows['label'][dev], rows['source'][dev]
    packed, metrics, diagnostic_values = {}, {}, {}
    for system in plan['systems']:
        file = output/f'{task}_{system}.npz'
        with np.load(file, allow_pickle=False) as archive:
            saved = {key: archive[key] for key in archive.files}
        packed[system] = saved
        for key, value in axis.items():
            assert np.array_equal(value, saved[key]), (task, system, key)
            COUNTS['axis_arrays'] += 1
        if system in ['r4_punct', 'frozen_r4']:
            assert digest(file) == digest(previous/file.name)
            COUNTS['R4_byte_copies'] += 1
        else:
            compare('score_'+system, readouts[system][dev], saved['score'])
            COUNTS['recomputed_score_rows'] += len(labels)
        if system in available:
            assert np.array_equal(available[system][dev], saved['available'])
            COUNTS['availability_masks'] += 1
        metrics[system] = metric(labels, saved['score'])
        compare_metric(metrics[system], evaluation['metrics'][system])
        COUNTS['metric_systems'] += 1
        value, mixed_answers, total_answers = diagnostics(labels, saved)
        diagnostic_values[system] = value
        expected = saved_diagnostics['systems'][system]
        assert value['tokens'] == expected['tokens']
        assert value['available'] == expected['available']
        assert value['unavailable_positive'] == expected['unavailable_positive']
        if 'span_sizes' in value:
            assert value['span_sizes'] == expected['span_sizes']
        for key in ['auroc', 'ap']:
            a, b = value['macro_within_answer'][key], expected['macro_within_answer'][key]
            if a is None:
                assert b is None
            else:
                compare('within_answer_'+key, a, b, 5e-15)
    assert mixed_answers == saved_diagnostics['mixed_answers']
    assert total_answers == saved_diagnostics['total_answers']
    intervals = bootstrap(labels, packed, groups, plan['contrasts'])
    for name, row in intervals.items():
        expected = evaluation['paired_differences'][name]
        assert all(row[key] == expected[key] for key in ['seed', 'repetitions', 'unit'])
        for key in ['auroc', 'ap']:
            assert row[key]['defined'] == expected[key]['defined']
            assert row[key]['degenerate'] == expected[key]['degenerate']
            if row[key]['interval95'] is None:
                assert expected[key]['interval95'] is None
            else:
                compare('bootstrap_'+key, row[key]['interval95'], expected[key]['interval95'], 5e-15)
        COUNTS['paired_bootstrap_comparisons'] += 1
    print('RECOMPUTED', output.name, task, len(labels), int(labels.sum()),
          'primary', metrics['direction_dependency'], 'R4', metrics['r4_punct'], flush=True)
    return {'metrics': metrics, 'diagnostics': diagnostic_values, 'bootstrap': intervals,
            'dev_sources': len(np.unique(groups)), 'dev_answers': len(np.unique(answer[dev])),
            'fit_sources': len(np.unique(rows['source'][fit])),
            'fit_answers': len(np.unique(answer[fit])), 'valid_DEV_tokens': len(labels),
            'positive_DEV_tokens': int(labels.sum()), 'all_original_token_axes_retained': True}


def audit_cases(inputs, native, responses):
    saved = read(ROOT/'case_readouts'/'scores.json')['cases']
    assert {row['id'] for row in saved} == {'3182', '3180', '10988', '2390'}
    observed = {row['id']: row for row in read(ROOT/'collection.json')['records']}
    result = []
    for case in saved:
        arrays, response = native[case['id']], responses[case['id']]
        offsets = np.asarray(response['offsets'])
        target = np.flatnonzero((offsets[:, 1] > offsets[:, 0])
                               & ~np.isin(arrays['token_id'], response['special_ids']))
        assert np.array_equal(target, case['target'])
        assert np.array_equal(arrays['token_id'][target], case['token_id'])
        assert np.array_equal(offsets[target], case['offsets'])
        delta = arrays['present_hidden'][target].astype(float)-arrays['absent_hidden'][target]
        compare('case_minus_delta_norm', -np.linalg.norm(delta, axis=1), case['scores']['minus_delta_norm'])
        compare('case_native_gap', arrays['absent_logp'][target].astype(float)
                - arrays['present_logp'][target], case['scores']['native_gap'])
        for method, key in [('punct', 'punct'), ('dependency', 'direct_mdl')]:
            units = np.searchsorted(observed[case['id']]['cuts'][key][1:], target, side='right')
            score, _ = geometry(delta, np.zeros(len(target), int), units)
            compare('case_direction_'+method, score, case['scores']['direction_'+method])
        result.append({'id': case['id'], 'source_id': case['source_id'], 'tokens': len(target)})
    return result


def main():
    TRACE.mkdir(parents=True, exist_ok=True)
    plan, inputs, collection = read(ROOT/'plan.json'), read(ROOT/'inputs.json'), read(ROOT/'collection.json')
    complete = read(ROOT/'score_complete.json')
    verify(ROOT)
    for name, sha in complete['prediction_seals'].items():
        output = ROOT/f'scores_{name}'
        verify(output)
        assert digest(output/'seal.json') == sha
    verify(ROOT/'case_readouts')
    assert digest(ROOT/'case_readouts'/'seal.json') == complete['case_seal_sha256']
    assert digest(ROOT/'seal.json') == complete['state_seal_sha256']
    prelaunch_source = read(ROOT/'prelaunch.json')['source']
    assert collection['source'] == complete['source'] == prelaunch_source
    identity_mismatches = []
    for name, wanted in prelaunch_source['source_sha256'].items():
        file = (Path('experiments/research')/name.split('/', 1)[1]
                if name.startswith('experiments.research/') else Path(name))
        assert digest(file) == wanted, name
        COUNTS['prelaunch_source_bindings'] += 1
        if plan['source']['source_sha256'].get(name) != wanted:
            identity_mismatches.append({'file': name, 'prepare_sha256':
                                       plan['source']['source_sha256'].get(name), 'prelaunch_sha256': wanted})
    native, responses, gap_summary = load_native(plan, inputs, collection)
    annotation_path = Path(plan['dataset'])/'response.jsonl'
    annotation_sha256 = digest(annotation_path)
    assert annotation_sha256 == 'e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073'
    annotations = {str(row['id']): row for row in map(json.loads, annotation_path.read_text().splitlines())}
    tasks, cohorts, source_sets = {}, {}, []
    input_hashes = {str(file): digest(file) for file in [ROOT/'plan.json', ROOT/'inputs.json',
                    ROOT/'collection.json', ROOT/'score_complete.json', ROOT/'prelaunch.json',
                    ROOT/'final_prelaunch_review.json', ROOT/'seal.json']}
    for name, cohort in plan['cohorts'].items():
        source, previous, output = Path(cohort['source']), Path(cohort['scores']), ROOT/f'scores_{name}'
        records = read(source/'inputs.json')['records']
        source_collection = read(source/'collection.json')['records']
        verify(source)
        verify(previous)
        fits = {row['source_id'] for row in records if row['partition'] == 'fit'}
        devs = {row['source_id'] for row in records if row['partition'] == 'dev'}
        assert not fits & devs
        source_sets.append(fits | devs)
        assert read(output/'plan.json')['source'] == read(output/'complete.json')['source'] == prelaunch_source
        evaluation, diagnostics_saved = read(output/'evaluation.json'), read(output/'diagnostics.json')
        assert evaluation['annotation_sha256'] == annotation_sha256
        for file in [output/'plan.json', output/'complete.json', output/'seal.json',
                     output/'evaluation.json', output/'diagnostics.json']:
            input_hashes[str(file)] = digest(file)
        tasks[name] = {task: audit_task(task, records, source, source_collection, output, previous,
                      responses, native, annotations, plan, evaluation['tasks'][task],
                      diagnostics_saved['tasks'][task]) for task in ['QA', 'Summary', 'Data2txt']}
        cohorts[name] = {'answers': len(records), 'FIT_sources': len(fits), 'DEV_sources': len(devs)}
    assert not source_sets[0] & source_sets[1]
    cases = audit_cases(inputs, native, responses)
    report = {'command': 'OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 '
              '/share/home/tm902089733300000/a903202310/lys/conda_envs/research/bin/python '
              'experiments/delta_local_integrity_20261028.py',
              'script_sha256': digest(Path(__file__)), 'counts': dict(COUNTS), 'max_abs_errors': ERRORS,
              'native_minus_primitive': gap_summary, 'tasks': tasks, 'cases': cases,
              'cohorts': cohorts, 'annotation_sha256': annotation_sha256,
              'prepare_vs_prelaunch_identity_differences': identity_mismatches,
              'audited_input_hashes': input_hashes}
    with (TRACE/'independent_recomputation.json').open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    print('AUDIT NUMERICS COMPLETE', dict(COUNTS))
    print('MAX ERRORS', ERRORS)


if __name__ == '__main__':
    main()

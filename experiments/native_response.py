"""Post-result native-context response and source-paired localization diagnostics."""

from pathlib import Path

import numpy as np

from experiments import data, fullgap_dataset
from experiments.datasets import _check_spans, read_jsonl, sentence_token_labels
from experiments.evaluate import token_labels
from experiments.fit_ablation import verify_seal
from experiments.local_evaluation import paired_span_differences
from experiments.native_dataset import read_native
from experiments.prevalence_probe import influence, select_span_candidates, span_candidates
from experiments.provenance import completed_identity, research_identity
from myproject.detector import unit_ranges
from myproject.sequence import witness_emissions
from myproject.shared import SharedDetector


METHODS = ('witness', 'matched_local', 'symmetric_local', 'fixed_local', 'fixed_persistence')
AXES = ('target', 'token_id', 'answer_index', 'unit_index')
TOLERANCE = 1e-8
DELTA = .01


def _records(native, scores):
    """Authenticate both supported formats before returning minimal token records."""
    score_plan = data.read_json(scores/'plan.json')
    generic = 'methods' in score_plan
    if generic:
        plan, samples, observed, methods = fullgap_dataset._verify_scores(native, scores)
        indexed = [(index, 0, sample, record) for index, (sample, record) in
                   enumerate(zip(samples, observed, strict=True))]
    else:
        verify_seal(scores)
        plan, samples, observed = read_native(native)
        if (Path(score_plan['native']).resolve() != native
                or data.file_hash(native/'seal.json') != score_plan['native_seal_sha256']):
            raise ValueError('Score/native binding changed')
        predictions = data.read_json(scores/'complete.json')['records']
        predicted = {row['id']: row for row in predictions}
        indexed = [(index, group, sample, record) for index, ((group, sample), record) in
                   enumerate(zip(samples, observed, strict=True)) if plan['groups'][group]['role'] == 'test']
        if len(predicted) != len(predictions) or set(predicted) != {sample.id for _, _, sample, _ in indexed}:
            raise ValueError('Require complete, unique TEST score records')
        methods = [method for method in METHODS if f'{method}_score' in score_plan['systems']]
    methods = [method for method in METHODS if method in methods]
    if 'witness' not in methods:
        raise ValueError('Sealed witness scores are required')
    records = []
    for index, group, sample, record in indexed:
        with np.load(native/record['file'], allow_pickle=False) as saved:
            raw = {name: saved[name] for name in saved.files}
        axes = {name: raw[name] for name in AXES}
        offsets = raw['offsets'][raw['target']]
        gap = (raw['logp'][raw['target'], 1]-raw['logp'][raw['target'], 0] if generic
               else raw['delta'][:, 1])
        vectors = {}
        for method in methods:
            if generic:
                prediction = data.read_json(data.prediction_path(scores/method/'predictions', sample.id))
                tokens = prediction['tokens']
                if (len(tokens) != len(gap) or any((token['token_id'], token['start'], token['end']) !=
                        (int(axes['token_id'][i]), int(offsets[i, 0]), int(offsets[i, 1]))
                        for i, token in enumerate(tokens))):
                    raise ValueError('Full-gap JSON original token axis changed')
                vectors[method] = np.array([token['score'] for token in tokens])
            else:
                prediction = predicted[sample.id]
                if prediction['group'] != group or prediction['response_sha256'] != data.text_hash(sample.response):
                    raise ValueError('Score response/group identity changed')
                path = scores/prediction['file']
                if data.file_hash(path) != prediction['sha256']:
                    raise ValueError('Score record hash changed')
                with np.load(path, allow_pickle=False) as saved:
                    if (any(not np.array_equal(saved[name], raw[name]) for name in (*AXES, 'offsets'))
                            or not np.array_equal(saved['full_gap'], gap)):
                        raise ValueError('Native score original axis or full gap changed')
                    vectors[method] = saved[f'{method}_score']
        if not np.isfinite(gap).all() or any(v.shape != gap.shape or not np.isfinite(v).all() for v in vectors.values()):
            raise ValueError('Finite original-token gaps and score vectors required')
        prepared = Path(plan['prepared'] if generic else plan['groups'][group]['path'])
        records.append({'sample_index': index, 'sample': sample, 'prepared': prepared,
                        'cohort': f"{plan.get('dataset', 'fullgap')}:{sample.task}" if generic else prepared.name,
                        'gap': gap, 'offsets': offsets, 'scores': vectors, **axes})
    return plan, score_plan, methods, records


def _begin(native, scores, protocol, output, label_reads):
    native, scores, protocol, output = map(lambda path: Path(path).resolve(), (native, scores, protocol, output))
    if output.exists():
        raise FileExistsError('Diagnostic output already exists')
    identity = research_identity()
    plan, score_plan, methods, records = _records(native, scores)
    bindings = {str(path): data.file_hash(path) for path in (native/'seal.json', scores/'seal.json', protocol)}
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output/'plan.json', {'native': str(native), 'scores': str(scores), 'protocol': str(protocol),
        'input_bindings': bindings, 'observation_source': plan['source'], 'scorer_source': score_plan['source'],
        'analysis_source': identity, 'methods': methods, 'annotation_reads': label_reads,
        'new_model_calls': 0, 'delta_emission': DELTA, 'numeric_tolerance': TOLERANCE,
        'scope': 'post-result explanatory diagnostic; no independent confirmation or causal source edit'})
    return native, scores, output, identity, bindings, methods, records


def _finish(native, scores, output, identity, bindings, summary, extra_files=()):
    _records(native, scores)
    if any(data.file_hash(Path(path)) != expected for path, expected in bindings.items()):
        raise ValueError('Diagnostic input changed during execution')
    summary['verified_input_bindings'] = bindings
    summary['analysis_source'] = completed_identity(identity)
    data.write_json(output/'summary.json', summary)
    names = ['plan.json', 'summary.json', *extra_files]
    data.write_json(output/'seal.json', {name: data.file_hash(output/name) for name in names})
    return summary


def _certificates(record, methods):
    emission = witness_emissions(record['gap'])
    arrays = {'full_gap': record['gap'], 'emission': emission, 'offsets': record['offsets'],
              **{name: record[name] for name in AXES}}
    starts, stops = unit_ranges(record['answer_index'], record['unit_index'])
    magnitude = np.empty(len(emission))
    linear = {method: np.empty(len(emission)) for method in methods}
    for a, b in zip(starts, stops, strict=True):
        e = emission[a:b]
        magnitude[a:b] = np.abs(e).sum()-np.abs(e)
        distance = np.abs(np.arange(b-a)[:, None]-np.arange(b-a)[None, :])
        for method in methods:
            linear[method][a:b] = influence(distance, method) @ e-e
    arrays['context_l1'] = magnitude
    counts = {}
    for method in methods:
        score = record['scores'][method]
        lower, upper = emission+linear[method]-magnitude**2/8, emission+linear[method]+magnitude**2/8
        positive, negative = lower > 0, upper < 0
        reversal = (emission > 0) & (score < 0)
        arrays.update({f'{method}_{key}': value for key, value in
                       [('score', score), ('linear', linear[method]), ('lower', lower), ('upper', upper)]})
        counts[method] = {'tokens': len(score), 'positive_certificates': int(positive.sum()),
            'negative_certificates': int(negative.sum()), 'certified': int((positive | negative).sum()),
            'positive_own_reversals': int(reversal.sum()), 'certified_positive_own_reversals': int((reversal & negative).sum()),
            'bound_violations': int(((score < lower-TOLERANCE) | (score > upper+TOLERANCE)).sum())}
    return arrays, counts


def probe_native(native, scores, protocol, output):
    native, scores, output, identity, bindings, methods, records = _begin(native, scores, protocol, output, 0)
    candidates, totals, files, certificate_records = [], {}, [], []
    for record in records:
        sample = record['sample']
        candidates.extend(span_candidates(record['sample_index'], sample.source_id, sample.id,
                                           record['answer_index'], record['unit_index']))
        arrays, counts = _certificates(record, methods)
        file = Path('certificates')/f'{data.text_hash(sample.id)}.npz'
        (output/file.parent).mkdir(exist_ok=True)
        with (output/file).open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        files.append(str(file))
        certificate_records.append({'id': sample.id, 'source_id': sample.source_id,
            'cohort': record['cohort'], 'file': str(file), 'sha256': data.file_hash(output/file)})
        for method, values in counts.items():
            total = totals.setdefault(record['cohort'], {}).setdefault(method, dict.fromkeys(values, 0))
            for key, value in values.items():
                total[key] += value
    selected, eligible = select_span_candidates(candidates)
    lookup, cases = {record['sample_index']: record for record in records}, []
    for choice in selected:
        record = lookup[choice['sample_index']]
        rows = slice(choice['start'], choice['stop'])
        gap = record['gap'][rows]
        target = len(gap)//2
        far = 0 if target >= len(gap)-1-target else len(gap)-1
        axes = {name: record[name][rows] for name in AXES}
        responses = []
        for role, donor in [('near_left', target-1), ('far_endpoint', far)]:
            changed = gap.copy()
            x = witness_emissions(gap)[donor]+DELTA+np.log(2.)
            changed[donor] = x+np.log(-np.expm1(-x))  # stable inverse, including huge positive gaps
            for method in methods:
                before = float(record['scores'][method][rows][target])
                after = float(SharedDetector(record['sample'].task, method=method).score_full_gap(
                    changed, axes['answer_index'], axes['unit_index'])[target])
                secant = (after-before)/DELTA
                responses.append({'method': method, 'donor_role': role, 'donor_position': donor,
                    'donor_target': int(axes['target'][donor]), 'donor_token_id': int(axes['token_id'][donor]),
                    'distance': abs(donor-target), 'before': before, 'after': after, 'secant': secant,
                    'neutral_influence': float(influence(abs(donor-target), method)),
                    'absolute_bound_violation': bool(abs(secant) > 1+TOLERANCE)})
        cases.append({**choice, 'id': record['sample'].id, 'source_id': record['sample'].source_id,
            'full_gap': gap.tolist(), 'target_position': target,
            'axes': {name: value.tolist() for name, value in axes.items()},
            'offsets': record['offsets'][rows].tolist(), 'responses': responses})
    for cohort in totals.values():
        for values in cohort.values():
            values['certificate_fraction'] = values['certified']/values['tokens'] if values['tokens'] else None
    responses = [row for case in cases for row in case['responses']]
    data.write_json(output/'cases.json', {'cases': cases, 'eligible_spans': eligible, 'selected_spans': len(cases)})
    data.write_json(output/'collection.json', {'records': certificate_records})
    summary = {'cohorts': totals, 'selected_spans': len(cases), 'eligible_spans': eligible,
        'negative_secants': sum(row['secant'] < 0 for row in responses),
        'absolute_bound_violations': sum(row['absolute_bound_violation'] for row in responses),
        'max_absolute_secant': max((abs(row['secant']) for row in responses), default=0.),
        'responses_by_method_role': {f'{method}:{role}': {
            'count': sum(row['method'] == method and row['donor_role'] == role for row in responses),
            'negative': sum(row['method'] == method and row['donor_role'] == role and row['secant'] < 0 for row in responses)}
            for method in methods for role in ('near_left', 'far_endpoint')},
        'annotation_reads': 0, 'new_model_calls': 0,
        'interpretation': 'native one-sided secants, not exact derivatives or causal source interventions'}
    return _finish(native, scores, output, identity, bindings, summary, ['cases.json', 'collection.json', *files])


def local_uncertainty(native, scores, protocol, output):
    native, scores, output, identity, bindings, methods, records = _begin(native, scores, protocol, output, 0)
    annotations, blocks = {}, {}
    for record in records:
        prepared, sample = record['prepared'], record['sample']
        if prepared not in annotations:
            manifest = data.read_json(prepared/'manifest.json')
            path = prepared/'annotations.jsonl'
            bindings[str(path)] = manifest['annotations_sha256']
            if data.file_hash(path) != bindings[str(path)]:
                raise ValueError('Released annotations changed')
            rows = list(read_jsonl(path))
            annotations[prepared] = {row['id']: row for row in rows}
            wanted = {row['sample'].id for row in records if row['prepared'] == prepared}
            if len(annotations[prepared]) != len(rows) or set(annotations[prepared]) != wanted:
                raise ValueError('Require exact unique annotation identities')
        annotation = annotations[prepared][sample.id]
        if annotation['response_sha256'] != data.text_hash(sample.response):
            raise ValueError('Annotation response identity changed')
        _check_spans(sample.response, annotation['spans'])
        if annotation['granularity'] == 'sentence':
            tokens = [{'start': int(a), 'end': int(b)} for a, b in record['offsets']]
            projected = sentence_token_labels(annotation, tokens)
            known = np.array([row['label'] is not None for row in projected])
            y = np.array([bool(row['label']) for row in projected])
        else:
            if annotation['granularity'] != 'token' or annotation['coverage'] != 'complete':
                raise ValueError('Complete original token labels or sentence projection required')
            if any(span['label'] != 1 for span in annotation['spans']):
                raise ValueError('Token annotations require positive spans and complete coverage')
            y = token_labels({'labels': annotation['spans']}, {'offsets': record['offsets']})
            known = np.ones(len(y), bool)
        block = blocks.setdefault(record['cohort'], [])
        block.append((record, known, y))
    groups = {}
    for cohort, parts in blocks.items():
        y = np.concatenate([labels[known] for _, known, labels in parts])
        sources = np.concatenate([np.repeat(row['sample'].source_id, int(known.sum())) for row, known, _ in parts])
        answers, units = [np.concatenate([row[key][known] for row, known, _ in parts]) for key in ('answer_index', 'unit_index')]
        values = {method: np.concatenate([row['scores'][method][known] for row, known, _ in parts]) for method in methods}
        groups[cohort] = {'responses_scored': len(parts), 'known_tokens': len(y), 'positive_tokens': int(y.sum()),
            'unknown_tokens': sum(int((~known).sum()) for _, known, _ in parts),
            'contrasts': {f'{method}_minus_witness': paired_span_differences(y, values['witness'], values[method],
                sources, answers, units, repetitions=1000, seed=17) for method in methods if method != 'witness'}}
    return _finish(native, scores, output, identity, bindings,
        {'groups': groups, 'annotation_reads': len(annotations), 'label_join': 'after score verification', 'new_model_calls': 0,
         'scope': 'post-result source-paired within-span AUROC uncertainty; no independent confirmation'})

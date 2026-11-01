"""Label-free cavity-response checks on hash-selected sealed native spans."""

from pathlib import Path
import time

import numpy as np

from experiments import data
from experiments.native_dataset import read_native
from experiments.provenance import completed_identity, research_identity
from myproject.detector import unit_ranges
from myproject.sequence import witness_emissions
from myproject.shared import SharedDetector

METHODS = ('witness', 'matched_local', 'symmetric_local')
MAX_SPANS = 256
EPSILON = .1
NUMERIC_TOLERANCE = 1e-9
DONOR_GAPS = (0., -2., -8., -32.)


def influence(distance, method):
    """Analytic neutral susceptibility; never used to generate detector scores."""
    distance = np.asarray(distance, float)
    if method in {'witness', 'persistent'}:
        return 1/3+2/(3*(distance+1))
    if method == 'matched_local':
        return 2/(distance+2)
    if method == 'symmetric_local':
        return 1/(distance+1)
    if method == 'fixed_local':
        return (2/3)**distance
    if method == 'fixed_persistence':
        return 1/3+(2/3)*(.5)**distance
    raise ValueError('Unknown probe readout')


def shrink_context(gaps, target):
    """Shrink surrounding mixture emissions, preserving the actual target gap."""
    gaps = np.asarray(gaps, float)
    emissions = witness_emissions(gaps)
    context = emissions.copy()
    context[target] = 0.
    magnitude = float(np.abs(context).sum())
    scale = min(1., EPSILON/magnitude) if magnitude else 1.
    scaled = context*scale
    # Scaled context lies in [-.1,.1], safely above the mixture's negative floor.
    transformed = np.log(np.expm1(scaled+np.log(2.)))
    transformed[target] = gaps[target]
    return transformed, context, scale


def span_candidates(index, source_id, sample_id, answers, units):
    """Pure original-axis candidate construction shared by both probe stages."""
    if not len(answers):
        return []
    starts, stops = unit_ranges(answers, units)
    return [{'selection_sha256': data.text_hash(f'{source_id}/{sample_id}/{int(units[a])}'),
             'sample_index': index, 'unit': int(units[a]), 'start': int(a), 'stop': int(b)}
            for a, b in zip(starts, stops, strict=True) if b-a >= 2]


def select_span_candidates(candidates, limit=MAX_SPANS):
    return sorted(candidates, key=lambda row: (row['selection_sha256'],
                   row['sample_index'], row['start']))[:limit], len(candidates)


def selected_spans(native, plan, samples, records, limit=MAX_SPANS):
    """Select by source/sample/unit hash, without reading gaps or annotations."""
    candidates = []
    for index, ((group, sample), record) in enumerate(zip(samples, records, strict=True)):
        if plan['groups'][group]['role'] != 'test':
            continue
        with np.load(native/record['file'], allow_pickle=False) as raw:
            candidates.extend(span_candidates(index, sample.source_id, sample.id,
                                              raw['answer_index'], raw['unit_index']))
    return select_span_candidates(candidates, limit)


def inspect_span(gaps, answers, units, detectors):
    """Evaluate library readouts against analytic predictions on one fixed span."""
    target = len(gaps)//2
    donor = 0 if target >= len(gaps)-1-target else len(gaps)-1
    changed, context, scale = shrink_context(gaps, target)
    own = float(witness_emissions(gaps)[target])
    distance = np.abs(np.arange(len(gaps))-target)
    bound = float((scale*np.abs(context).sum())**2/8)
    linear = {}
    for method in METHODS:
        cavity = float(detectors[method].score_full_gap(changed, answers, units)[target]-own)
        predicted = float(scale*np.dot(influence(distance, method), context))
        error = abs(cavity-predicted)
        linear[method] = {'cavity': cavity, 'predicted_cavity': predicted,
                          'absolute_error': error,
                          'within_bound': error <= bound+NUMERIC_TOLERANCE}
    stress = []
    for donor_gap in DONOR_GAPS:
        changed = np.zeros_like(gaps)
        changed[target], changed[donor] = gaps[target], donor_gap
        for method in ('witness', 'persistent'):
            donor_evidence = float(witness_emissions([donor_gap])[0]) if method == 'witness' else donor_gap
            own_evidence = own if method == 'witness' else float(gaps[target])
            effect = float(detectors[method].score_full_gap(changed, answers, units)[target]-own_evidence)
            coupling = float(influence(abs(donor-target), method))
            # Stable closed form for the prior pair's conditional log-MGF ratio.
            predicted = float(np.logaddexp(np.log1p(-coupling), np.log1p(coupling)+donor_evidence)
                              -np.logaddexp(np.log1p(coupling), np.log1p(-coupling)+donor_evidence))
            error = abs(effect-predicted)
            stress.append({'method': method, 'donor_raw_gap': donor_gap,
                           'donor_evidence': donor_evidence, 'effect': effect,
                           'predicted_effect': predicted, 'absolute_error': error,
                           'within_tolerance': error <= NUMERIC_TOLERANCE,
                           'negative_infinite_evidence_limit': float(np.log1p(-coupling)-np.log1p(coupling))})
    return {'target_position': target, 'donor_position': donor,
            'target_raw_gap': float(gaps[target]), 'target_emission': own,
            'context_scale': scale, 'context_l1': float(np.abs(context).sum()),
            'theory_bound': bound, 'linear_response': linear, 'single_site': stress}


def probe(native, output, protocol):
    """Create a sealed, bounded diagnostic with no labels and no new LM calls."""
    native, output, protocol = map(Path, (native, output, protocol))
    plan, samples, records = read_native(native)
    identity = research_identity()
    native_seal = data.file_hash(native/'seal.json')
    protocol_hash = data.file_hash(protocol)
    selected, eligible = selected_spans(native, plan, samples, records)
    detectors = {method: SharedDetector(plan['task'], method=method)
                 for method in (*METHODS, 'persistent')}
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output/'plan.json', {'native': str(native.resolve()),
        'native_seal_sha256': native_seal, 'observation_source': plan['source'],
        'protocol': str(protocol.resolve()), 'protocol_sha256': protocol_hash,
        'source': identity, 'max_spans': MAX_SPANS, 'eligible_spans': eligible,
        'selection': 'ascending SHA256(source_id/id/unit), non-singleton test spans',
        'epsilon': EPSILON, 'numeric_tolerance': NUMERIC_TOLERANCE,
        'theory_tolerance': EPSILON**2/8,
        'donor_raw_gaps': DONOR_GAPS, 'methods': METHODS,
        'annotation_reads': 0, 'new_model_calls': 0})
    cases, start_time = [], time.monotonic()
    for selection in selected:
        index = selection['sample_index']
        _, sample = samples[index]
        record = records[index]
        rows = slice(selection['start'], selection['stop'])
        with np.load(native/record['file'], allow_pickle=False) as raw:
            gaps = raw['delta'][rows, 1]
            answers, units = raw['answer_index'][rows], raw['unit_index'][rows]
            axes = {key: raw[key][rows].tolist() for key in
                    ('target', 'token_id', 'answer_index', 'unit_index')}
            axes['offsets'] = raw['offsets'][raw['target'][rows]].tolist()
        if not np.isfinite(gaps).all():
            raise ValueError('Nonfinite selected native contrast')
        cases.append({**selection, 'id': sample.id, 'source_id': sample.source_id,
            'observation_file': record['file'], 'observation_sha256': record['sha256'],
            'protocol_sha256': protocol_hash, 'native_seal_sha256': native_seal,
            'source_sha256': data.text_hash(sample.source),
            'prompt_sha256': data.text_hash(sample.prompt),
            'response_sha256': record['response_sha256'], 'axes': axes,
            'full_gap': gaps.tolist(), **inspect_span(gaps, answers, units, detectors)})
    read_native(native)
    if (data.file_hash(native/'seal.json') != native_seal
            or data.file_hash(protocol) != protocol_hash):
        raise ValueError('Native seal or probe protocol changed during probe')
    data.write_json(output/'cases.json', {'cases': cases, 'annotation_reads': 0})
    linear_errors = [row['absolute_error'] for case in cases for row in case['linear_response'].values()]
    stress_errors = [row['absolute_error'] for case in cases for row in case['single_site']]
    summary = {'selected_spans': len(cases), 'eligible_spans': eligible,
        'max_linear_error': max(linear_errors, default=0.),
        'max_single_site_error': max(stress_errors, default=0.),
        'linear_bound_violations': sum(not row['within_bound'] for case in cases
                                       for row in case['linear_response'].values()),
        'single_site_violations': sum(not row['within_tolerance'] for case in cases
                                      for row in case['single_site']),
        'seconds': time.monotonic()-start_time, 'annotation_reads': 0,
        'new_model_calls': 0, 'source': completed_identity(identity)}
    data.write_json(output/'summary.json', summary)
    data.write_json(output/'seal.json', {name: data.file_hash(output/name)
                    for name in ('plan.json', 'cases.json', 'summary.json')})
    return summary

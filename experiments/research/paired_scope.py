"""H016 exposed-toy compound/atomic plans and four fixed, reference-blind arms.

The two frozen compilers define the complete question bundles. This module has
no tokenizer, model, files or truth: source grammar alone licenses the toy
bijection. Callers must separately gate the fixed 16-case/8-source population
and validate/persist complete shared-prefix vectors before supplying readouts.
"""
from collections.abc import Mapping
from copy import deepcopy
import math

from . import atomic_reconstruction as atomic
from . import masked_reconstruction as masked
from .bijection_readout import assignment_distribution
from .source_reconstruction import candidate_distribution

CONTRACT_VERSION = 'paired-compound-atomic-b0-v1'
SCOPES = ('compound', 'atomic')
READOUTS = ('independent_rows_max_slot', 'bijection_joint_assignment_mass')
ARMS = tuple(f'{scope}_{readout}' for scope in SCOPES for readout in READOUTS)


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _sample(value):
    get = value.get if isinstance(value, Mapping) else lambda k: getattr(value, k, None)
    return {key: deepcopy(get(key)) for key in ('id', 'source_id', 'source', 'response')}


def _unknown(alignment, reason):
    value = deepcopy(alignment)
    value.update(status='unknown', reason=reason)
    for slot in value['occurrences']:
        slot.update(status='unknown', reason=reason, candidate_index=None,
                    query_key=None, observation_keys=[])
    return value


def prepare(samples):
    """Keep every original and both slot mappings per scope; only exact b0 scores.

This does not select a b0 subset from a larger population. Non-b0 or malformed
inputs remain explicit unknowns. Candidate members/order and event order come
only from the existing source parser. Keys from each frozen compiler are kept.
"""
    samples = [_sample(value) for value in samples]
    ids = [sample['id'] for sample in samples]
    _require(all(isinstance(key, str) and key for key in ids) and len(set(ids)) == len(ids),
             'Unique nonempty original IDs required')
    plan = dict(schema=CONTRACT_VERSION, sources={}, queries={}, candidate_rows={}, originals=[])
    for sample in samples:
        original = dict(id=sample['id'], sample=deepcopy(sample), scopes={})
        plan['originals'].append(original)
        for scope, compiler in (('compound', masked.prepare_masked_queries),
                                ('atomic', atomic.prepare_atomic_queries)):
            try:
                compiled = compiler([sample])
                alignment = deepcopy(compiled['originals'][0])
            except ValueError as error:
                compiled = masked.prepare_masked_queries([sample])
                alignment = _unknown(compiled['originals'][0], 'atomic_compiler:' + str(error))
            if alignment['status'] == 'ok' and sample['response'][alignment['target_span'][1]:] != '':
                alignment = _unknown(alignment, 'nonzero_burden_outside_paired_b0_contract')
            original['scopes'][scope] = alignment
            original['source_key'] = alignment['source_sha256']
            for key, source in compiled['sources'].items():
                value = deepcopy(source)
                value['source_is_admitted_bijection'] = source['status'] == 'ok'
                _require(key not in plan['sources'] or plan['sources'][key] == value, 'Source identity collision')
                plan['sources'][key] = value
            if alignment['status'] != 'ok':
                continue
            for slot in alignment['occurrences']:
                key = slot['query_key']
                query = dict(deepcopy(compiled['queries'][key]), scope=scope)
                _require(key not in plan['queries'] or plan['queries'][key] == query, 'Query identity collision')
                plan['queries'][key] = query
                for row_key in slot['observation_keys']:
                    row = dict(deepcopy(compiled['requests'][row_key]), scope=scope)
                    _require(row_key not in plan['candidate_rows'] or plan['candidate_rows'][row_key] == row,
                             'Candidate row identity collision')
                    plan['candidate_rows'][row_key] = row
    plan['sources'] = dict(sorted(plan['sources'].items()))
    plan['queries'] = dict(sorted(plan['queries'].items(), key=lambda pair:
        (SCOPES.index(pair[1]['scope']), pair[1]['source_sha256'], pair[1]['event'], pair[0])))
    plan['candidate_rows'] = dict(sorted(plan['candidate_rows'].items()))
    by_scope = {}
    for scope in SCOPES:
        slots = [slot for original in plan['originals'] for slot in original['scopes'][scope]['occurrences']]
        by_scope[scope] = dict(originals=len(samples), sources=len(plan['sources']), logical_queries=len(slots),
            unique_queries=sum(query['scope'] == scope for query in plan['queries'].values()),
            logical_candidates=sum(len(slot['observation_keys']) for slot in slots),
            unique_candidates=sum(row['scope'] == scope for row in plan['candidate_rows'].values()),
            unknown_slots=sum(slot['status'] != 'ok' for slot in slots))
    plan['counts'] = dict(originals=len(samples), sources=len(plan['sources']), unique_queries=len(plan['queries']),
        unique_candidates=len(plan['candidate_rows']), by_scope=by_scope)
    plan['limitations'] = dict(exposed_toy_only=True, natural_coverage_established=False,
        parser_can_answer_toy=True, calibrated_truth_probability=False, question_bundle_effect_only=True,
        source_bijection_is_not_inferred_from_shape=True, actual_token_node_budget_not_estimated=True,
        pure_kernel_model_calls=0, evaluation_material_accessed=False)
    return plan


def _finite(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and math.isfinite(float(value)) and value <= 0
    except (OverflowError, ValueError):
        return False


def _row(query, readout):
    _require(isinstance(readout, Mapping) and readout.get('status') in ('ok', 'unknown'),
             'Each planned query requires an explicit ok/unknown tree result')
    scores, reason = None, readout.get('reason')
    if readout['status'] == 'ok':
        paths = readout.get('paths')
        _require(isinstance(paths, Mapping) and set(paths) == set(query['candidates'])
            and all(isinstance(path, Mapping) and 'raw_joint_log_probability' in path for path in paths.values()),
            'Exact complete candidate paths required')
        raw = {name: paths[name]['raw_joint_log_probability'] for name in query['candidates']}
        if all(_finite(value) for value in raw.values()):
            scores = raw
        else:
            reason = 'nonfinite_or_invalid_complete_candidate_scores'
    distribution = (candidate_distribution(query, scores) if scores is not None else
                    dict(status='unknown', reason=reason or 'unknown_tree_readout', probabilities=None))
    if distribution['status'] != 'ok':
        scores = None
    winner = None if scores is None or len(set(scores.values())) == 1 else max(range(2), key=lambda i:
        scores[query['candidates'][i]])
    return dict(query_key=query['query_key'], scope=query['scope'], source_key=query['source_sha256'],
        event=query['event'], candidates=deepcopy(query['candidates']), status=distribution['status'],
        reason=distribution['reason'], sequence_logprobs=scores, distribution=distribution,
        candidate_argmax_index=winner, tie=None if scores is None else winner is None,
        tree_readout_status=readout['status'], tree_readout_reason=readout.get('reason'))


def derive(plan, per_query_readouts):
    """Derive all four fixed arms from validated H015-style pure tree results.

Mapping keys must equal every planned query, including explicit unknown tree
results. Complete paths supply raw_joint_log_probability, never conditional
mass or risk. Structural mismatches raise; unusable numeric results retain
unknown/risk1. Full vector/tree provenance is the caller's separate artifact.
"""
    _require(isinstance(plan, dict) and isinstance(plan.get('originals'), list), 'Canonical paired plan required')
    expected = prepare([original['sample'] for original in plan['originals']])
    _require(masked._key(plan) == masked._key(expected),
             'Paired plan changed from frozen compiler reconstruction')
    _require(isinstance(per_query_readouts, Mapping) and set(per_query_readouts) == set(plan['queries']),
             'Exact planned query readout population required')
    queries = {key: _row(query, per_query_readouts[key]) for key, query in plan['queries'].items()}
    matrices, matrix_for = {}, {}
    for scope in SCOPES:
        for source_key, source in plan['sources'].items():
            events, names = source['events'], source['people']
            keys = [key for key, query in plan['queries'].items()
                    if query['scope'] == scope and query['source_sha256'] == source_key]
            by_event = {plan['queries'][key]['event']: key for key in keys}
            _require(len(by_event) == len(keys), 'More than one b0 query for a canonical source/event')
            complete = (source['source_is_admitted_bijection'] is True and len(events) == len(names) == 2
                        and set(by_event) == set(events))
            ordered = [by_event[event] for event in events] if complete else keys
            matrix = ([[queries[key]['sequence_logprobs'][name] for name in names] for key in ordered]
                      if complete and all(queries[key]['status'] == 'ok' for key in ordered) else None)
            joint = assignment_distribution(matrix, events=events, candidates=names,
                source_is_admitted_bijection=source['source_is_admitted_bijection'])
            matrix_id = masked._key(dict(contract=CONTRACT_VERSION, scope=scope, source_key=source_key))
            rows = [deepcopy(queries[key]) for key in ordered]
            winners = [row['candidate_argmax_index'] for row in rows]
            matrices[matrix_id] = dict(matrix_id=matrix_id, scope=scope, source_key=source_key,
                events=deepcopy(events), candidates=deepcopy(names), query_keys=ordered,
                source_is_admitted_bijection=source['source_is_admitted_bijection'],
                raw_joint_log_probability_matrix=matrix, independent_rows=rows, bijection=joint,
                independent_assignment=winners,
                independent_same_name_collapse=(len(set(winners)) == 1 if len(winners) == 2
                                                and all(value is not None for value in winners) else None))
            matrix_for[scope, source_key] = matrix_id
    cases = []
    for original in plan['originals']:
        case = dict(id=original['id'], source_key=original['source_key'], scopes={}, readouts={})
        for scope in SCOPES:
            alignment = original['scopes'][scope]
            matrix_id = matrix_for.get((scope, original['source_key']))
            matrix = matrices.get(matrix_id)
            slots = alignment['occurrences']
            available = alignment['status'] == 'ok' and len(slots) == 2
            risks, indices = [], None
            if available:
                for slot in slots:
                    row = queries[slot['query_key']]
                    risks.append(1. - row['distribution']['probabilities'][slot['text']]
                                 if row['status'] == 'ok' else 1.)
                by_event = {slot['event']: slot for slot in slots}
                if matrix and set(by_event) == set(matrix['events']):
                    indices = [matrix['candidates'].index(by_event[event]['text']) for event in matrix['events']]
            independent_unknown = not available or any(queries[slot['query_key']]['status'] != 'ok' for slot in slots)
            valid = indices in ([0, 1], [1, 0])
            assignment_index = ([0, 1], [1, 0]).index(indices) if valid else None
            joint_ok = valid and matrix is not None and matrix['bijection']['status'] == 'ok'
            joint_reason = (None if joint_ok else alignment['reason'] if not available else
                            'response_mapping_not_bijection' if not valid else matrix['bijection']['reason'])
            case['scopes'][scope] = dict(matrix_id=matrix_id, alignment=deepcopy(alignment),
                observed_candidate_indices_by_canonical_event=indices,
                observed_mapping_is_bijection=valid, observed_assignment_index=assignment_index)
            case['readouts'][f'{scope}_{READOUTS[0]}'] = dict(risk=max(risks) if risks else 1.,
                unknown_fallback=independent_unknown,
                unknown_reason=alignment['reason'] if not available else 'unknown_query_readout' if independent_unknown else None,
                calibrated_truth_probability=False)
            case['readouts'][f'{scope}_{READOUTS[1]}'] = dict(
                risk=1. - matrix['bijection']['assignment_probabilities'][assignment_index] if joint_ok else 1.,
                unknown_fallback=not joint_ok, unknown_reason=joint_reason, calibrated_truth_probability=False)
        cases.append(case)
    return dict(schema=CONTRACT_VERSION + '-readouts', matrices=matrices, queries=queries, cases=cases,
        arms=list(ARMS), counts=deepcopy(plan['counts']), all_fixed_readouts_retained=True,
        pure_kernel_model_calls=0, evaluation_material_accessed=False, natural_task_performance_claim=False)

"""Fixed H009 bijection readout on all H011 cached matrices; zero new forwards.

This exposed-toy posthoc control retains both readouts and all original failures.
Conditional assignment energy mass is not a calibrated factuality posterior.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

from . import data, masked_reconstruction_evaluation as donor
from . import masked_reconstruction_qualification as profile
from . import reconstruction_evaluation as shared
from . import reconstruction_qualification as collector
from .datasets import Sample
from .provenance import completed_identity, research_identity
from .reconstruction_assignment import _paths, _freeze
from .research.bijection_readout import assignment_distribution
from .research.source_reconstruction import FILLERS, candidate_distribution

DEFAULT_PROTOCOL = dict(schema='masked-cached-assignment-v1',
    purpose='exposed_toy_H011_fixed_bijection_readout', required_sources=8,
    required_burdens=[0, 4, 12], required_matrices=24, required_queries=48,
    required_completions=96, required_cases=48, required_slots=96,
    model_calls_allowed=False, tokenizer_only_validation=True,
    readouts=['independent_rows_max_slot', 'bijection_joint_assignment_mass'],
    canonical_order='sorted_complete_source_event_and_person_strings',
    burden_rule='exact_original_response_suffix_after_original_first_unit',
    source_bijection_must_be_explicitly_admitted=True,
    assignment_contract='two-by-two-admitted-bijection-v1',
    score='original_full_candidate_plus_eos_joint_log_probability',
    temperature=1.0, length_normalization=False, synthetic_offsets_allowed=False,
    target_label_selection_allowed=False, threshold_search_allowed=False,
    fitted_calibration_allowed=False, posthoc_exposed_diagnostic=True,
    natural_task_performance_claim=False, method_promoted=False)
_FROZEN = {'plan.json', 'protocol.json', 'source_identity.json', 'scores.json', 'freeze.json'}
_REFERENCE = shared._CPU_FROZEN | {'reference-release.json', 'h008-control.json',
    'summary.json', 'queries.json', 'cases.json', 'candidate-observations.json', 'complete.json'}
_REFERENCE_HELPERS = donor._HELPERS + ('experiments/masked_reconstruction_evaluation.py',)
_require = collector._require
_equal = shared.reader._equal


def _protocol(path):
    value = data.read_json(path)
    _require(isinstance(value, dict) and not set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
        and all(type(value.get(k)) is type(v) and value[k] == v for k, v in DEFAULT_PROTOCOL.items()),
        'Exact fixed cached-H011 assignment protocol required')
    return value


def _argmax(values):
    best = [i for i, value in enumerate(values) if value == max(values)]
    return best[0] if len(best) == 1 else None


def _scores(prepared, executions, responses, complete):
    """All matrices/readouts precede reference access; suffixes come from R."""
    suffixes = {' '.join(FILLERS[:b]): b for b in (0, 4, 12)}
    lookup = {(e['query_key'], e['candidate']): e['key'] for e in executions}
    _require(len(executions) == len(lookup) == len(responses) == 96,
             'All96 distinct cached completions required')
    metadata, owners, references = {}, defaultdict(set), Counter()
    for original in prepared['originals']:
        response = original['sample']['response']
        suffix = response[original['alignment']['target_span'][1]:]
        _require(suffix in suffixes, 'Original response suffix is outside fixed0/4/12 inventory')
        burden, source_key = suffixes[suffix], original['source_key']
        owners[source_key, burden].add(original['sample']['id'])
        for slot in original['alignment']['slots']:
            key = slot['query_key']
            query = prepared['queries'][key]
            value = dict(source_key=source_key, burden=burden, event=slot['event'])
            _require(data.text_hash(query['source']) == source_key and query['event'] == slot['event']
                and (key not in metadata or metadata[key] == value), 'Source/event/suffix query identity changed')
            metadata[key] = value
            references[key] += 1
    _require(len(prepared['originals']) == 48 and len(prepared['sources']) == 8
        and set(metadata) == set(prepared['queries']) and len(metadata) == 48
        and set(references.values()) == {2} and len(owners) == 24
        and all(len(ids) == 2 for ids in owners.values())
        and all({b for s, b in owners if s == key} == {0, 4, 12} for key in prepared['sources']),
        'Complete48 originals/24 source-burden matrices/48 paired queries required')
    matrices, queries, matrix_for = {}, {}, {}
    for (source_key, burden), ids in sorted(owners.items()):
        source = prepared['sources'][source_key]
        events, names = sorted(source['events']), sorted(source['people'])
        _require(source['status'] == 'ok' and len(events) == len(names) == 2,
                 'Explicit admitted source bijection required;2x2 shape is insufficient')
        keys = sorted((key for key, meta in metadata.items()
                       if meta['source_key'] == source_key and meta['burden'] == burden),
                      key=lambda key: metadata[key]['event'])
        _require([metadata[key]['event'] for key in keys] == events
            and all(prepared['queries'][key]['candidates'] == names for key in keys),
            'Complete canonical event/person domain required')
        matrix = [[responses[lookup[key, name]]['joint_log_probability'] for name in names] for key in keys]
        joint = assignment_distribution(matrix, events=events, candidates=names,
                                        source_is_admitted_bijection=True)
        matrix_id = collector._digest(dict(source_key=source_key, burden=burden, events=events, candidates=names))
        independent = []
        for i, key in enumerate(keys):
            distribution = candidate_distribution(prepared['queries'][key], dict(zip(names, matrix[i], strict=True)))
            _require(distribution['status'] == 'ok', 'Finite complete independent row required')
            _equal(distribution, complete['queries'][key]['distribution'], 'Original H011 row probabilities changed')
            winner = _argmax(matrix[i])
            marginal = joint['marginals'][i] if joint['status'] == 'ok' else None
            marginal_winner = joint['map_assignment'][i] if joint['status'] == 'ok' and not joint['tie'] else None
            row = dict(query_key=key, matrix_id=matrix_id, source_key=source_key, burden=burden,
                event=events[i], candidates=names, row_index=i, distribution=distribution,
                candidate_argmax_index=winner, tie=winner is None,
                bijection_marginals=marginal, marginal_argmax_index=marginal_winner,
                marginal_tie=joint['status'] == 'ok' and joint['tie'],
                marginal_unknown=marginal is None)
            independent.append(row); queries[key] = row
        matrices[matrix_id] = dict(matrix_id=matrix_id, source_key=source_key, burden=burden,
            events=events, candidates=names, query_keys=keys, original_ids=sorted(ids),
            source_is_admitted_bijection=True, raw_joint_log_probability_matrix=matrix,
            independent_rows=independent, bijection=joint)
        matrix_for[source_key, burden] = matrix_id
    records = []
    _require(len(complete['records']) == 48, 'All48 original case readouts required')
    for original, old in zip(prepared['originals'], complete['records'], strict=True):
        slots = original['alignment']['slots']
        burden = metadata[slots[0]['query_key']]['burden']
        matrix_id = matrix_for[original['source_key'], burden]; matrix = matrices[matrix_id]
        by_event = {slot['event']: slot for slot in slots}
        indices = ([matrix['candidates'].index(by_event[event]['text']) for event in matrix['events']]
                   if len(by_event) == 2 and set(by_event) == set(matrix['events']) else None)
        valid = indices in ([0, 1], [1, 0])
        index = ([0, 1], [1, 0]).index(indices) if valid else None
        available = valid and matrix['bijection']['status'] == 'ok'
        risk = 1. - matrix['bijection']['assignment_probabilities'][index] if available else 1.
        independent_risk = max(1. - queries[s['query_key']]['distribution']['probabilities'][s['text']] for s in slots)
        _require(old['id'] == original['sample']['id'] and independent_risk == old['case_max_slot_risk'],
                 'Independent H011 original case risk changed')
        records.append(dict(id=old['id'], source_id=old['source_id'], source_key=original['source_key'],
            burden=burden, matrix_id=matrix_id, slots=deepcopy(slots),
            observed_candidate_indices_by_canonical_event=indices, observed_mapping_is_bijection=valid,
            observed_assignment_index=index,
            independent_rows_max_slot=dict(risk=independent_risk, unknown_fallback=False),
            bijection_joint_assignment_mass=dict(risk=risk, unknown_fallback=not available,
                unknown_reason=None if available else 'response_mapping_not_bijection' if not valid
                else matrix['bijection']['reason'], calibrated_truth_probability=False)))
    return dict(schema='masked-cached-assignment-scores-v1', matrices=matrices, queries=queries, records=records,
        coverage=deepcopy(complete['coverage']), donor_costs=deepcopy(complete['costs']),
        new_model_calls=0, new_language_observations=False, evaluation_material_accessed=False,
        posthoc_exposed_diagnostic=True, all_fixed_readouts_retained=True, synthetic_offsets_applied=False)


def _reference_snapshot(root):
    shared._cpu_inventory(root, _REFERENCE, sealed=True)
    seal, raw = shared.reader._seal_content(root, _REFERENCE)
    snapshot = (root, _REFERENCE, seal, data.text_hash(raw.decode('utf-8')), [])
    shared.reader._verify_observation_snapshot(*snapshot)
    return snapshot


def _review(prepared, executions, responses, complete, plan, root, donor_seal, current):
    packet = {name: data.read_json(root / name) for name in sorted(_REFERENCE)}
    protocol = donor._protocol(root / 'protocol.json')
    expected = dict(schema='masked-reconstruction-evaluation-complete-v1',
        state='sealed_masked_reconstruction_diagnosis', cases=48, unique_queries=48, unique_completions=96,
        response_slots=96, model_calls=0, tokenizer_only_validation=True, evaluation_only_after_predictions_frozen=True,
        natural_task_performance_claim=False, observation_initial_seal_sha256=donor_seal,
        h008_control_cpu_seal_sha256=protocol['h008_control_cpu_seal_sha256'],
        h008_control_observation_seal_sha256=protocol['h008_control_observation_seal_sha256'])
    _equal(packet['complete.json'], expected, 'H011 reference belongs to another donor or incomplete evaluation')
    source, cpu_plan = packet['source_identity.json'], packet['plan.json']
    for helper in _REFERENCE_HELPERS:
        digest = source.get('source_sha256', {}).get(helper)
        _require(shared.reader._sha(digest) and digest == current.get('source_sha256', {}).get(helper),
                 'Incompatible H011 reference/helper: ' + helper)
    _require(cpu_plan.get('schema') == 'masked-reconstruction-evaluation-plan-v1'
        and cpu_plan.get('state') == 'frozen_before_reference_release'
        and cpu_plan.get('observation_initial_seal_sha256') == donor_seal
        and cpu_plan.get('original_inputs_sha256') == plan['original_inputs_sha256']
        and cpu_plan.get('original_observation_protocol_sha256') == plan['protocol_sha256']
        and cpu_plan.get('source_identity_sha256') == collector._digest(source)
        and cpu_plan.get('protocol') == protocol
        and cpu_plan.get('evaluator_protocol_sha256') == data.file_hash(root / 'protocol.json')
        and all(cpu_plan.get(k) is False for k in
                ('construction_truth_accessed', 'reference_design_accessed', 'h008_control_accessed')),
        'H011 CPU freeze does not bind this complete donor')
    shared._check_cpu_freeze(root, (packet['freeze.json'], data.file_hash(root / 'freeze.json')), _REFERENCE, sealed=True)
    model = plan['model_identity']; _require(len(model['model_files_sha256']) == 9, 'All nine model assets required')
    differences = []
    for path, suffix in [(('execution', 'installed_observer_wheel_sha256'), 'observer_wheel_sha256'),
                         (('execution', 'CPU_module_target'), 'cpu_module_target'), (('readout', 'prefix'), 'readout_prefix')]:
        value = model
        for part in path: value = value[part]
        _require(value == protocol['h011_' + suffix], 'H011 retained model identity/control pin changed')
        differences.append(dict(path='.'.join(path), h008=protocol['h008_' + suffix], h011=value))
    control = packet['h008-control.json']
    _require(isinstance(control, dict) and set(control) == donor._CONTROL_NAMES
        and control['plan.json']['original_inputs_sha256'] == plan['original_inputs_sha256']
        and control['complete.json']['observation_initial_seal_sha256']
            == protocol['h008_control_observation_seal_sha256']
        and packet['reference-release.json'].get('h008_control_cpu_seal_sha256')
            == protocol['h008_control_cpu_seal_sha256'],
        'Full previously validated H008 control packet/pins must remain retained')
    design, truth = [], []
    _require(isinstance(packet['cases.json'], list) and len(packet['cases.json']) == 48, 'All48 H011 reference cases required')
    for original, old, case in zip(prepared['originals'], complete['records'], packet['cases.json'], strict=True):
        factor, binary = case['factor_cell'], case['construction_binary_truth']
        _require(type(binary) is int and binary in (0, 1), 'Exact binary construction reference required')
        _equal(case, dict(old, factor_cell=factor, construction_binary_truth=binary,
                          primary_evidence_is_query_level=True), 'H011 original case readout changed')
        sample, alignment = original['sample'], original['alignment']
        design.append(dict(id=sample['id'], source_id=sample['source_id'], factor_cell=factor,
            input_text_sha256=original['input_text_sha256'], target_span=alignment['target_span'],
            target_text=sample['response'][alignment['target_span'][0]:alignment['target_span'][1]],
            whole_span=[0, len(sample['response'])]))
        outcome = 'contradicted' if binary else 'supported'
        truth.append(dict(id=sample['id'], source_id=sample['source_id'],
            source_sha256=original['input_text_sha256']['source'], response_sha256=original['input_text_sha256']['response'],
            target_span=alignment['target_span'], target_outcome=outcome, whole_outcome=outcome,
            filler_outcomes=['supported'] * factor['distractor_count'], target_benchmark_labels_used=False,
            construction_rule='entity_event_bijections_equal_iff_source_direction_equals_response_direction'))
    source_view = collector._prepare([Sample(**o['sample']) for o in prepared['originals']])
    rebuilt = donor._analysis(prepared, source_view, executions, responses, complete,
        dict(schema='h007-burden-controls-plan-v1', records=design), truth,
        dict(packet=control, identity_differences=differences))
    for name, value in zip(('summary.json', 'queries.json', 'cases.json', 'candidate-observations.json'), rebuilt, strict=True):
        _equal(packet[name], value, 'Original sealed H011 full reconstruction changed: ' + name)
    return packet


def _analysis(scores, packet):
    old_queries, old_cases = packet['queries.json'], packet['cases.json']
    refs = {q['query_key']: q for q in old_queries}
    _require(len(old_queries) == 48 and set(refs) == set(scores['queries']), 'All48 H011 query references required')
    matrices, queries, cases = [], [], []
    counts = {b: {name: Counter() for name in DEFAULT_PROTOCOL['readouts']} for b in (0, 4, 12)}
    query_counts = deepcopy(counts); discordance = {b: Counter() for b in (0, 4, 12)}
    def category(prediction, truth, tie=False, unknown=False):
        return 'unknown' if unknown else 'ties' if tie else 'correct' if prediction == truth else 'wrong'
    for matrix in scores['matrices'].values():
        reference = [matrix['candidates'].index(refs[key]['reference_candidate']) for key in matrix['query_keys']]
        _require(reference in ([0, 1], [1, 0]), 'Reference source must preserve its admitted bijection')
        independent = [r['candidate_argmax_index'] for r in matrix['independent_rows']]
        joint = matrix['bijection']; burden = matrix['burden']
        ind_category = category(independent, reference, tie=any(i is None for i in independent))
        joint_category = category(joint['map_assignment'], reference, tie=joint['tie'] is True, unknown=joint['status'] != 'ok')
        counts[burden]['independent_rows_max_slot'][ind_category] += 1
        counts[burden]['bijection_joint_assignment_mass'][joint_category] += 1
        collapsed = None not in independent and len(set(independent)) == 1
        discordance[burden][('independent_collapse_' if collapsed else 'independent_' + ind_category + '_') + 'joint_' + joint_category] += 1
        matrices.append(dict(matrix, reference_candidate_indices=reference,
            independent_category=ind_category, joint_category=joint_category, independent_same_name_collapse=collapsed))
        for i, row in enumerate(matrix['independent_rows']):
            query = refs[row['query_key']]
            _require(query['source_key'] == matrix['source_key'] and query['burden'] == burden
                and query['event'] == matrix['events'][i] and query['candidates'] == matrix['candidates'],
                'H011 source/event/burden/candidate reference identity changed')
            query_counts[burden]['independent_rows_max_slot'][category(row['candidate_argmax_index'], reference[i], row['tie'])] += 1
            query_counts[burden]['bijection_joint_assignment_mass'][category(row['marginal_argmax_index'],
                reference[i], row['marginal_tie'], row['marginal_unknown'])] += 1
            queries.append(dict(row, reference_candidate_index=reference[i], reference_candidate=matrix['candidates'][reference[i]]))
    _require(len(old_cases) == len(scores['records']) == 48, 'All48 case references required')
    for record, old in zip(scores['records'], old_cases, strict=True):
        _require(record['id'] == old['id'] and record['burden'] == old['factor_cell']['distractor_count'],
                 'Original case/suffix/reference roster changed')
        cases.append(dict(record, factor_cell=old['factor_cell'], construction_binary_truth=old['construction_binary_truth']))
    def metrics(records, name):
        return shared.prior._metrics([r['construction_binary_truth'] for r in records],
            [r[name]['risk'] for r in records], [r[name]['unknown_fallback'] for r in records])
    def full(counter, total):
        return dict(total=total, **{k: counter[k] for k in ('correct', 'wrong', 'ties', 'unknown')})
    by_burden = {}
    for burden in (0, 4, 12):
        group = [r for r in cases if r['burden'] == burden]
        _require(len(group) == 16 and all(sum(c.values()) == 8 for c in counts[burden].values())
            and all(sum(c.values()) == 16 for c in query_counts[burden].values()), 'Complete per-burden denominators required')
        by_burden[str(burden)] = dict(matrices=8, queries=16, original_cases=16,
            readouts={name: dict(source_assignment=full(counts[burden][name], 8),
                query_argmax=full(query_counts[burden][name], 16), secondary_case_metrics=metrics(group, name))
                for name in DEFAULT_PROTOCOL['readouts']}, discordance=dict(discordance[burden]))
    summary = dict(schema='masked-cached-assignment-diagnosis-v1', unique_sources=8, source_burden_matrices=24,
        unique_queries=48, cached_completions=96, original_cases=48, response_slots=96,
        by_burden=by_burden, pooled_secondary_case_metrics={name: metrics(cases, name) for name in DEFAULT_PROTOCOL['readouts']},
        original_h011_summary=deepcopy(packet['summary.json']), donor_costs=scores['donor_costs'],
        new_model_calls=0, new_language_observations=False, synthetic_offsets_applied=False,
        posthoc_exposed_diagnostic=True, all_fixed_readouts_retained=True, winning_readout_selected=False,
        reference_access_only_after_score_freeze=True, natural_task_performance_claim=False, method_promoted=False,
        limitations=['Eight sources repeated across three burdens;24 matrices are not24 new independent sources.',
            'Conditional bijection energy mass is not a calibrated factuality posterior.',
            'Improvement cannot uniquely establish additive name bias or a generator/internal mechanism.',
            'The exact toy parser can solve these bindings; no natural coverage or LLM necessity is established.'])
    return summary, matrices, queries, cases


def analyze(inputs, observations, model_path, reconstruction_review, output, protocol):
    paths = _paths(inputs, observations, model_path, reconstruction_review, output, protocol)
    root, output = paths['root'], paths['output']
    inventory = shared._inventory(root, profile=profile); shared._snapshot(root, inventory, profile=profile)
    frozen = _protocol(paths['protocol']); protocol_hash = data.file_hash(paths['protocol']); source = research_identity()
    prepared, executions, responses, complete, plan, model_guard = shared._validate(
        paths['inputs'], root, paths['model'], source, profile=profile, helpers=donor._HELPERS)
    _require(len(plan['model_identity']['model_files_sha256']) == 9, 'All nine frozen model assets required')
    scores = _scores(prepared, executions, responses, complete)
    cpu_plan = dict(schema='masked-cached-assignment-plan-v1', state='scores_frozen_before_reference_release',
        observation_root=str(root), observation_initial_seal_sha256=inventory[2],
        original_inputs_sha256=plan['original_inputs_sha256'], original_observation_protocol_sha256=plan['protocol_sha256'],
        model_identity_sha256=collector._digest(plan['model_identity']), evaluator_protocol_sha256=protocol_hash,
        reference_review_path=str(paths['review']), source_identity_sha256=collector._digest(source), protocol=frozen,
        reference_accessed=False, posthoc_exposed_diagnostic=True, new_model_calls=0)
    cpu_freeze = _freeze(output, cpu_plan, frozen, source, scores)
    members, hashes, seal_hash, reference = set(_FROZEN), {}, None, None
    def unchanged():
        shared._check_cpu_freeze(output, cpu_freeze, members, sealed=seal_hash is not None)
        _require(all(data.file_hash(output / n) == h for n, h in hashes.items()), 'Retained cached assignment results changed')
        if seal_hash is not None: _require(data.file_hash(output / 'seal.json') == seal_hash, 'Initial CPU seal changed')
        shared._snapshot(root, inventory, profile=profile); completed_identity(source); model_guard()
        _require(data.file_hash(paths['inputs']) == plan['original_inputs_sha256']
            and data.file_hash(paths['protocol']) == protocol_hash, 'Frozen inputs/assignment protocol changed')
        if reference is not None:
            shared._cpu_inventory(paths['review'], _REFERENCE, sealed=True)
            shared.reader._verify_observation_snapshot(*reference)
    def retain(name, value):
        data.write_json(output / name, value); members.add(name); shared._cpu_inventory(output, members)
        _equal(data.read_json(output / name), value, 'Cached assignment result readback changed')
        hashes[name] = data.file_hash(output / name)
    unchanged()
    reference = _reference_snapshot(paths['review'])
    packet = _review(prepared, executions, responses, complete, plan, paths['review'], inventory[2], source)
    summary, matrices, queries, cases = _analysis(scores, packet)
    for name, value in [('reference-release.json', dict(reference_accessed_after_score_freeze=True,
        donor_initial_seal_sha256=inventory[2], reference_initial_seal_sha256=reference[3])),
        ('h011-reference.json', packet), ('summary.json', summary), ('matrices.json', matrices),
        ('queries.json', queries), ('cases.json', cases)]:
        retain(name, value)
    unchanged()
    try:
        retain('complete.json', dict(schema='masked-cached-assignment-complete-v1',
            state='sealed_cached_H011_assignment_diagnostic', matrices=24, unique_sources=8, queries=48,
            cached_completions=96, original_cases=48, response_slots=96, fixed_readouts=2,
            new_model_calls=0, posthoc_exposed_diagnostic=True, natural_task_performance_claim=False,
            method_promoted=False, observation_initial_seal_sha256=inventory[2], reference_initial_seal_sha256=reference[3]))
        unchanged()
        expected = dict(cpu_freeze[0], **{'freeze.json': cpu_freeze[1]}, **hashes)
        shared._seal_cpu(output, members, expected); shared._cpu_inventory(output, members, sealed=True)
        _equal(data.read_json(output / 'seal.json'), expected, 'Final cached assignment seal differs from retained hashes')
        seal_hash = data.file_hash(output / 'seal.json')
        data.verify_seal(output, expected=members); unchanged()
    except BaseException:
        (output / 'complete.json').unlink(missing_ok=True); (output / 'seal.json').unlink(missing_ok=True)
        raise
    return summary

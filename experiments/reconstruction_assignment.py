"""Frozen CPU ablation of two explicit bijections on the H008 cached scores.

The fixed row/column nuisance is a synthetic score intervention. It adds no
language observation, and the constrained energy mass is not a truth posterior.
"""
from copy import deepcopy
import math
from pathlib import Path

from . import data, reconstruction_evaluation as baseline
from . import reconstruction_qualification as observer
from . import verdict_evaluation as reader
from .provenance import completed_identity, research_identity
from .research.bijection_readout import assignment_distribution
from .research.source_reconstruction import candidate_distribution


DEFAULT_PROTOCOL = dict(schema='source-reconstruction-assignment-v1',
    purpose='preregistered_cached_bijection_nuisance_ablation', required_sources=8,
    required_queries=16, required_completions=32, required_cases=48, required_slots=96,
    model_calls_allowed=False, tokenizer_only_validation=True,
    source_bijection_must_be_explicitly_admitted=True, canonical_order='sorted_complete_event_and_name_strings',
    energy='sum_of_two_full_candidate_plus_eos_joint_log_probabilities', temperature=1.0,
    length_normalization=False, arms=['baseline', 'fixed_additive_nuisance'],
    nuisance_row_offsets=[-2, -5], nuisance_column_offsets=[-1, -9],
    readouts=['independent_rows_max_slot', 'bijection_joint_assignment_mass'], unknown_risk=1.0,
    numerical_absolute_tolerance=1e-10, numerical_relative_tolerance=1e-12,
    target_label_selection_allowed=False, threshold_search_allowed=False, fitted_calibration_allowed=False,
    natural_task_performance_claim=False, method_promoted=False, population_previously_exposed=True,
    nuisance_is_synthetic_score_intervention=True, new_language_observations=False)
_FROZEN = {'plan.json', 'protocol.json', 'source_identity.json', 'scores.json', 'freeze.json'}
_REVIEW_NAMES = baseline._CPU_FROZEN | {'reference-release.json', 'summary.json', 'queries.json',
    'cases.json', 'candidate-observations.json', 'complete.json'}
_require = observer._require


def _paths(inputs, observations, model_path, reconstruction_review, output, protocol):
    values = [Path(p).absolute() for p in
              (inputs, observations, model_path, reconstruction_review, output, protocol)]
    _require(all(p.resolve() == p for p in values), 'Unaliased assignment paths required')
    inputs, root, model, review, output, protocol = values
    trees = (inputs.parent, root, model, review, output)
    _require(inputs.name == 'inputs.jsonl' and 'outputs' in output.parts
        and all(not a.is_relative_to(b) and not b.is_relative_to(a)
                for i, a in enumerate(trees) for b in trees[i + 1:])
        and all(not protocol.is_relative_to(tree) for tree in trees)
        and protocol.name.casefold() not in observer.ANNOTATION_NAMES | {'construction_truth.jsonl'},
        'Independent inference/donor/model/reference/output trees and external protocol required')
    _require(not any(part.casefold() in {'construction_truth', 'evaluation_only', 'annotations'}
                     for part in protocol.parent.parts), 'Protocol cannot be inside an evaluation-label bundle')
    if output.exists(): raise FileExistsError('Retained assignment run must not be overwritten or retried')
    return dict(zip(('inputs', 'root', 'model', 'review', 'output', 'protocol'), values))


def _protocol(path):
    value = data.read_json(path)
    _require(isinstance(value, dict) and not set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
        and all(type(value.get(k)) is type(v) and value[k] == v for k, v in DEFAULT_PROTOCOL.items()),
        'Exact preregistered assignment protocol required')
    return value


def _argmax(values):
    winners = [i for i, value in enumerate(values) if value == max(values)]
    return winners[0] if len(winners) == 1 else None


def _scores(prepared, executions, responses, complete, protocol):
    """No reference access: use all32 cached joint scores and exact original slots."""
    keys = {(row['query_key'], row['candidate']): row['key'] for row in executions}
    sources = {}
    for source_key, source in sorted(prepared['sources'].items()):
        _require(source['status'] == 'ok' and len(source['people']) == len(source['events']) == 2,
                 'An admitted source bijection is required; matrix shape alone is insufficient')
        events, candidates = sorted(source['events']), sorted(source['people'])
        queries = sorted(source['queries'], key=lambda row: row['event'])
        _require([q['event'] for q in queries] == events and all(q['candidates'] == candidates for q in queries),
                 'Complete canonical two-event/two-name domain changed')
        matrix = [[responses[keys[q['query_key'], candidate]]['joint_log_probability']
                   for candidate in candidates] for q in queries]
        sources[source_key] = dict(source_key=source_key, events=events, candidates=candidates,
            query_keys=[q['query_key'] for q in queries], raw_joint_log_probability_matrix=matrix,
            source_is_admitted_bijection=True)
    _require(len(sources) == 8 and len(keys) == 32, 'Complete8-source/32-completion cache required')
    arms = {}
    for arm in protocol['arms']:
        readouts = {}
        for source_key, original in sources.items():
            matrix = deepcopy(original['raw_joint_log_probability_matrix'])
            if arm == 'fixed_additive_nuisance':
                matrix = [[math.fsum((value, protocol['nuisance_row_offsets'][i],
                                     protocol['nuisance_column_offsets'][j]))
                           for j, value in enumerate(row)] for i, row in enumerate(matrix)]
            joint = assignment_distribution(matrix, events=original['events'], candidates=original['candidates'],
                                             source_is_admitted_bijection=True)
            rows = []
            for i, key in enumerate(original['query_keys']):
                distribution = candidate_distribution(prepared['queries'][key],
                    dict(zip(original['candidates'], matrix[i], strict=True)))
                _require(distribution['status'] == 'ok', 'Finite independent row distribution required')
                winner = _argmax(matrix[i])
                marginal = joint['marginals'][i] if joint['status'] == 'ok' else None
                # Two feasible assignments induce each row's ordering directly.
                # Rounded .5/.5 masses must not erase a strict energy ordering.
                marginal_winner = joint['map_assignment'][i] if joint['status'] == 'ok' and not joint['tie'] else None
                rows.append(dict(query_key=key, event=original['events'][i], distribution=distribution,
                    candidate_argmax_index=winner, tie=winner is None,
                    bijection_marginals=marginal, marginal_argmax_index=marginal_winner,
                    marginal_tie=joint['status'] == 'ok' and joint['tie'],
                    rounded_marginal_probability_tie=marginal is not None and _argmax(marginal) is None,
                    marginal_unknown=marginal is None))
            readouts[source_key] = dict(original, score_matrix=matrix, independent_rows=rows, bijection=joint)
        records = []
        for original, old in zip(prepared['originals'], complete['records'], strict=True):
            source = readouts[original['source_key']]
            slots = original['alignment']['slots']; by_event = {slot['event']: slot for slot in slots}
            indices = ([source['candidates'].index(by_event[event]['text']) for event in source['events']]
                       if len(by_event) == 2 and set(by_event) == set(source['events']) else None)
            valid = indices in ([0, 1], [1, 0])
            assignment_index = ([0, 1], [1, 0]).index(indices) if valid else None
            available = valid and source['bijection']['status'] == 'ok'
            joint_risk = 1. - source['bijection']['assignment_probabilities'][assignment_index] if available else 1.
            row_risks = [1. - source['independent_rows'][source['events'].index(slot['event'])]
                         ['distribution']['probabilities'][slot['text']] for slot in slots]
            independent_risk = max(row_risks)
            if arm == 'baseline':
                _require(independent_risk == old['case_max_slot_risk'], 'H008 baseline max-slot score changed')
            records.append(dict(id=original['sample']['id'], source_id=original['sample']['source_id'],
                source_key=original['source_key'], slots=deepcopy(slots),
                observed_candidate_indices_by_canonical_event=indices,
                observed_mapping_is_bijection=valid, observed_assignment_index=assignment_index,
                independent_rows_max_slot=dict(risk=independent_risk, unknown_fallback=False),
                bijection_joint_assignment_mass=dict(risk=joint_risk, unknown_fallback=not available,
                    unknown_reason=None if available else 'response_mapping_not_bijection' if not valid
                    else source['bijection']['reason'], calibrated_truth_probability=False)))
        arms[arm] = dict(sources=readouts, records=records)
    return dict(schema='source-reconstruction-assignment-scores-v1', arms=arms,
        coverage=deepcopy(complete['coverage']), donor_costs=deepcopy(complete['costs']),
        new_model_calls=0, new_language_observations=False, evaluation_material_accessed=False,
        response_used_in_source_qa=False, all_fixed_arms_retained=True, method_promoted=False)


def _freeze(output, plan, protocol, source, scores):
    output.mkdir(parents=True)
    metadata = {'plan.json': plan, 'protocol.json': protocol, 'source_identity.json': source, 'scores.json': scores}
    for name, value in metadata.items(): data.write_json(output / name, value)
    baseline._cpu_inventory(output, set(metadata))
    for name, value in metadata.items(): reader._equal(data.read_json(output / name), value, 'Assignment freeze readback changed')
    frozen = {name: data.file_hash(output / name) for name in metadata}
    data.write_json(output / 'freeze.json', frozen)
    baseline._cpu_inventory(output, _FROZEN)
    return frozen, data.file_hash(output / 'freeze.json')


def _review_snapshot(root):
    """Called only after scores/plan/source/protocol are frozen."""
    baseline._cpu_inventory(root, _REVIEW_NAMES, sealed=True)
    seal, raw = reader._seal_content(root, _REVIEW_NAMES)
    snapshot = (root, _REVIEW_NAMES, seal, data.text_hash(raw.decode('utf-8')), [])
    reader._verify_observation_snapshot(*snapshot)
    return snapshot


def _review(prepared, executions, responses, complete, plan, root, initial_donor_seal, current):
    expected = dict(schema='source-reconstruction-evaluation-complete-v1', state='sealed_source_reconstruction_diagnosis',
        cases=48, unique_queries=16, unique_completions=32, sourceflip_pairs=8, eventflip_pairs=8,
        model_calls=0, tokenizer_only_validation=True, evaluation_only_after_predictions_frozen=True,
        natural_task_performance_claim=False, observation_initial_seal_sha256=initial_donor_seal)
    reader._equal(data.read_json(root / 'complete.json'), expected, 'Reference review belongs to another donor or incomplete evaluation')
    source = data.read_json(root / 'source_identity.json')
    for helper in (*baseline._HELPERS, 'experiments/reconstruction_evaluation.py'):
        digest = source.get('source_sha256', {}).get(helper)
        _require(reader._sha(digest) and digest == current.get('source_sha256', {}).get(helper),
                 'Incompatible frozen H008 reference reader helper: ' + helper)
    cpu_plan = data.read_json(root / 'plan.json')
    _require(cpu_plan.get('state') == 'frozen_before_reference_release'
        and cpu_plan.get('observation_initial_seal_sha256') == initial_donor_seal
        and cpu_plan.get('original_inputs_sha256') == plan['original_inputs_sha256']
        and cpu_plan.get('original_observation_protocol_sha256') == plan['protocol_sha256']
        and cpu_plan.get('source_identity_sha256') == observer._digest(source)
        and cpu_plan.get('construction_truth_accessed') is False and cpu_plan.get('reference_design_accessed') is False,
        'H008 reference evaluator freeze does not bind this donor')
    protocol = data.read_json(root / 'protocol.json')
    _require(isinstance(protocol, dict) and not set(protocol) - set(baseline.DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
        and all(type(protocol.get(k)) is type(v) and protocol[k] == v for k, v in baseline.DEFAULT_PROTOCOL.items()),
        'H008 reference evaluation protocol changed')
    freeze = data.read_json(root / 'freeze.json')
    names = baseline._CPU_FROZEN - {'freeze.json'}
    _require(set(freeze) == names and all(data.file_hash(root / name) == value for name, value in freeze.items()),
             'H008 reference evaluator freeze changed')
    packets = data.read_json(root / 'cases.json')
    _require(isinstance(packets, list) and len(packets) == 48, 'All48 sealed H008 reference cases required')
    design, truth = [], []
    for original, record, packet in zip(prepared['originals'], complete['records'], packets, strict=True):
        factor, binary = packet['factor_cell'], packet['construction_binary_truth']
        _require(type(binary) is int and binary in (0, 1), 'Exact binary construction reference required')
        reader._equal(packet, dict(record, factor_cell=factor, construction_binary_truth=binary,
                                  primary_evidence_is_query_level=True), 'H008 frozen case readout changed')
        sample, alignment = original['sample'], original['alignment']
        design.append(dict(id=sample['id'], source_id=sample['source_id'], factor_cell=factor,
            input_text_sha256=original['input_text_sha256'], target_span=alignment['target_span'],
            target_text=alignment['target_text'], whole_span=[0, len(sample['response'])]))
        outcome = 'contradicted' if binary else 'supported'
        truth.append(dict(id=sample['id'], source_id=sample['source_id'],
            source_sha256=original['input_text_sha256']['source'], response_sha256=original['input_text_sha256']['response'],
            target_span=alignment['target_span'], target_outcome=outcome, whole_outcome=outcome,
            filler_outcomes=['supported'] * factor['distractor_count'], target_benchmark_labels_used=False,
            construction_rule='entity_event_bijections_equal_iff_source_direction_equals_response_direction'))
    derived_design = dict(schema='h007-burden-controls-plan-v1', records=design)
    old_summary, old_queries, old_cases, old_raw = baseline._analysis(
        prepared, executions, responses, complete, derived_design, truth)
    for name, value in [('summary.json', old_summary), ('queries.json', old_queries),
                        ('cases.json', old_cases), ('candidate-observations.json', old_raw)]:
        reader._equal(data.read_json(root / name), value, 'Frozen H008 metric/reference reconstruction changed: ' + name)
    return old_summary, old_queries, old_cases


def _analysis(scores, old_summary, old_queries, old_cases, protocol):
    refs = {q['query_key']: q for q in old_queries}
    sources, queries, cases, arms = [], [], [], {}
    for arm, readout in scores['arms'].items():
        counts = {name: dict(correct=0, ties=0, unknown=0, wrong=0) for name in protocol['readouts']}
        query_counts = deepcopy(counts)
        for source_key, source in readout['sources'].items():
            indices = [source['candidates'].index(refs[key]['reference_candidate']) for key in source['query_keys']]
            _require(indices in ([0, 1], [1, 0]), 'Reference source is not an admitted bijection')
            independent = [row['candidate_argmax_index'] for row in source['independent_rows']]
            joint = source['bijection']; joint_map = joint['map_assignment']
            predictions = {'independent_rows_max_slot': (independent, any(i is None for i in independent), False),
                           'bijection_joint_assignment_mass': (joint_map, joint['tie'] is True, joint['status'] != 'ok')}
            for name, (prediction, tie, unknown) in predictions.items():
                category = 'unknown' if unknown else 'ties' if tie else 'correct' if prediction == indices else 'wrong'
                counts[name][category] += 1
            sources.append(dict(arm=arm, **source, reference_candidate_indices=indices,
                reference_candidates=[source['candidates'][i] for i in indices],
                independent_source_correct=independent == indices,
                bijection_source_correct=joint_map == indices,
                factor={k: refs[source['query_keys'][0]][k] for k in ('relation_family', 'vocabulary', 'source_direction')}))
            for i, row in enumerate(source['independent_rows']):
                predictions = {'independent_rows_max_slot': (row['candidate_argmax_index'], row['tie'], False),
                    'bijection_joint_assignment_mass': (row['marginal_argmax_index'], row['marginal_tie'], row['marginal_unknown'])}
                for name, (prediction, tie, unknown) in predictions.items():
                    category = 'unknown' if unknown else 'ties' if tie else 'correct' if prediction == indices[i] else 'wrong'
                    query_counts[name][category] += 1
                queries.append(dict(arm=arm, source_key=source_key, **row,
                    reference_candidate_index=indices[i], reference_candidate=source['candidates'][indices[i]]))
        labels = [row['construction_binary_truth'] for row in old_cases]
        metrics = {}
        for name in protocol['readouts']:
            metrics[name] = dict(source_map=dict(total=8, **counts[name]),
                query_marginal_argmax=dict(total=16, **query_counts[name]),
                secondary_case_metrics=baseline.prior._metrics(labels,
                    [r[name]['risk'] for r in readout['records']], [r[name]['unknown_fallback'] for r in readout['records']]))
        for record, reference in zip(readout['records'], old_cases, strict=True):
            _require(record['id'] == reference['id'], 'Assignment case/reference roster changed')
            cases.append(dict(arm=arm, **record, construction_binary_truth=reference['construction_binary_truth'],
                              factor_cell=reference['factor_cell']))
        arms[arm] = metrics
    invariant = []
    close = lambda a, b: math.isclose(a, b, abs_tol=protocol['numerical_absolute_tolerance'],
                                     rel_tol=protocol['numerical_relative_tolerance'])
    for source_key in scores['arms']['baseline']['sources']:
        first = scores['arms']['baseline']['sources'][source_key]['bijection']
        second = scores['arms']['fixed_additive_nuisance']['sources'][source_key]['bijection']
        known = first['status'] == second['status'] == 'ok'
        invariant.append(dict(source_key=source_key, both_finite=known,
            delta_equal_within_tolerance=known and close(first['delta'], second['delta']),
            probabilities_equal_within_tolerance=known and all(close(a, b) for a, b in zip(
                first['assignment_probabilities'], second['assignment_probabilities'], strict=True)),
            map_and_tie_identical=known and first['map_assignment'] == second['map_assignment'] and first['tie'] == second['tie']))
    summary = dict(schema='source-reconstruction-assignment-diagnosis-v1', arms=arms,
        invariance=dict(sources=8, records=invariant,
            finite_sources=sum(r['both_finite'] for r in invariant),
            delta_invariant_sources=sum(r['delta_equal_within_tolerance'] for r in invariant),
            probability_invariant_sources=sum(r['probabilities_equal_within_tolerance'] for r in invariant),
            map_and_tie_invariant_sources=sum(r['map_and_tie_identical'] for r in invariant)),
        original_h008_summary=deepcopy(old_summary), donor_costs=scores['donor_costs'],
        new_model_calls=0, new_language_observations=False, reference_access_only_after_score_freeze=True,
        all_fixed_arms_retained=True, winning_readout_selected=False, fitted_calibration_performed=False,
        natural_task_performance_claim=False, method_promoted=False,
        limitations=['Eight source matrices/16 unique questions;48 responses are not48 independent reconstructions.',
            'Baseline is already at the toy ceiling; this is a nuisance-invariance ablation, not a performance improvement claim.',
            'The synthetic fixed offsets are score interventions, not new neural observations or a proof that real name bias is additive.',
            'Two-assignment energies are conditional on an explicitly admitted source bijection; neither mass nor1-mass is calibrated truth.',
            'Toy grammar already permits deterministic solution; no natural coverage, LLM necessity, or novelty is established.'])
    return summary, sources, queries, cases


def analyze(inputs, observations, model_path, reconstruction_review, output, protocol):
    paths = _paths(inputs, observations, model_path, reconstruction_review, output, protocol)
    root, output = paths['root'], paths['output']
    inventory = baseline._inventory(root); baseline._snapshot(root, inventory)
    frozen = _protocol(paths['protocol']); protocol_hash = data.file_hash(paths['protocol']); source = research_identity()
    prepared, executions, responses, complete, donor_plan, model_guard = baseline._validate(
        paths['inputs'], root, paths['model'], source)
    scores = _scores(prepared, executions, responses, complete, frozen)
    plan = dict(schema='source-reconstruction-assignment-plan-v1', state='scores_frozen_before_reference_release',
        observation_root=str(root), observation_initial_seal_sha256=inventory[2],
        original_inputs_sha256=donor_plan['original_inputs_sha256'],
        original_observation_protocol_sha256=donor_plan['protocol_sha256'],
        model_identity_sha256=observer._digest(donor_plan['model_identity']),
        evaluator_protocol_sha256=protocol_hash, reference_review_path=str(paths['review']),
        source_identity_sha256=observer._digest(source), protocol=frozen,
        evaluation_material_accessed=False, new_model_calls=0, new_language_observations=False)
    cpu_freeze = _freeze(output, plan, frozen, source, scores)
    members = set(_FROZEN); result_hashes = {}; cpu_seal_hash = None; reference = None

    def unchanged():
        baseline._check_cpu_freeze(output, cpu_freeze, members, sealed=cpu_seal_hash is not None)
        _require(all(data.file_hash(output / name) == digest for name, digest in result_hashes.items()),
                 'Previously retained assignment results changed')
        if cpu_seal_hash is not None:
            _require(data.file_hash(output / 'seal.json') == cpu_seal_hash, 'Initial assignment seal changed')
        baseline._snapshot(root, inventory); completed_identity(source); model_guard()
        _require(data.file_hash(paths['inputs']) == donor_plan['original_inputs_sha256']
            and data.file_hash(paths['protocol']) == protocol_hash, 'Frozen inputs/assignment protocol changed')
        if reference is not None:
            baseline._cpu_inventory(paths['review'], _REVIEW_NAMES, sealed=True)
            reader._verify_observation_snapshot(*reference)

    def retain(name, value):
        data.write_json(output / name, value); members.add(name); baseline._cpu_inventory(output, members)
        reader._equal(data.read_json(output / name), value, 'Assignment result readback differs from prepared value')
        result_hashes[name] = data.file_hash(output / name)

    unchanged()
    reference = _review_snapshot(paths['review'])
    old_summary, old_queries, old_cases = _review(prepared, executions, responses, complete, donor_plan,
        paths['review'], inventory[2], source)
    summary, sources, queries, cases = _analysis(scores, old_summary, old_queries, old_cases, frozen)
    release = dict(reference_material_accessed_after_score_freeze=True, reference_initial_seal_sha256=reference[3],
        donor_initial_seal_sha256=inventory[2], reconstruction_review_path=str(paths['review']))
    for name, value in [('reference-release.json', release), ('summary.json', summary), ('sources.json', sources),
        ('queries.json', queries), ('cases.json', cases), ('h008-summary.json', old_summary),
        ('h008-queries.json', old_queries), ('h008-cases.json', old_cases)]: retain(name, value)
    unchanged()
    try:
        retain('complete.json', dict(schema='source-reconstruction-assignment-complete-v1',
            state='sealed_cached_bijection_nuisance_ablation', sources=8, unique_queries=16, cached_completions=32,
            original_cases=48, fixed_arms=2, readouts_per_arm=2, new_model_calls=0,
            reference_access_only_after_score_freeze=True, all_fixed_arms_retained=True,
            natural_task_performance_claim=False, method_promoted=False,
            observation_initial_seal_sha256=inventory[2], reference_initial_seal_sha256=reference[3]))
        unchanged()
        expected = dict(cpu_freeze[0], **{'freeze.json': cpu_freeze[1]}, **result_hashes)
        baseline._seal_cpu(output, members, expected); baseline._cpu_inventory(output, members, sealed=True)
        reader._equal(data.read_json(output / 'seal.json'), expected, 'Assignment seal differs from recorded payload hashes')
        cpu_seal_hash = data.file_hash(output / 'seal.json')
        data.verify_seal(output, expected=members); unchanged()
    except BaseException:
        (output / 'complete.json').unlink(missing_ok=True); (output / 'seal.json').unlink(missing_ok=True)
        raise
    return summary

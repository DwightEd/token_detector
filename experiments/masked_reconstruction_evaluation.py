"""CPU-only, complete-population review of the fixed toy masked-R bridge.

All observations are reconstructed before CPU freeze; reference/control packets
are first opened afterwards. No model calls, fitting, selected burden or natural
edit-localization claim is admitted by this reader.
"""
from collections import defaultdict
from copy import deepcopy
import math
from pathlib import Path

from . import data, masked_reconstruction_qualification as observer
from . import reconstruction_evaluation as shared
from . import reconstruction_qualification as source_observer
from .datasets import read_jsonl, read_samples
from .provenance import completed_identity, research_identity

_require = shared._require
_HELPERS = shared._HELPERS + ('experiments/masked_reconstruction_qualification.py',
                             'experiments.research/masked_reconstruction.py')
_REQUIRED_SHA = ('h008_control_cpu_seal_sha256', 'h008_control_observation_seal_sha256',
                 'h008_observer_wheel_sha256', 'h011_observer_wheel_sha256')
_REQUIRED_TEXT = ('h008_cpu_module_target', 'h011_cpu_module_target')
_CONTROL_NAMES = shared._CPU_FROZEN | {'reference-release.json', 'summary.json', 'queries.json',
                                      'cases.json', 'candidate-observations.json', 'complete.json'}
DEFAULT_PROTOCOL = dict(shared.DEFAULT_PROTOCOL, schema='masked-reconstruction-evaluation-v1',
    purpose='exposed_known_slot_masked_response_bridge', required_queries=48, required_completions=96,
    burdens=[0, 4, 12], queries_per_burden=16, supported_occurrences_per_masked_query=1,
    primary='raw_joint_log_probability_flip_directions_by_burden',
    all_burdens_retained=True, control_result_selection_allowed=False,
    h008_control_cpu_seal_sha256=None, h008_control_observation_seal_sha256=None,
    h008_observer_wheel_sha256=None, h011_observer_wheel_sha256=None,
    h008_cpu_module_target=None, h011_cpu_module_target=None,
    h008_readout_prefix='source_only_role_question_no_R_values',
    h011_readout_prefix='entire_response_person_literals_erased_fixed_masked_question')


def _protocol(path):
    value = data.read_json(path)
    required = set(_REQUIRED_SHA + _REQUIRED_TEXT)
    _require(isinstance(value, dict) and not set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
        and all(type(value.get(k)) is type(v) and value[k] == v for k, v in DEFAULT_PROTOCOL.items() if k not in required)
        and all(shared.reader._sha(value.get(k)) for k in _REQUIRED_SHA)
        and all(isinstance(value.get(k), str) and value[k].strip() == value[k] and value[k] for k in _REQUIRED_TEXT),
        'Prespecified masked reconstruction review and external control identity pins required')
    return value


def _paths(inputs, design, construction, observations, model_path, h008_review, output, protocol):
    paths = shared._paths(inputs, design, construction, observations, model_path, output, protocol)
    control = Path(h008_review).absolute()
    trees = [paths[key] if key not in ('inputs', 'construction') else paths[key].parent
             for key in ('inputs', 'construction', 'root', 'model', 'output')]
    _require(control.resolve() == control and all(not control.is_relative_to(tree)
        and not tree.is_relative_to(control) for tree in trees)
        and not paths['protocol'].is_relative_to(control), 'Independent unaliased H008 control tree required')
    paths['control'] = control
    return paths


def _packet(root, names, expected_seal):
    shared._cpu_inventory(root, names, sealed=True)
    seal, raw = shared.reader._seal_content(root, names)
    digest = data.text_hash(raw.decode('utf-8'))
    _require(digest == expected_seal, 'Pinned H008 CPU seal changed')
    snapshot = (root, set(names), seal, digest, [])
    shared.reader._verify_observation_snapshot(*snapshot)
    return snapshot


def _identity_pair(old, new, protocol):
    def differences(a, b, prefix=()):
        if type(a) is not type(b):
            return {prefix}
        if isinstance(a, dict):
            return {prefix + (k,) for k in set(a) ^ set(b)} | set().union(
                *(differences(a[k], b[k], prefix + (k,)) for k in set(a) & set(b)))
        return set() if a == b else {prefix}
    paths = {('execution', 'installed_observer_wheel_sha256'): 'observer_wheel_sha256',
             ('execution', 'CPU_module_target'): 'cpu_module_target', ('readout', 'prefix'): 'readout_prefix'}
    _require(isinstance(old, dict) and isinstance(new, dict)
        and len(old.get('model_files_sha256', {})) == 9 and len(new.get('model_files_sha256', {})) == 9,
        'Both instruments must preserve all nine model assets')
    _require(differences(old, new) == set(paths), 'Unapproved model/runtime/instrument identity difference')
    result = []
    for path, suffix in paths.items():
        a, b = old, new
        for part in path:
            _require(isinstance(a, dict) and isinstance(b, dict) and part in a and part in b,
                     'Required instrument identity pin missing')
            a, b = a[part], b[part]
        _require(a == protocol['h008_' + suffix] and b == protocol['h011_' + suffix],
                 'Allowed identity difference does not match its external protocol pin')
        result.append(dict(path='.'.join(path), h008=a, h011=b))
    return result


def _control(paths, protocol, cpu_freeze, model, original_hash):
    shared._check_cpu_freeze(paths['output'], cpu_freeze)
    cpu = _packet(paths['control'], _CONTROL_NAMES, protocol['h008_control_cpu_seal_sha256'])
    packet = {name: data.read_json(paths['control'] / name) for name in sorted(_CONTROL_NAMES)}
    plan, complete = packet['plan.json'], packet['complete.json']
    _require(plan.get('schema') == 'source-reconstruction-evaluation-plan-v1'
        and plan.get('state') == 'frozen_before_reference_release'
        and plan.get('original_inputs_sha256') == original_hash
        and plan.get('observation_initial_seal_sha256') == protocol['h008_control_observation_seal_sha256']
        and plan.get('source_identity_sha256') == source_observer._digest(packet['source_identity.json'])
        and complete.get('schema') == 'source-reconstruction-evaluation-complete-v1'
        and complete.get('state') == 'sealed_source_reconstruction_diagnosis'
        and all(complete.get(k) == v for k, v in dict(cases=48, unique_queries=16, unique_completions=32,
            sourceflip_pairs=8, eventflip_pairs=8, model_calls=0, tokenizer_only_validation=True).items())
        and complete.get('observation_initial_seal_sha256') == protocol['h008_control_observation_seal_sha256'],
        'Complete unchanged H008 CPU qualification packet required')
    shared._check_cpu_freeze(paths['control'], (packet['freeze.json'], data.file_hash(paths['control'] / 'freeze.json')),
                             _CONTROL_NAMES, sealed=True)
    old_protocol = packet['protocol.json']
    _require(isinstance(old_protocol, dict)
        and not set(old_protocol) - set(shared.DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
        and all(type(old_protocol.get(k)) is type(v) and old_protocol[k] == v
                for k, v in shared.DEFAULT_PROTOCOL.items()), 'H008 fixed CPU protocol changed')
    _require(plan.get('protocol') == old_protocol
        and plan.get('evaluator_protocol_sha256') == data.file_hash(paths['control'] / 'protocol.json'),
        'H008 CPU protocol/plan identity changed')
    old_root = Path(plan['observation_root']).absolute()
    trees = [paths[k] if k not in ('inputs', 'construction') else paths[k].parent
             for k in ('inputs', 'construction', 'root', 'model', 'output', 'control')]
    _require(old_root.resolve() == old_root and all(not old_root.is_relative_to(tree)
        and not tree.is_relative_to(old_root) for tree in trees)
        and not paths['protocol'].is_relative_to(old_root), 'Independent pinned H008 observation path required')
    inventory = shared._inventory(old_root)
    _require(inventory[2] == protocol['h008_control_observation_seal_sha256'], 'Pinned H008 observation seal changed')
    shared._snapshot(old_root, inventory)
    old_model = source_observer._model_identity(data.read_json(old_root / 'model_identity.json'))
    difference = _identity_pair(old_model, model, protocol)
    observation_plan = data.read_json(old_root / 'plan.json')
    observation_protocol = source_observer._protocol(old_root / 'protocol.json')
    _require(observation_plan.get('original_inputs_sha256') == original_hash
        and observation_plan.get('model_identity') == old_model
        and observation_protocol['original_inputs_sha256'] == original_hash
        and observation_plan.get('protocol_sha256') == data.file_hash(old_root / 'protocol.json')
        and plan.get('original_observation_protocol_sha256') == observation_plan['protocol_sha256'],
        'Old CPU control is not bound to its sealed observation/model identity')
    observed = data.read_json(old_root / 'complete.json')
    _require(packet['summary.json'].get('costs') == observed.get('costs')
        and packet['summary.json'].get('coverage') == observed.get('coverage')
        and len(observed.get('records', [])) == 48
        and all(all(case.get(k) == value for k, value in record.items())
                for case, record in zip(packet['cases.json'], observed['records'], strict=True)),
        'H008 CPU case readouts/costs differ from their sealed observations')
    return dict(packet=packet, identity_differences=difference, model_identity=old_model,
                cpu_snapshot=cpu, observation_root=old_root, observation_snapshot=inventory)


def _references(prepared, source_view, design, truth):
    rows, labels, references, source_metadata = shared._references(source_view, design, truth)
    canonical = {(data.text_hash(q['source']), q['event']): key for key, q in source_view['queries'].items()}
    groups = defaultdict(list)
    for original, row, label in zip(prepared['originals'], rows, labels, strict=True):
        _require(original['sample']['id'] == row['id'], 'Masked case/design order changed')
        for slot in original['alignment']['slots']:
            groups[slot['query_key']].append((original, row, label, slot))
    _require(set(groups) == set(prepared['queries']) and len(groups) == 48, 'Complete masked query reference join required')
    joined = {}
    for key, group in groups.items():
        query = prepared['queries'][key]
        source_key = data.text_hash(query['source'])
        burdens = {r['factor_cell']['distractor_count'] for _, r, _, _ in group}
        canonical_key = canonical.get((source_key, query['event']))
        _require(len(group) == 2 and len(burdens) == 1
            and {r['factor_cell']['response_direction'] for _, r, _, _ in group} == {0, 1}
            and canonical_key is not None
            and query['candidates'] == source_view['queries'][canonical_key]['candidates']
            and all(o['source_key'] == source_key and slot['event'] == query['event'] for o, _, _, slot in group),
            'Masked source/event/burden/R-direction collapse changed')
        supported = [slot['text'] for _, _, label, slot in group if label['target_outcome'] == 'supported']
        reference = references[canonical_key]
        _require(supported == [reference]
            and all((slot['text'] == reference) == (label['target_outcome'] == 'supported')
                    for _, _, label, slot in group), 'Exactly one supported reference per masked query required')
        joined[key] = dict(reference=reference, source_query_key=canonical_key, burden=next(iter(burdens)),
                          **{k: v for k, v in source_metadata[source_key].items() if k != 'query_keys'})
    _require(all(sum(row['burden'] == b for row in joined.values()) == 16 for b in (0, 4, 12)),
             'Each burden must retain all16 queries')
    return rows, labels, joined


def _analysis(prepared, source_view, executions, responses, complete, design, truth, control):
    rows, labels, joined = _references(prepared, source_view, design, truth)
    packet = control['packet']
    old_queries = packet['queries.json']
    _require(isinstance(old_queries, list) and len(old_queries) == 16
        and {q['query_key'] for q in old_queries} == set(source_view['queries'])
        and len(packet['cases.json']) == 48 and [c['id'] for c in packet['cases.json']] == [r['id'] for r in rows]
        and len(packet['candidate-observations.json']) == 32, 'Entire H008 control population must be retained')
    old_lookup = {q['query_key']: q for q in old_queries}
    old_pairs = set()
    for row in packet['candidate-observations.json']:
        query = old_lookup.get(row['query_key']); pair = row['query_key'], row['candidate']
        _require(query is not None and pair not in old_pairs and row['candidate'] in query['candidates']
            and row['response']['joint_log_probability'] == query['sequence_logprobs'][row['candidate']],
            'H008 raw candidate/readout join changed')
        old_pairs.add(pair)
    _require(old_pairs == {(key, name) for key, q in source_view['queries'].items() for name in q['candidates']},
             'Every old control candidate must be retained')
    queries = []
    for key, query in prepared['queries'].items():
        meta = joined[key]
        old = old_lookup[meta['source_query_key']]
        _require(old['candidates'] == query['candidates'] and old['event'] == query['event']
            and old['source_key'] == data.text_hash(query['source'])
            and old['question'] == source_view['queries'][meta['source_query_key']]['question']
            and old['reference_candidate'] == meta['reference'], 'H008 source/event/reference alignment changed')
        distribution = complete['queries'][key]['distribution']
        values, candidates = distribution['sequence_logprobs'], query['candidates']
        winners = [c for c in candidates if values[c] == max(values.values())]
        greedy = winners[0] if len(winners) == 1 else None
        mass = math.exp(distribution['log_candidate_normalizer'])
        reference = meta['reference']
        queries.append(dict(query_key=key, source_key=data.text_hash(query['source']), **meta,
            event=query['event'], question=query['question'], candidates=candidates, reference_candidate=reference,
            greedy_candidate=greedy, tie=greedy is None, greedy_correct=greedy == reference,
            sequence_logprobs=values, raw_canonical_log_odds=values[candidates[0]] - values[candidates[1]],
            raw_reference_margin=values[reference] - values[next(c for c in candidates if c != reference)],
            conditional_pool_probabilities=distribution['probabilities'],
            log_candidate_mass=distribution['log_candidate_normalizer'], candidate_mass=mass,
            candidate_mass_underflow=mass == 0, candidate_token_counts=complete['queries'][key]['candidate_token_counts'],
            candidate_mass_is_not_truth_probability=True, name_length_or_frequency_bias_removed=False,
            h008_control_query=deepcopy(old)))
    cases = [dict(record, factor_cell=row['factor_cell'], construction_binary_truth=int(label['target_outcome'] == 'contradicted'),
                  primary_evidence_is_query_level=True) for row, label, record in zip(rows, labels, complete['records'], strict=True)]
    _require([case['id'] for case in cases] == [row['id'] for row in rows], 'All48 case readouts required in order')
    by_burden = {}
    for burden in (0, 4, 12):
        cohort = [q for q in queries if q['burden'] == burden]
        scenarios = defaultdict(dict)
        for q in cohort:
            scenarios[q['relation_family'], q['vocabulary']].setdefault(q['source_direction'], []).append(q)
        sourceflip, eventflip = [], []
        _require(len(scenarios) == 4 and all(set(d) == {0, 1} for d in scenarios.values()), 'All four scenarios per burden required')
        for (family, vocabulary), directions in scenarios.items():
            directions = {d: sorted(qs, key=lambda q: q['event']) for d, qs in directions.items()}
            _require(all(len(qs) == 2 for qs in directions.values())
                and [q['event'] for q in directions[0]] == [q['event'] for q in directions[1]], 'Exact event pairs per burden required')
            for first, second in zip(directions[0], directions[1], strict=True):
                sourceflip.append(shared._flip_pair(first, second, 'sourceflip', dict(burden=burden,
                    relation_family=family, vocabulary=vocabulary, event=first['event'])))
            for direction, pair in directions.items():
                eventflip.append(shared._flip_pair(*pair, 'eventflip', dict(burden=burden,
                    relation_family=family, vocabulary=vocabulary, source_direction=direction, events=[q['event'] for q in pair])))
        _require(len(sourceflip) == len(eventflip) == 8, 'All8+8 paired effects required per burden')
        _require(all(math.isfinite(row['oriented_raw_interaction']) for row in sourceflip + eventflip),
                 'Paired interaction overflow is not a finite diagnosis')
        def flip(records):
            return dict(pairs=8, correctly_reversed_pairs=sum(r['correctly_reversed'] for r in records),
                positive_raw_interactions=sum(r['positive_source_or_event_response'] for r in records), records=records,
                positive_interaction_alone_is_not_correct_reconstruction=True)
        subcases = [c for c in cases if c['factor_cell']['distractor_count'] == burden]
        observed = [responses[row['key']] for row in executions if joined[row['query_key']]['burden'] == burden]
        _require(len(observed) == 32 and len(subcases) == 16, 'All32 completions/16 cases per burden required')
        by_burden[str(burden)] = dict(queries=16, correct=sum(q['greedy_correct'] for q in cohort),
            ties=sum(q['tie'] for q in cohort), wrong=sum(not q['greedy_correct'] and not q['tie'] for q in cohort),
            costs=dict(actual_forward_passes=sum(r['actual_forward_passes'] for r in observed),
                input_tokens=sum(r['input_tokens'] for r in observed),
                teacher_forced_target_tokens=sum(r['target_tokens'] for r in observed), output_tokens=0,
                seconds_forward=math.fsum(r['seconds'] for r in observed)),
            sourceflip=flip(sourceflip), eventflip=flip(eventflip), secondary_case_metrics=shared.prior._metrics(
                [c['construction_binary_truth'] for c in subcases], [c['case_max_slot_risk'] for c in subcases],
                [c['unknown_fallback'] for c in subcases]))
    summary = dict(schema='masked-reconstruction-diagnosis-v1', original_cases=48, unique_sources=8,
        unique_queries=48, unique_completions=96, response_slots=96, logical_completion_requests=192,
        by_burden=by_burden, h008_control=deepcopy(packet['summary.json']),
        identity_differences=control['identity_differences'], costs=complete['costs'], coverage=complete['coverage'],
        runtime_observations=dict(max_peak_cuda_allocated_bytes=max(r['peak_cuda_allocated_bytes'] for r in responses.values()),
                                  max_peak_cuda_reserved_bytes=max(r['peak_cuda_reserved_bytes'] for r in responses.values())),
        secondary_case_metrics=shared.prior._metrics([c['construction_binary_truth'] for c in cases],
            [c['case_max_slot_risk'] for c in cases], [c['unknown_fallback'] for c in cases]),
        model_calls=0, tokenizer_only_validation=True, completion_metadata_rebuilt=96, all_results_retained=True,
        method_promoted=False, natural_task_performance_claim=False, candidate_length_bias_removed=False,
        threshold_search_performed=False, target_label_selection_performed=False,
        limitations=['Known toy slots/bijection and correct events; no natural frontend or unique edit-localization certificate.',
            '48 originals share8 sources and48 masked queries; Rflip differences arise from original-value alignment.',
            'All three burdens retained; H00832 versus H01196 unique completions are not a matched-budget comparison.',
            'Finite candidate mass/max-slot risk is not calibrated truth; full-name likelihood retains length/frequency priors.'])
    raw = [dict(query_key=row['query_key'], candidate=row['candidate'], key=row['key'], response=responses[row['key']]) for row in executions]
    return summary, queries, cases, raw


def evaluate(inputs, design, construction, observations, model_path, h008_review, output, protocol):
    paths = _paths(inputs, design, construction, observations, model_path, h008_review, output, protocol)
    root, output = paths['root'], paths['output']
    inventory = shared._inventory(root, profile=observer)
    shared._snapshot(root, inventory, profile=observer)
    frozen = _protocol(paths['protocol'])
    protocol_hash, source = data.file_hash(paths['protocol']), research_identity()
    prepared, executions, responses, complete, plan, model_guard = shared._validate(
        paths['inputs'], root, paths['model'], source, profile=observer, helpers=_HELPERS)
    source_view = source_observer._prepare(list(read_samples(paths['inputs'])))
    _require(all(a['sample'] == b['sample'] and a['input_text_sha256'] == b['input_text_sha256']
        and a['source_key'] == b['source_key'] for a, b in zip(prepared['originals'], source_view['originals'], strict=True)),
        'Masked/source-defined views must compile the same complete original inputs')
    cpu_plan = dict(schema='masked-reconstruction-evaluation-plan-v1', state='frozen_before_reference_release',
        original_inputs_sha256=plan['original_inputs_sha256'], observation_root=str(root),
        observation_initial_seal_sha256=inventory[2], original_observation_protocol_sha256=plan['protocol_sha256'],
        evaluator_protocol_sha256=protocol_hash, reference_design_path=str(paths['design']),
        construction_truth_path=str(paths['construction']), h008_control_path=str(paths['control']),
        source_identity_sha256=source_observer._digest(source), protocol=frozen,
        construction_truth_accessed=False, reference_design_accessed=False, h008_control_accessed=False,
        model_calls=0, tokenizer_only_validation=True, completion_metadata_rebuilt=96)
    cpu_freeze = shared._freeze_cpu(output, cpu_plan, frozen, source)
    members, hashes, seal_hash = set(shared._CPU_FROZEN), {}, None
    references, control = [], None

    def unchanged():
        shared._check_cpu_freeze(output, cpu_freeze, members, sealed=seal_hash is not None)
        _require(all(data.file_hash(output / name) == digest for name, digest in hashes.items()), 'CPU masked results changed')
        if seal_hash is not None:
            _require(data.file_hash(output / 'seal.json') == seal_hash, 'Initial CPU masked seal changed')
        shared._snapshot(root, inventory, profile=observer)
        completed_identity(source)
        _require(data.file_hash(paths['inputs']) == plan['original_inputs_sha256']
            and data.file_hash(paths['protocol']) == protocol_hash, 'Inputs/evaluator protocol changed')
        model_guard()
        for snapshot in references:
            parent, names = snapshot[:2]
            _require({p.name for p in parent.iterdir()} == names | {'seal.json'}, 'Reference bundle inventory changed')
            shared.reader._verify_observation_snapshot(*snapshot)
        if control is not None:
            shared._cpu_inventory(paths['control'], _CONTROL_NAMES, sealed=True)
            shared.reader._verify_observation_snapshot(*control['cpu_snapshot'])
            shared._snapshot(control['observation_root'], control['observation_snapshot'])

    unchanged()
    # Admit the third known reference namespace before either H007 payload hash.
    shared._cpu_inventory(paths['control'], _CONTROL_NAMES, sealed=True)
    references = shared._reference_snapshot(paths, cpu_freeze)
    control = _control(paths, frozen, cpu_freeze, plan['model_identity'], plan['original_inputs_sha256'])
    design_value, truth = data.read_json(paths['design']), list(read_jsonl(paths['construction']))
    old_release = control['packet']['reference-release.json']
    expected_release = dict(reference_material_accessed_after_cpu_freeze=True,
        inference_seal_sha256=references[0][3], construction_seal_sha256=references[1][3],
        design_sha256=data.file_hash(paths['design']), construction_truth_sha256=data.file_hash(paths['construction']))
    _require(all(old_release.get(key) == value for key, value in expected_release.items()),
             'H008 control must bind the same original design/construction reference bundle')
    summary, queries, cases, raw = _analysis(prepared, source_view, executions, responses, complete, design_value, truth, control)
    release = dict(reference_material_accessed_after_cpu_freeze=True, inference_seal_sha256=references[0][3],
        construction_seal_sha256=references[1][3], design_sha256=data.file_hash(paths['design']),
        construction_truth_sha256=data.file_hash(paths['construction']),
        h008_control_cpu_seal_sha256=control['cpu_snapshot'][3],
        h008_control_observation_seal_sha256=control['observation_snapshot'][2])

    def retain(name, value):
        data.write_json(output / name, value); members.add(name)
        shared._cpu_inventory(output, members)
        shared.reader._equal(data.read_json(output / name), value, 'Written masked CPU result differs from reconstructed value')
        hashes[name] = data.file_hash(output / name)

    for name, value in [('reference-release.json', release), ('h008-control.json', control['packet']),
                        ('summary.json', summary), ('queries.json', queries), ('cases.json', cases), ('candidate-observations.json', raw)]:
        retain(name, value)
    unchanged()
    try:
        retain('complete.json', dict(schema='masked-reconstruction-evaluation-complete-v1',
            state='sealed_masked_reconstruction_diagnosis', cases=48, unique_queries=48, unique_completions=96,
            response_slots=96, model_calls=0, tokenizer_only_validation=True, evaluation_only_after_predictions_frozen=True,
            natural_task_performance_claim=False, observation_initial_seal_sha256=inventory[2],
            h008_control_cpu_seal_sha256=control['cpu_snapshot'][3],
            h008_control_observation_seal_sha256=control['observation_snapshot'][2]))
        unchanged()
        expected = dict(cpu_freeze[0], **{'freeze.json': cpu_freeze[1]}, **hashes)
        shared._seal_cpu(output, members, expected)
        shared._cpu_inventory(output, members, sealed=True)
        shared.reader._equal(data.read_json(output / 'seal.json'), expected, 'Final masked CPU seal differs')
        seal_hash = data.file_hash(output / 'seal.json')
        data.verify_seal(output, expected=members); unchanged()
    except BaseException:
        (output / 'complete.json').unlink(missing_ok=True); (output / 'seal.json').unlink(missing_ok=True)
        raise
    return summary

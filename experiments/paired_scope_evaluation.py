"""CPU replay and four-arm exposed-toy diagnosis; references follow score freeze."""
from collections import Counter
from pathlib import Path

from . import data, paired_scope_qualification as observer
from . import reconstruction_evaluation as reference_reader
from . import reconstruction_qualification as source_compiler
from .datasets import read_jsonl, read_samples
from .focused_verdict_evaluation import _metrics
from .provenance import completed_identity, research_identity
from .research import paired_scope as kernel

_require = observer._require
_digest = observer._digest
EVALUATION_CONTRACT = dict(schema='paired-scope-evaluation-v1', arms=list(kernel.ARMS),
    source_assignment_categories=['correct', 'wrong', 'ties', 'unknown'],
    secondary_metrics=['auroc', 'ap'], primary_independent_sources=8,
    threshold_selection=False, winning_arm_selection=False, model_calls_allowed=False,
    previously_exposed_toy=True, question_bundle_effect_only=True,
    natural_task_performance_claim=False, calibrated_truth_probability=False)


def _snapshot(root, keys):
    names = observer._inventory(root, keys, keys, complete=True, sealed=True)
    seal = data.read_json(root / 'seal.json')
    _require(isinstance(seal, dict) and set(seal) == names, 'Exact complete observation seal required')
    _require(all(data.file_hash(root / n) == h for n, h in seal.items()), 'Initial observation payload changed')
    return dict(seal_sha256=data.file_hash(root / 'seal.json'), file_sha256=seal)


def validate(observations):
    """Admit inventories/seals, rebuild framing and replay every full FP16 vector."""
    root = observer._path(observations)
    query_root = observer.paths._member(root, 'queries', directory=True)
    keys = sorted(p.name for p in query_root.iterdir())
    initial = _snapshot(root, keys)
    manifest = data.read_json(root / 'manifest.json')
    _require(set(keys) == set(manifest['queries']), 'Manifest/query inventory changed')
    identity = observer._model_identity(data.read_json(root / 'model_identity.json'))
    _require(identity == manifest['model_identity'], 'Observer identity changed')
    inputs, protocol, model = map(observer._path,
        (manifest['inputs_path'], manifest['protocol_path'], manifest['model_path']))
    frozen = observer._protocol(protocol)
    _require(frozen == manifest['protocol'] == data.read_json(root / 'protocol.json')
        and data.file_hash(inputs) == frozen['original_inputs_sha256']
        and data.file_hash(protocol) == manifest['protocol_sha256'], 'Input/protocol identity changed')
    prep = data.read_json(root / 'preparation.json')
    prep_root = observer._path(prep['path'])
    _require(observer._flat_snapshot(prep_root, observer.PREPARATION_FILES) == prep['snapshot'], 'Initial preparation seal changed')
    _require(all(data.read_json(root / n) == data.read_json(prep_root / n)
        for n in ('manifest.json', 'protocol.json', 'model_identity.json', 'source_identity.json')), 'Preparation copied metadata changed')
    source = data.read_json(root / 'source_identity.json')
    _require(research_identity()['source_sha256'] == source['source_sha256'], 'CPU reader source differs from frozen observer')
    donor = observer._engineering(manifest['engineering']['root'], identity)
    _require(donor == manifest['engineering'] and donor['snapshot']['seal_sha256'] == frozen['engineering_seal_sha256'], 'Qualified cache changed')
    rebuilt = observer._build(inputs, protocol, frozen, model, identity, donor, observer._tokenizer(model, identity))
    _require(rebuilt == manifest, 'Full canonical tokenizer manifest changed')
    _require(not any(q['reused_node_ids'] for q in manifest['queries'].values()), 'Unimplemented cached-tree import')
    guard_model = observer._model_guard(model, identity)
    freeze = data.read_json(root / 'freeze.json')
    expected_frozen = observer.ROOT_FILES - {'freeze.json'}
    expected_frozen |= {f'queries/{key}/request.json' for key in keys}
    _require(set(freeze) == expected_frozen and all(initial['file_sha256'][n] == h for n, h in freeze.items()), 'Observation initial freeze changed')
    results = {}
    for key, query in manifest['queries'].items():
        child = root / 'queries' / key
        observer._flat_snapshot(child, {'request.json', 'raw.json', 'vectors.npz', 'replay.json'})
        _require(data.read_json(child / 'request.json') == query, 'Frozen request changed')
        raw = data.read_json(child / 'raw.json')
        _require(raw.get('model_identity') == identity and raw.get('model_identity_sha256') == _digest(identity), 'Raw model identity changed')
        replay = observer._replay(child, query['trie'], raw, manifest['full_vocabulary_size'])
        _require(data.read_json(child / 'replay.json') == dict(same_kernel_replay=True,
            independent_mathematical_implementation=False, numpy_allow_pickle=False, result=replay), 'Stored CPU replay changed')
        results[key] = raw
    complete = observer._complete(manifest, results)
    _require(complete == data.read_json(root / 'complete.json')
        and complete['costs']['actual_forward_passes'] == complete['costs']['attempted_model_calls'] == manifest['budget']['new_forward_passes']
        and complete['costs']['input_tokens'] == manifest['budget']['planned_input_tokens'], 'Complete four-arm scores/costs changed')
    def unchanged():
        _require(_snapshot(root, keys) == initial
            and observer._flat_snapshot(prep_root, observer.PREPARATION_FILES) == prep['snapshot']
            and observer._engineering(donor['root'], identity) == donor
            and data.file_hash(inputs) == frozen['original_inputs_sha256']
            and data.file_hash(protocol) == manifest['protocol_sha256']
            and research_identity()['source_sha256'] == source['source_sha256'], 'Initial observation/source/input/cache changed')
        guard_model()
    unchanged()
    return manifest, complete, initial, unchanged


def _category(prediction, reference, *, tie=False, unknown=False):
    return 'unknown' if unknown else 'ties' if tie else 'correct' if prediction == reference else 'wrong'


def _analysis(scores, source_prepared, design, truth):
    """Reuse the original48 reference identity checks; evaluate only frozen b0 IDs."""
    rows, labels, refs, metadata = reference_reader._references(source_prepared, design, truth)
    source_queries = source_prepared['queries']
    by_event = {(q['source_sha256'] if 'source_sha256' in q else data.text_hash(q['source']), q['event']): refs[key]
                for key, q in source_queries.items()}
    case_refs = {row['id']: (row['factor_cell'], int(label['target_outcome'] == 'contradicted'))
                 for row, label in zip(rows, labels, strict=True)}
    matrices, queries, cases = [], [], []
    counts = {arm: dict(source=Counter(), query=Counter()) for arm in kernel.ARMS}
    source_comparison = {}
    for matrix in scores['matrices'].values():
        scope, source_key = matrix['scope'], matrix['source_key']
        reference = [matrix['candidates'].index(by_event[source_key, event]) for event in matrix['events']]
        _require(reference in ([0, 1], [1, 0]), 'Reference is not the source-licensed bijection')
        independent, joint = matrix['independent_rows'], matrix['bijection']
        predictions = [r['candidate_argmax_index'] for r in independent]
        ind_cat = _category(predictions, reference, tie=any(r['tie'] is True for r in independent),
            unknown=any(r['status'] != 'ok' for r in independent))
        joint_cat = _category(joint['map_assignment'], reference, tie=joint['tie'] is True, unknown=joint['status'] != 'ok')
        entry = dict(matrix, reference_candidate_indices=reference, source_factors=metadata[source_key],
            independent_category=ind_cat, joint_category=joint_cat)
        matrices.append(entry)
        source_comparison.setdefault(source_key, {})[scope] = dict(independent_category=ind_cat,
            joint_category=joint_cat, independent_assignment=predictions, joint_assignment=joint['map_assignment'],
            same_name_collapse=matrix['independent_same_name_collapse'], reference=reference, delta=joint['delta'])
        for readout, category in zip(kernel.READOUTS, (ind_cat, joint_cat), strict=True):
            counts[f'{scope}_{readout}']['source'][category] += 1
        for i, row in enumerate(independent):
            marginal = joint['marginals'][i] if joint['status'] == 'ok' else None
            jt = marginal is not None and marginal[0] == marginal[1]
            jp = None if marginal is None or jt else max(range(2), key=marginal.__getitem__)
            ic = _category(row['candidate_argmax_index'], reference[i], tie=row['tie'] is True, unknown=row['status'] != 'ok')
            jc = _category(jp, reference[i], tie=jt, unknown=marginal is None)
            for readout, category in zip(kernel.READOUTS, (ic, jc), strict=True):
                counts[f'{scope}_{readout}']['query'][category] += 1
            queries.append(dict(row, reference_candidate_index=reference[i],
                reference_candidate=matrix['candidates'][reference[i]], independent_category=ic,
                joint_category=jc, joint_marginals=marginal, joint_candidate_argmax_index=jp))
    for case in scores['cases']:
        factors, label = case_refs[case['id']]
        _require(factors['distractor_count'] == 0, 'Nonprimary case in frozen score population')
        cases.append(dict(case, factor_cell=factors, construction_binary_truth=label))
    def full(counter, total):
        _require(sum(counter.values()) == total, 'Incomplete evaluation denominator')
        return dict(total=total, **{k: counter[k] for k in EVALUATION_CONTRACT['source_assignment_categories']})
    arms = {arm: dict(source_assignment=full(counts[arm]['source'], 8),
        query_argmax=full(counts[arm]['query'], 16), secondary_case_metrics=_metrics(
            [r['construction_binary_truth'] for r in cases], [r['readouts'][arm]['risk'] for r in cases],
            [r['readouts'][arm]['unknown_fallback'] for r in cases])) for arm in kernel.ARMS}
    summary = dict(EVALUATION_CONTRACT, primary_cases=len(cases), unique_queries=len(queries),
        arms=arms, all_fixed_readouts_retained=True, reference_access_only_after_score_freeze=True,
        new_model_calls=0, method_promoted=False,
        limitations=['Eight source clusters; directions and roles are not independent samples.',
            'Exact source parser can answer this toy; LLM necessity and natural coverage remain untested.',
            'Atomic scope changes an entire question bundle; no unique internal causal mechanism follows.',
            'Conditional candidate/assignment mass is not a calibrated truth posterior.'])
    return summary, matrices, queries, cases, source_comparison


def evaluate(observations, design, construction, output):
    root, output = map(observer._path, (observations, output))
    _require(not output.exists() and 'outputs' in output.parts
        and not output.is_relative_to(root) and not root.is_relative_to(output), 'Fresh independent CPU output required')
    # Paths can be checked now; no reference payload or reference seal is read yet.
    design, construction = map(lambda p: Path(p).absolute(), (design, construction))
    _require(design.name == 'plan.json' and construction.name == 'construction_truth.jsonl'
        and design.resolve() == design and construction.resolve() == construction
        and design.parent != construction.parent
        and not output.is_relative_to(design.parent) and not output.is_relative_to(construction.parent)
        and not design.is_relative_to(output) and not construction.is_relative_to(output), 'Independent original reference paths required')
    manifest, complete, initial, observation_guard = validate(root)
    _require(design.parent == observer._path(manifest['inputs_path']).parent, 'Original H007 design path required')
    source = research_identity()
    source_prepared = source_compiler._prepare(list(read_samples(manifest['inputs_path'])))
    output.mkdir(parents=True)
    for name, value in [('scores.json', complete['readouts']), ('source_identity.json', source),
        ('evaluation_protocol.json', EVALUATION_CONTRACT), ('plan.json', dict(state='all_four_scores_frozen_before_references',
            observation_path=str(root), initial_observation_snapshot=initial, manifest_sha256=_digest(manifest),
            original_design_path=str(design), construction_path=str(construction), evaluation_material_accessed=False))]:
        data.write_json(output / name, value)
    frozen = {p.name: data.file_hash(p) for p in output.iterdir()}
    data.write_json(output / 'freeze.json', frozen)
    retained = dict(frozen, **{'freeze.json': data.file_hash(output / 'freeze.json')})
    references = None
    def unchanged(sealed=False):
        _require({p.name for p in output.iterdir()} == set(retained) | ({'seal.json'} if sealed else set())
            and all(data.file_hash(observer.paths._member(output, n)) == h for n, h in retained.items()), 'Frozen CPU scores/results changed')
        if sealed:
            _require(data.read_json(output / 'seal.json') == retained, 'CPU result seal changed')
        observation_guard(); completed_identity(source)
        if references is not None:
            _require(observer._flat_snapshot(design.parent, {'inputs.jsonl', 'plan.json'}) == references[0]
                and observer._flat_snapshot(construction.parent, {'construction_truth.jsonl'}) == references[1], 'Initial reference seal changed')
    def retain(name, value):
        data.write_json(output / name, value)
        _require(data.read_json(output / name) == value, 'CPU result readback changed')
        retained[name] = data.file_hash(output / name)
    try:
        unchanged()
        references = [observer._flat_snapshot(design.parent, {'inputs.jsonl', 'plan.json'}),
            observer._flat_snapshot(construction.parent, {'construction_truth.jsonl'})]
        design_value, truth = data.read_json(design), list(read_jsonl(construction))
        analysis = _analysis(complete['readouts'], source_prepared, design_value, truth)
        retain('reference-release.json', dict(reference_accessed_after_score_freeze=True,
            initial_snapshots=references, design=design_value, construction=truth))
        for name, value in zip(('summary.json', 'matrices.json', 'queries.json', 'cases.json', 'source-discordance.json'), analysis, strict=True):
            retain(name, value)
        retain('complete.json', dict(state='four_arm_exposed_toy_diagnosis_complete', all_four_arms_retained=True,
            observation_initial_seal_sha256=initial['seal_sha256'], new_model_calls=0, natural_task_performance_claim=False))
        unchanged()
        data.write_json(output / 'seal.json', retained)
        unchanged(sealed=True)
    except BaseException:
        (output / 'complete.json').unlink(missing_ok=True)
        (output / 'seal.json').unlink(missing_ok=True)
        raise
    return analysis[0]

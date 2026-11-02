"""CPU/tokenizer-only evaluation of a sealed, context-limited external NLI pilot.

Standard source-premise/unit-hypothesis NLI omits the original answer and QA
context. Its externally supervised scores are not Qwen's matched control, a
causal mechanism, calibrated probabilities, or an adopted unsupervised method.
"""
from pathlib import Path

from . import data, nli_qualification as nli
from . import focused_verdict_evaluation as focused, verdict_evaluation as reader
from . import typed_verdict_evaluation as controls_reader
from .datasets import read_samples
from .provenance import completed_identity, research_identity

HFEntailment = nli.HFEntailment
DEFAULT_PROTOCOL = dict(schema='nli-evaluation-v1', purpose='posthoc_exploratory_evaluation',
    required_cases=34, required_units=104, natural_cases=18, constructed_cases=16,
    model_calls_allowed=False, tokenizer_only_validation=True, target_label_selection_allowed=False,
    threshold_sweeps_allowed=False, fitted_calibration_allowed=False, unknown_risk=1.0,
    natural_axis='released_token_span_character_axis_or_native_sentence_axis',
    sentence_reduction='maximum_overlapping_unit_risk',
    controls_reduction='maximum_all_unit_risks_empty_response_zero',
    document_score_name='document_max_unit_broadcast', token_label_rewrites_allowed=False)
_HELPERS = ('experiments/nli_qualification.py', 'experiments.research/focused_verdict.py',
            'experiments/semantic_qualification.py', 'experiments/data.py', 'experiments/datasets.py')
_PANEL_FILES = {'inputs.jsonl', 'manifest.json', 'evaluation_only.jsonl'}
_METADATA = {'plan.json', 'complete.json', 'protocol.json', 'source_identity.json',
             'model_identity.json', 'freeze.json'}
_require = controls_reader._require


def _inventory(root):
    """Admit every nested path and seal key before hashing or source discovery."""
    names = {path.name for path in root.iterdir()} if root.is_dir() else set()
    _require(root.resolve() == root and names in (_METADATA | {'seal.json'},
        _METADATA | {'seal.json', 'batches'}), 'Incomplete or aliased NLI archive')
    members, children = set(_METADATA), []
    for name in _METADATA | {'seal.json'}:
        reader._member(root, name)
    if 'batches' in names:
        parent = reader._member(root, 'batches', directory=True)
        batch_names = sorted(path.name for path in parent.iterdir())
        _require(batch_names and batch_names == [f'{i:05d}' for i in range(len(batch_names))],
                 'Invalid or incomplete NLI batch directory inventory')
        for name in batch_names:
            child = reader._member(parent, name, directory=True)
            expected = {'request.json', 'backend_raw.json'}
            _require({path.name for path in child.iterdir()} == expected | {'seal.json'},
                     'Unexpected or incomplete NLI batch members')
            for member in expected | {'seal.json'}:
                members.add(str(reader._member(child, member).relative_to(root)))
            children.append((child, expected))
    seal, raw = reader._seal_content(root, members)
    for child, expected in children:
        reader._seal_content(child, expected)
    return members, seal, data.text_hash(raw.decode('utf-8')), children


def _snapshot(root, inventory):
    reader._verify_observation_snapshot(root, *inventory)


def _validate(inputs, root, model_path, current):
    plan = data.read_json(root / 'plan.json')
    protocol = nli._protocol(root / 'protocol.json')
    model = nli._identity(data.read_json(root / 'model_identity.json'))
    source = data.read_json(root / 'source_identity.json')
    for name in _HELPERS:
        digest = source.get('source_sha256', {}).get(name)
        _require(reader._sha(digest) and digest == current.get('source_sha256', {}).get(name),
                 'Incompatible archived NLI helper: ' + name)
    input_hash = data.file_hash(inputs)
    _require(input_hash == protocol['original_inputs_sha256'], 'Original NLI input hash changed')
    _require(reader._sha(plan.get('protocol_sha256')), 'Missing original NLI protocol hash')
    samples = list(read_samples(inputs))
    backend = HFEntailment(model_path, model, admit_resources=False)
    backend.verify_frozen_identity()
    records, pairs, batches = nli._prepare(samples, backend)  # Tokenizer only; never classify().
    expected_plan = nli._plan(inputs, input_hash, plan['protocol_sha256'], protocol,
                              model, source, records, pairs, batches)
    reader._equal(plan, expected_plan, 'NLI original roster/partition/pair/batch plan changed')
    expected_directories = {f'{index:05d}' for index in range(len(batches))}
    actual_directories = {p.name for p in (root / 'batches').iterdir()} if (root / 'batches').exists() else set()
    _require(actual_directories == expected_directories, 'NLI batch population changed')
    expected_frozen = {'plan.json', 'protocol.json', 'model_identity.json', 'source_identity.json'} | {
        str(Path('batches') / f'{i:05d}' / 'request.json') for i in range(len(batches))}
    freeze = data.read_json(root / 'freeze.json')
    _require(isinstance(freeze, dict) and set(freeze) == expected_frozen
        and all(reader._sha(value) for value in freeze.values()), 'NLI pre-observation freeze inventory changed')
    _require(all(data.file_hash(root / name) == digest for name, digest in freeze.items()),
             'NLI pre-observation frozen metadata changed')
    costs, observations = nli._costs(records, pairs), {}
    for index, keys in enumerate(batches):
        child = root / 'batches' / f'{index:05d}'
        reader._equal(data.read_json(child / 'request.json'),
            dict(keys=keys, pairs=[pairs[key] for key in keys]), 'Frozen NLI batch token IDs changed')
        wrapper = data.read_json(child / 'backend_raw.json')
        _require(isinstance(wrapper, dict) and set(wrapper) == {'value'}, 'Invalid retained raw NLI batch')
        values, batch_costs = nli._batch(wrapper['value'], keys, pairs, index)
        observations.update(values)
        for name, value in batch_costs.items():
            costs[name] += value
    complete = nli._completion(records, observations, protocol, source, model, costs)
    reader._equal(data.read_json(root / 'complete.json'), complete,
                  'NLI max-entailment readout/unknown/cost/coverage completion changed')
    backend.verify_frozen_identity()
    _require(getattr(backend, 'model', None) is None, 'Tokenizer-only validation constructed a model')
    return samples, complete, plan, backend


def _document(row):
    return dict(risk=max((unit['risk'] for unit in row['units']), default=0.0),
        unknown_fallback=any(unit['unknown_fallback'] for unit in row['units']),
        name='document_max_unit_broadcast', actual_whole_model_judgment=False)


def evaluate(inputs, observations, natural_panel, controls_panel, model_path, output, protocol):
    """Validate every observation before freezing the CPU release/export stage."""
    inputs, root, natural, controls, model_path, output, protocol = (Path(p).absolute() for p in
        (inputs, observations, natural_panel, controls_panel, model_path, output, protocol))
    paths = (inputs, root, natural, controls, model_path, output, protocol)
    _require(all(path.resolve() == path for path in paths) and inputs.name == 'inputs.jsonl'
        and protocol.name.casefold() not in nli.ANNOTATION_NAMES, 'Unaliased independent evaluation paths required')
    trees = (root, natural, controls, model_path, output)
    _require(all(not a.is_relative_to(b) and not b.is_relative_to(a)
        for index, a in enumerate(trees) for b in trees[index + 1:])
        and all(not protocol.is_relative_to(path) for path in trees)
        and not inputs.is_relative_to(root) and not inputs.is_relative_to(output),
        'Disjoint NLI evaluation trees required')
    if output.exists():
        raise FileExistsError('Retained NLI evaluation output must not be overwritten')
    inventory = _inventory(root)
    _snapshot(root, inventory)
    frozen = data.read_json(protocol)
    _require(isinstance(frozen, dict) and not set(frozen) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
        and all(type(frozen.get(key)) is type(value) and frozen[key] == value
                for key, value in DEFAULT_PROTOCOL.items()), 'Prespecified NLI evaluation protocol required')
    source, protocol_hash = research_identity(), data.file_hash(protocol)
    samples, complete, observer_plan, backend = _validate(inputs, root, model_path, source)
    panels = []
    for panel, subset in ((natural, samples[:18]), (controls, samples[18:])):
        _require({path.name for path in panel.iterdir()} == _PANEL_FILES | {'seal.json'},
                 'Exact original four-file panel required')
        seal, raw = reader._seal_content(panel, _PANEL_FILES)
        _require(data.file_hash(panel / 'inputs.jsonl') == seal['inputs.jsonl'], 'Panel input hash changed')
        reader._equal([focused._identity(sample) for sample in read_samples(panel / 'inputs.jsonl')],
                      [focused._identity(sample) for sample in subset], 'Panel exact original 18/16 roster changed')
        panels.append((panel, seal, data.text_hash(raw.decode('utf-8'))))
    _snapshot(root, inventory)
    output.mkdir(parents=True)
    data.write_json(output / 'plan.json', dict(schema='nli-evaluation-plan-v1',
        state='evaluator_frozen_before_release', inputs_sha256=observer_plan['original_inputs_sha256'],
        observation_root=str(root), observation_seal_sha256=inventory[2],
        original_observation_protocol_sha256=observer_plan['protocol_sha256'],
        protocol_provenance_note='Original external protocol hash is retained; exact protocol values and sealed serialized copy are independently validated.',
        panel_seals=[dict(path=str(path), seal_sha256=digest, seal=seal) for path, seal, digest in panels],
        original_ids=[sample.id for sample in samples], protocol_sha256=protocol_hash,
        evaluator_source_identity_sha256=nli._digest(source), evaluation_material_accessed=False,
        model_calls=0, tokenizer_only_validation=True, tokenizer_only_pair_preparations=104))
    data.write_json(output / 'protocol.json', frozen)
    data.write_json(output / 'source_identity.json', source)
    frozen_files = {path.name: data.file_hash(path) for path in output.iterdir()}
    def unchanged():
        _snapshot(root, inventory)
        completed_identity(source)
        _require(data.file_hash(inputs) == observer_plan['original_inputs_sha256']
            and data.file_hash(protocol) == protocol_hash, 'Original NLI input/evaluation protocol changed')
        for name, digest in frozen_files.items():
            _require(data.file_hash(output / name) == digest, 'Frozen NLI evaluator metadata changed')
        for panel, seal, digest in panels:
            _require(data.file_hash(panel / 'seal.json') == digest, 'Panel resealed during NLI evaluation')
            reader._equal(data.verify_seal(panel, expected=_PANEL_FILES), seal, 'Panel changed during NLI evaluation')
        backend.verify_frozen_identity()
    unchanged()  # First panel manifest/gold hash/read occurs only after CPU metadata is frozen.
    manifest = reader._manifest(natural, samples[:18], panels[0][1])
    natural_labels = reader._released_rows(natural, samples[:18], manifest)
    control_manifest, control_labels = controls_reader._controls(controls, samples[18:], panels[1][1])
    packet_root = output / 'cases'
    packet_root.mkdir()
    records, packets, control_y, control_scores, control_unknown = [], [], [], [], []
    for index, (sample, row) in enumerate(zip(samples, complete['records'], strict=True)):
        document = _document(row)
        packet = dict(schema='nli-evaluation-case-v1', id=sample.id, source_id=sample.source_id,
            task=sample.task, generator=sample.generator, original_inputs=sample.inputs, readouts=row,
            document_max_unit_broadcast=document, human_semantic_review=dict(state='pending', complete_human_truth_created=False))
        if index < 18:
            released = natural_labels[index]
            native = focused._natural(sample, dict(units=row['units'], whole=document), released)
            native['document_max_unit_broadcast'] = native.pop('whole')
            native['document_max_unit_broadcast']['actual_whole_model_judgment'] = False
            packet.update(released_evaluation_record=released, native_evaluation=native)
            packets.append(packet)
        else:
            released = control_labels[index - 18]
            truth = int(released['construction_logical_outcome'] != 'supported')
            packet.update(released_evaluation_record=released, construction_binary_truth=truth,
                construction_risks={'unit': document['risk']}, unknown_fallback={'unit': document['unknown_fallback']})
            control_y.append(truth)
            control_scores.append(document['risk'])
            control_unknown.append(document['unknown_fallback'])
        path = packet_root / (data.text_hash(sample.id) + '.json')
        data.write_json(path, packet)
        records.append(dict(id=sample.id, file=str(path.relative_to(output)), sha256=data.file_hash(path)))
    views = [dict(packet, native_evaluation=dict(packet['native_evaluation'],
        whole=packet['native_evaluation']['document_max_unit_broadcast'])) for packet in packets]
    natural_summary = focused._natural_summary(views)
    for cohort in natural_summary['cohorts'].values():
        cohort['document_max_unit_broadcast'] = cohort.pop('whole')
        cohort['document_max_unit_broadcast']['actual_whole_model_judgment'] = False
    control_index, pair_rows = {sample.id: i for i, sample in enumerate(samples[18:])}, []
    for pair in control_manifest['pairs']:
        indices = [control_index[key] for key in pair['ids']]
        difference = next((control_scores[b] - control_scores[a] for a in indices for b in indices
                           if control_y[a] == 0 and control_y[b] == 1), None)
        pair_rows.append(dict(pair_id=pair['pair_id'], ids=pair['ids'],
            positive_minus_supported_risk={'unit': difference}, descriptive_only=True))
    summary = dict(whole_case_denominator=34, natural=natural_summary,
        controls=dict(denominator=16, unit=focused._metrics(control_y, control_scores, control_unknown),
            pairs=pair_rows, truth='construction only: supported=0; conflict/not_determined=1', empty_response_unit_risk=0.0),
        costs=complete['costs'], coverage=complete['coverage'], model_calls=0, tokenizer_only_validation=True,
        tokenizer_only_pair_preparations=104, target_label_selection_performed=False, threshold_sweeps=0,
        method_promoted=False, six_task_performance_claim=False, limitations=observer_plan['limitations'],
        document_score='document_max_unit_broadcast is a descriptive maximum over unit scores, not an actual whole-model judgment')
    data.write_json(output / 'summary.json', summary)
    unchanged()
    result = dict(schema='nli-evaluation-complete-v1', state='sealed_exploratory_native_unit_evaluation',
        cases=34, packets=34, records=records, evaluation_material_accessed=True,
        model_calls=0, tokenizer_only_validation=True, observation_root_seal_sha256=inventory[2])
    data.write_json(output / 'complete.json', result)
    nli._seal(output, [str(path.relative_to(output)) for path in output.rglob('*') if path.is_file()])
    try:
        data.verify_seal(output)
        unchanged()
    except Exception:
        (output / 'complete.json').unlink(missing_ok=True)
        (output / 'seal.json').unlink(missing_ok=True)
        raise
    return result

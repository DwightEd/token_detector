"""Read-only admission of sealed H003 observations across collector releases."""

from copy import deepcopy
from pathlib import Path

from . import data, semantic_qualification as backend, verdict_evaluation as reader
from . import verdict_qualification as qualification
from .datasets import read_samples
from .research import semantic_verdict as verdict


def effective_model_identity(identity):
    """Ignore only deployment packaging; every instrument/runtime field stays bound."""
    value = backend._model_identity(identity)
    if 'execution' in value:
        if not isinstance(value['execution'], dict):
            raise ValueError('Model execution identity must be a mapping')
        for name in ('installed_observer_wheel_sha256', 'CPU_module_target'):
            value['execution'].pop(name, None)
    return value


def equivalence_key(execution):
    """Identify a model observation independently of administrative run identity."""
    semantic = {'messages', 'labels', 'model_identity', 'scoring', 'admission'}
    administrative = {'protocol_sha256', 'source_identity_sha256'}
    if not isinstance(execution, dict) or set(execution) not in (semantic, semantic | administrative):
        raise ValueError('Unknown request execution fields cannot be ignored for reuse')
    return backend._digest({name: (effective_model_identity(execution[name])
                                  if name == 'model_identity' else execution[name])
                            for name in ('messages', 'labels', 'model_identity', 'scoring', 'admission')})


def _roster(samples, protocol, model, plan):
    rows, executions = [], {}
    for sample in samples:
        context, transformation = qualification._context(sample.inputs)
        requests = []
        for assignment in verdict.label_assignments():
            request = verdict.build_request(sample.source, sample.response, context, assignment)
            execution = {'messages': request['messages'], 'labels': list(verdict.LABELS),
                         'model_identity': model, 'protocol_sha256': plan['protocol_sha256'],
                         'source_identity_sha256': plan['source_identity_sha256'],
                         'scoring': {'use_cache': False, 'logits_to_keep': 1,
                                     'full_prefill': True, 'output_tokens': 0},
                         'admission': {'max_input_tokens': protocol['max_input_tokens']}}
            key = backend._digest(execution)
            executions[key] = execution
            requests.append({'request_key': key, 'label_assignment': assignment})
        rows.append({'id': sample.id, 'source_id': sample.source_id,
                     'input_text_sha256': {k: data.text_hash(v) for k, v in sample.inputs.items()},
                     'pre_generation_context_sha256': data.text_hash(context),
                     'context_transformation': transformation, 'requests': requests})
    reader._equal(plan.get('records'), rows, 'Donor original input/context/request roster changed')
    return rows, executions


def load_donor(root, model_identity, expected_inputs_sha256):
    """Verify a complete original archive and return immutable-origin response references.

    Only original inputs and observation artifacts are opened. No tokenizer, model,
    panel manifest or released annotation is accessed, and no artifact is rewritten.
    """
    root = Path(root).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError('Redirected or missing donor archive')
    inventory, seal, _, children = reader._observation_inventory(root)
    seal_hash = data.file_hash(root / 'seal.json')
    reader._verify_observation_snapshot(root, inventory, seal, seal_hash, children)
    plan = data.read_json(root / 'plan.json')
    protocol = qualification._protocol(root / 'protocol.json')
    model = backend._model_identity(data.read_json(root / 'model_identity.json'))
    source = data.read_json(root / 'source_identity.json')
    fields = {'schema', 'state', 'inputs_path', 'inputs_sha256', 'protocol_sha256', 'protocol',
              'model_identity_sha256', 'source_identity_sha256', 'records',
              'evaluation_material_accessed', 'case_ids_are_model_inputs',
              'engineering_failure_is_semantic_refutation'}
    if (not isinstance(plan, dict) or set(plan) != fields
            or plan.get('schema') != 'verdict-qualification-plan-v1'
            or plan.get('state') != 'observations_only'
            or any(plan.get(name) is not False for name in
                   ('evaluation_material_accessed', 'case_ids_are_model_inputs',
                    'engineering_failure_is_semantic_refutation'))
            or not reader._sha(expected_inputs_sha256)
            or plan.get('inputs_sha256') != expected_inputs_sha256
            or not reader._sha(plan.get('protocol_sha256'))
            or not isinstance(source, dict) or not isinstance(source.get('source_sha256'), dict)
            or not source['source_sha256']
            or any(not isinstance(name, str) or not name or not reader._sha(digest)
                   for name, digest in source['source_sha256'].items())):
        raise ValueError('Invalid donor plan/input/source identity')
    reader._equal(plan['protocol'], protocol, 'Donor protocol differs from frozen plan')
    if (plan['model_identity_sha256'] != backend._digest(model)
            or plan['source_identity_sha256'] != backend._digest(source)):
        raise ValueError('Donor archived model/source identity changed')
    reader._equal(effective_model_identity(model), effective_model_identity(model_identity),
                  'Donor effective model/runtime identity differs')
    for name, module in (('experiments/semantic_qualification.py', backend),
                         ('experiments.research/semantic_verdict.py', verdict)):
        if source['source_sha256'].get(name) != data.file_hash(Path(module.__file__)):
            raise ValueError('Donor scoring/prompt kernel requires its compatible reader: ' + name)
    if not isinstance(plan['inputs_path'], str):
        raise ValueError('Invalid donor original input path')
    inputs = Path(plan['inputs_path']).absolute()
    if inputs.name != 'inputs.jsonl' or inputs.resolve() != inputs or not inputs.is_file():
        raise ValueError('Donor requires unaliased original inputs.jsonl')
    if data.file_hash(inputs) != expected_inputs_sha256:
        raise ValueError('Donor original inputs changed')
    samples = list(read_samples(inputs))
    if len(samples) != protocol['expected_cases'] or len({s.id for s in samples}) != len(samples):
        raise ValueError('Donor must retain its whole original 18/16-case cohort')
    rows, executions = _roster(samples, protocol, model, plan)
    if (set(p.name for p in (root / 'requests').iterdir()) != set(executions)
            or set(p.name for p in (root / 'cases').iterdir()) != {data.text_hash(s.id) for s in samples}
            or len(executions) > protocol['max_forward_passes']):
        raise ValueError('Donor has missing or unreferenced requests/cases')
    responses = {key: qualification._cached(root / 'requests', key, execution)
                 for key, execution in executions.items()}
    cases, records = [], []
    for sample, row in zip(samples, rows, strict=True):
        case = reader._case(sample, row, root, responses, protocol)
        path = root / 'cases' / data.text_hash(sample.id) / 'case.json'
        cases.append(case)
        records.append({'id': sample.id, 'file': str(path.relative_to(root)),
                        'sha256': data.file_hash(path)})
    reader._equal(data.read_json(root / 'complete.json'),
                  reader._completion(cases, responses, records, protocol),
                  'Donor completion/coverage/costs changed')
    admitted = {}
    for key, execution in executions.items():
        equivalent = equivalence_key(execution)
        response = responses[key]
        if equivalent in admitted and admitted[equivalent]['response'] != response:
            raise ValueError('Conflicting responses share one observation identity')
        request_root = root / 'requests' / key
        names = qualification._members(request_root)
        admitted[equivalent] = {
            'response': deepcopy(response),
            'provenance': {'donor_root': str(root), 'root_seal_sha256': seal_hash,
                           'old_request_key': key,
                           'files_sha256': {name: data.file_hash(request_root / name) for name in sorted(names)},
                           'source_identity': deepcopy(source), 'model_identity': deepcopy(model)}}
    if data.file_hash(inputs) != expected_inputs_sha256:
        raise ValueError('Donor original inputs changed during admission')
    reader._verify_observation_snapshot(root, inventory, seal, seal_hash, children)
    return admitted

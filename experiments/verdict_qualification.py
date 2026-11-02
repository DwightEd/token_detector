"""Collect frozen whole-answer, finite-label observations without opening truth."""

import math
import os
from pathlib import Path

from . import data
from .datasets import read_samples
from .provenance import completed_identity, research_identity
from .research import semantic_verdict as verdict
from .research.semantic_probe import CONTEXT_POLICY
from .semantic_qualification import (
    ANNOTATION_NAMES, _artifact_path, _context, _digest, _model_identity, _seal,
)


DEFAULT_PROTOCOL = {
    'schema': 'verdict-qualification-v1', 'contract_version': verdict.CONTRACT_VERSION,
    'context_policy': CONTEXT_POLICY,
    'purpose': 'observations_only', 'expected_cases': 18, 'max_input_tokens': 4096,
    'labels': ['A', 'B', 'C'], 'permutations': 6, 'max_forward_passes': 108,
    'use_cache': False, 'logits_to_keep': 1, 'attention_implementation': 'sdpa',
}


def _protocol(path):
    value = data.read_json(path)
    population_fields = {'expected_cases', 'max_forward_passes'}
    if (not isinstance(value, dict) or any(type(value.get(k)) is not type(v) or value.get(k) != v
                                         for k, v in DEFAULT_PROTOCOL.items() if k not in population_fields)
            or type(value.get('expected_cases')) is not int
            or type(value.get('max_forward_passes')) is not int
            or (value['expected_cases'], value['max_forward_passes']) not in {(18, 108), (16, 96)}
            or set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}):
        raise ValueError('Unsupported frozen verdict-qualification-v1 protocol')
    return value


def _members(root):
    names = {'request.json', 'response.json'}
    if (root / 'backend_raw.json').is_file():
        names.add('backend_raw.json')
    if {p.name for p in root.iterdir()} != names | {'seal.json'}:
        raise ValueError('Unexpected or incomplete label request-cache members')
    return names


def _valid_label_tokens(value, labels):
    if not isinstance(value, dict):
        raise ValueError('Invalid label tokenizer preflight')
    ids, decodes = value.get('label_token_ids'), value.get('label_decodes')
    if (type(value.get('input_tokens')) is not int or value['input_tokens'] < 0
            or value.get('prefix_alignment_verified') is not True
            or not isinstance(ids, dict) or set(ids) != set(labels)
            or any(type(x) is not int or x < 0 for x in ids.values())
            or len(set(ids.values())) != len(labels)
            or not isinstance(decodes, dict) or set(decodes) != set(labels)
            or any(not isinstance(x, str) for x in decodes.values())):
        raise ValueError('Invalid label tokenizer preflight: distinct single-token metadata required')


def _valid_response(raw, count, labels, preflight=None):
    if not isinstance(raw, dict):
        raise ValueError('Label scorer returned invalid metadata')
    _valid_label_tokens(raw, labels)
    numbers = raw.get('label_logits')
    if (not isinstance(numbers, dict) or set(numbers) != set(labels)
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in numbers.values())
            or type(raw.get('full_vocab_logsumexp')) not in (float, int)
            or not math.isfinite(raw['full_vocab_logsumexp'])
            or raw['full_vocab_logsumexp'] + 1e-5 < max(numbers.values())
            or type(raw.get('input_tokens')) is not int or raw['input_tokens'] != count
            or type(raw.get('output_tokens')) is not int or raw['output_tokens'] != 0
            or type(raw.get('actual_forward_passes')) is not int or raw['actual_forward_passes'] != 1
            or raw.get('prefix_alignment_verified') is not True
            or type(raw.get('seconds')) not in (float, int) or not math.isfinite(raw['seconds'])
            or raw['seconds'] < 0):
        raise ValueError('Label scorer returned invalid logits/token-count/forward metadata')
    peak = max(numbers.values())
    log_label_mass = (peak - raw['full_vocab_logsumexp']) + math.log(math.fsum(
        math.exp(value - peak) for value in numbers.values()))
    if log_label_mass > 1e-10:
        raise ValueError('Label scorer full-vocabulary normalizer is below joint label logsumexp')
    if preflight is not None and any(raw.get(key) != preflight.get(key) for key in
            ('input_tokens', 'label_token_ids', 'label_decodes', 'prefix_alignment_verified')):
        raise ValueError('Label scorer tokenization changed after frozen preflight')


def _cached(requests, key, execution):
    root = _artifact_path(requests, key)
    if not root.exists():
        return None
    if not (root / 'seal.json').is_file():
        raise RuntimeError('Unfinished label request requires explicit artifact recovery: ' + key)
    data.verify_seal(root, expected=_members(root))
    if data.read_json(root / 'request.json') != execution or _digest(execution) != key:
        raise ValueError('Cached label request identity changed')
    response = data.read_json(root / 'response.json')
    count = response.get('input_tokens')
    if type(count) is not int or count < 0:
        raise ValueError('Cached label response token count is invalid')
    if response.get('status') == 'scored':
        if count > execution['admission']['max_input_tokens']:
            raise ValueError('Cached score violates input admission')
        _valid_response(response, count, execution['labels'])
        raw_path = root / 'backend_raw.json'
        if not raw_path.is_file() or data.read_json(raw_path) != {
                'value': {k: v for k, v in response.items() if k != 'status'}}:
            raise ValueError('Cached raw label response differs from scored response')
    elif response.get('status') == 'input_rejected':
        if (count <= execution['admission']['max_input_tokens'] or response != {
                'status': 'input_rejected', 'input_tokens': count, 'output_tokens': 0,
                'actual_forward_passes': 0, 'seconds': 0.0, 'reason': 'input_token_limit'}
                or (root / 'backend_raw.json').exists()):
            raise ValueError('Cached input rejection metadata changed')
    else:
        raise ValueError('Unknown cached label response status')
    return response


def _preflight(generator, execution):
    messages, labels = execution['messages'], execution['labels']
    if not callable(getattr(generator, 'validate_labels', None)):
        raise ValueError('Label scoring requires tokenizer-only validate_labels before any forward')
    count = generator.count_input_tokens(messages)
    if type(count) is not int or count < 0:
        raise ValueError('Label input counter must return a nonnegative integer')
    validation = generator.validate_labels(messages, labels)
    _valid_label_tokens(validation, labels)
    if validation['input_tokens'] != count:
        raise ValueError('Label tokenizer preflight input count differs from counter')
    return count, validation


def _execute(generator, requests, key, execution, preflight):
    cached = _cached(requests, key, execution)
    if cached is not None:
        return cached
    root = _artifact_path(requests, key)
    root.mkdir()
    data.write_json(root / 'request.json', execution)
    count, validation = preflight
    if count > execution['admission']['max_input_tokens']:
        response = {'status': 'input_rejected', 'input_tokens': count, 'output_tokens': 0,
                    'actual_forward_passes': 0, 'seconds': 0.0, 'reason': 'input_token_limit'}
    else:
        raw = generator.score_labels(execution['messages'], execution['labels'])
        data.write_json(root / 'backend_raw.json', {'value': raw})
        if isinstance(raw, dict) and 'status' in raw:
            raise ValueError('Label scorer must not supply collector status')
        _valid_response(raw, count, execution['labels'], validation)
        response = {**raw, 'status': 'scored'}
    data.write_json(root / 'response.json', response)
    _seal(root, sorted({'request.json', 'response.json'} |
                       ({'backend_raw.json'} if response['status'] == 'scored' else set())))
    return response


def _case(sample, row, responses, protocol):
    events = [{**item, 'execution': responses[item['request_key']]} for item in row['requests']]
    scored = [{'label_assignment': event['label_assignment'],
               'label_logits': event['execution']['label_logits'],
               'full_vocab_logsumexp': event['execution']['full_vocab_logsumexp']}
              for event in events if event['execution']['status'] == 'scored']
    reduced = (verdict.reduce_verdict(scored) if len(scored) == protocol['permutations'] else
               {'status': 'unknown', 'reason': 'input_token_limit', 'class_scores': None})
    return {'schema': 'verdict-qualification-case-v1', 'id': sample.id,
            'source_id': sample.source_id, 'task': sample.task, 'generator': sample.generator,
            'input_text_sha256': row['input_text_sha256'],
            'pre_generation_context_sha256': row['pre_generation_context_sha256'],
            'context_transformation': row['context_transformation'],
            'observations': events, 'verdict': reduced,
            'state': 'observations_sealed_not_evaluated'}


def qualify(inputs, protocol, output, generator, model_identity, *, request_builder=None,
            protocol_loader=None, sample_request_builder=None):
    """Score the entire original 18-case roster; reuse exact sealed observations.

    A scorer implements count_input_tokens(messages), score_labels(messages,
    labels), and tokenizer-only validate_labels(messages, labels).
    Partial cases reuse sealed requests; any unsealed request requires explicit
    artifact recovery and is never sent to the instrument again.
    """
    inputs, protocol, output = Path(inputs).resolve(), Path(protocol).resolve(), Path(output).resolve()
    if request_builder is not None and sample_request_builder is not None:
        raise ValueError('Choose one explicit request builder')
    if inputs.name != 'inputs.jsonl' or protocol.name.casefold() in ANNOTATION_NAMES:
        raise ValueError('Only original inputs.jsonl and an independent protocol may be read')
    if 'outputs' not in output.parts:
        raise ValueError('Verdict artifacts must be retained under outputs/')
    frozen = (protocol_loader or _protocol)(protocol)
    identity, source = _model_identity(model_identity), research_identity()
    build_request = request_builder or verdict.build_request
    if hasattr(generator, 'model_identity') and generator.model_identity != identity:
        raise ValueError('Generator and supplied model identities differ')
    input_hash, protocol_hash = data.file_hash(inputs), data.file_hash(protocol)
    samples = list(read_samples(inputs))
    if len(samples) != frozen['expected_cases']:
        raise ValueError(f"Whole original {frozen['expected_cases']}-case input roster is required")
    roster, executions = [], {}
    for sample in samples:
        context, transformation = _context(sample.inputs)
        requests = []
        for assignment in verdict.label_assignments():
            request = (sample_request_builder(sample, assignment) if sample_request_builder is not None else
                       build_request(sample.source, sample.response, context, assignment))
            execution = {'messages': request['messages'], 'labels': list(verdict.LABELS),
                         'model_identity': identity, 'protocol_sha256': protocol_hash,
                         'source_identity_sha256': _digest(source),
                         'scoring': {'use_cache': False, 'logits_to_keep': 1,
                                     'full_prefill': True, 'output_tokens': 0},
                         'admission': {'max_input_tokens': frozen['max_input_tokens']}}
            key = _digest(execution)
            executions[key] = execution
            requests.append({'request_key': key, 'label_assignment': assignment})
        roster.append({'id': sample.id, 'source_id': sample.source_id,
                       'input_text_sha256': {k: data.text_hash(v) for k, v in sample.inputs.items()},
                       'pre_generation_context_sha256': data.text_hash(context),
                       'context_transformation': transformation, 'requests': requests})
    if (len(executions) > frozen['max_forward_passes'] or
            any(len(row['requests']) != frozen['permutations'] for row in roster)):
        raise ValueError('Frozen permutation/forward budget changed')
    if data.file_hash(inputs) != input_hash or data.file_hash(protocol) != protocol_hash:
        raise ValueError('Inputs or protocol changed while being frozen')
    plan = {'schema': 'verdict-qualification-plan-v1', 'state': 'observations_only',
            'inputs_path': str(inputs), 'inputs_sha256': input_hash, 'protocol_sha256': protocol_hash,
            'protocol': frozen, 'model_identity_sha256': _digest(identity),
            'source_identity_sha256': _digest(source), 'records': roster,
            'evaluation_material_accessed': False, 'case_ids_are_model_inputs': False,
            'engineering_failure_is_semantic_refutation': False}
    output.mkdir(parents=True, exist_ok=True)
    lock = _artifact_path(output, '.qualification.lock')
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as error:
        raise RuntimeError('Existing qualification claim requires explicit liveness/recovery inspection') from error
    os.close(descriptor)
    try:
        metadata = {'plan.json': plan, 'protocol.json': frozen,
                    'model_identity.json': identity, 'source_identity.json': source}
        for name in [*metadata, 'complete.json', 'seal.json', 'cases', 'requests']:
            _artifact_path(output, name)
        if (output / 'plan.json').exists():
            if data.read_json(output / 'plan.json') != plan:
                raise ValueError('Resume plan differs: inputs/protocol/model/code/order must remain identical')
            for name, value in metadata.items():
                if data.read_json(output / name) != value:
                    raise ValueError('Frozen verdict identity changed: ' + name)
        else:
            if {p.name for p in output.iterdir()} != {lock.name}:
                raise RuntimeError('Unfinished qualification metadata requires explicit recovery')
            for name in ['protocol.json', 'model_identity.json', 'source_identity.json', 'plan.json']:
                data.write_json(output / name, metadata[name])
        cases, requests = output / 'cases', output / 'requests'
        cases.mkdir(exist_ok=True)
        requests.mkdir(exist_ok=True)
        if set(p.name for p in output.iterdir()) - {*metadata, lock.name, 'complete.json', 'seal.json', 'cases', 'requests'}:
            raise ValueError('Unexpected verdict qualification root artifacts')
        if set(p.name for p in cases.iterdir()) - {data.text_hash(s.id) for s in samples}:
            raise ValueError('Unexpected verdict case artifacts')
        if set(p.name for p in requests.iterdir()) - set(executions):
            raise ValueError('Unreferenced label cache requires explicit recovery')
        # Validate every existing cache/case before any additional model observation.
        cached = {key: _cached(requests, key, execution) for key, execution in executions.items()}
        old_cases = {}
        for sample, row in zip(samples, roster):
            root = _artifact_path(cases, data.text_hash(row['id']))
            if root.exists():
                members = {p.name for p in root.iterdir()}
                if members:
                    if members != {'case.json', 'seal.json'}:
                        raise RuntimeError('Unsealed verdict case requires explicit artifact recovery')
                    data.verify_seal(root, expected={'case.json'})
                    old_cases[row['id']] = data.read_json(root / 'case.json')
                    if any(cached[item['request_key']] is None for item in row['requests']):
                        raise ValueError('Sealed case is missing its exact request cache')
                    if old_cases[row['id']] != _case(sample, row, cached, frozen):
                        raise ValueError('Sealed verdict case differs from its frozen requests/reducer')
        if (output / 'seal.json').exists():
            data.verify_seal(output)
            if len(old_cases) != len(samples) or any(v is None for v in cached.values()):
                raise ValueError('Completed verdict qualification is incomplete')
        # All token boundaries are checked before the first forward, including
        # later permutations/cases and inputs that will be rejected for length.
        preflights = {key: _preflight(generator, execution) for key, execution in executions.items()
                      if cached[key] is None}
        records, case_values = [], []
        for sample, row in zip(samples, roster):
            for item in row['requests']:
                key = item['request_key']
                response = cached[key]
                if response is None:
                    response = _execute(generator, requests, key, executions[key], preflights[key])
                    cached[key] = response
            case = _case(sample, row, cached, frozen)
            root = _artifact_path(cases, data.text_hash(sample.id))
            if sample.id in old_cases:
                if old_cases[sample.id] != case:
                    raise ValueError('Sealed verdict case differs from its frozen requests/reducer')
            else:
                if (output / 'seal.json').exists():
                    raise ValueError('Completed verdict qualification is missing a case')
                root.mkdir(exist_ok=True)
                data.write_json(root / 'case.json', case)
                _seal(root, ['case.json'])
            records.append({'id': sample.id, 'file': str((root / 'case.json').relative_to(output)),
                            'sha256': data.file_hash(root / 'case.json')})
            case_values.append(case)
        completed_identity(source)
        if data.file_hash(inputs) != input_hash or data.file_hash(protocol) != protocol_hash:
            raise ValueError('Original inputs/protocol changed during qualification')
        if hasattr(generator, 'verify_frozen_identity'):
            generator.verify_frozen_identity()
        costs = list(cached.values())
        complete = {'schema': 'verdict-qualification-complete-v1',
                    'state': 'observations_sealed_not_evaluated', 'records': records,
                    'input_id_order_sha256': _digest([s.id for s in samples]),
                    'coverage': {'cases': len(records), 'permutations_per_case': frozen['permutations'],
                                 'scored_cases': sum(c['verdict'].get('status') != 'unknown' for c in case_values),
                                 'unknown_cases': sum(c['verdict'].get('status') == 'unknown' for c in case_values)},
                    'costs': {'scope': 'retained artifact lifetime; resume adds no repeated forwards',
                              'logical_requests': len(samples) * frozen['permutations'],
                              'unique_requests': len(costs),
                              'exact_request_reuses': len(samples) * frozen['permutations'] - len(costs),
                              'actual_forward_passes': sum(c['actual_forward_passes'] for c in costs),
                              'input_tokens_scored': sum(c['input_tokens'] for c in costs if c['status'] == 'scored'),
                              'output_tokens': 0, 'input_rejections': sum(c['status'] == 'input_rejected' for c in costs),
                              'seconds_forward': sum(c['seconds'] for c in costs)},
                    'evaluation_material_accessed': False, 'scalar_scores_created': True,
                    'scores_are_ground_truth': False, 'auroc_computed': False}
        names = list(metadata) + ['complete.json']
        names += [str((cases / data.text_hash(s.id) / name).relative_to(output))
                  for s in samples for name in ['case.json', 'seal.json']]
        names += [str((requests / key / name).relative_to(output)) for key in sorted(executions)
                  for name in sorted(_members(requests / key) | {'seal.json'})]
        if (output / 'complete.json').exists():
            if data.read_json(output / 'complete.json') != complete:
                raise ValueError('Existing verdict completion changed; explicit recovery required')
        else:
            data.write_json(output / 'complete.json', complete)
        if (output / 'seal.json').exists():
            data.verify_seal(output, expected=set(names))
        else:
            _seal(output, names)
        return complete
    finally:
        lock.unlink()

"""CPU-only, post-seal attachment of released annotations for human review.

This stage validates the entire observation population before opening or hashing
evaluation material. It creates coverage counts and review records, never model
calls, scalar scores, automatic semantic judgments, or token-level truth labels.
"""

from collections import Counter
from dataclasses import asdict
import json
import math
from pathlib import Path
import re

from . import data
from .datasets import read_jsonl, read_samples
from .provenance import completed_identity, research_identity
from .research import semantic_probe as probe
from . import semantic_qualification as qualification


DEFAULT_PROTOCOL = {
    'schema': 'semantic-qualification-evaluation-v1', 'purpose': 'posthoc_review_only',
    'required_panel_size': 18, 'whole_case_population': True,
    'scalar_scores_created': False, 'model_calls_allowed': False,
    'target_label_tuning_allowed': False,
    'literal_answer_inequality_is_semantic_conflict': False,
}
PANEL_SCHEMA = 'qualification-case-panel-v1'
STATE = 'observations_sealed_not_evaluated'
MODES = qualification.MODES
_METADATA = {'plan.json', 'protocol.json', 'model_identity.json',
             'source_identity.json', 'complete.json'}
_PANEL_FILES = {'inputs.jsonl', 'manifest.json', 'evaluation_only.jsonl'}
_OBSERVER_HELPERS = ('experiments/semantic_qualification.py', 'experiments.research/semantic_probe.py')


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _digest(value):
    return data.text_hash(_canonical(value))


def _equal(actual, expected, message):
    if _canonical(actual) != _canonical(expected):
        raise ValueError(message)


def _sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _member(root, name):
    """Admit a fixed artifact path before any read or member hashing."""
    path = root / name
    if path.resolve() != path or not path.is_file():
        raise ValueError('Redirected or missing evaluation-stage artifact: ' + name)
    return path


def _directory(root, name):
    path = root / name
    if path.resolve() != path or not path.is_dir():
        raise ValueError('Redirected or missing observation directory: ' + name)
    return path


def _observation_members(root):
    if {p.name for p in root.iterdir()} != _METADATA | {'cases', 'requests', 'seal.json'}:
        raise ValueError('Unexpected or incomplete observation root members')
    members = set(_METADATA)
    for name in _METADATA | {'seal.json'}:
        _member(root, name)
    for name in ('cases', 'requests'):
        parent = _directory(root, name)
        for entry in parent.iterdir():
            if not _sha(entry.name):
                raise ValueError('Unsafe observation artifact directory name')
            child = _directory(parent, entry.name)
            expected = {'case.json'} if name == 'cases' else {'request.json', 'response.json'}
            actual = {p.name for p in child.iterdir()}
            if name == 'requests' and 'backend_raw.json' in actual:
                expected.add('backend_raw.json')
            if actual != expected | {'seal.json'}:
                raise ValueError('Unexpected or incomplete observation artifact members')
            for member in expected | {'seal.json'}:
                path = _member(child, member)
                members.add(str(path.relative_to(root)))
    # Every path has been admitted before verify_seal can hash any member.
    initial_seal_hash = data.file_hash(root / 'seal.json')
    initial_seal = data.verify_seal(root, expected=members)
    for name in ('cases', 'requests'):
        for child in (root / name).iterdir():
            expected = {p.name for p in child.iterdir()} - {'seal.json'}
            data.verify_seal(child, expected=expected)
    if data.file_hash(root / 'seal.json') != initial_seal_hash:
        raise ValueError('Observation archive seal changed during initial verification')
    return members, initial_seal, initial_seal_hash


def _verify_observation_snapshot(root, members, initial_seal, initial_seal_hash):
    seal_path = _member(root, 'seal.json')
    if data.file_hash(seal_path) != initial_seal_hash:
        raise ValueError('Observation archive seal changed after initial verification')
    _equal(data.verify_seal(root, expected=members), initial_seal,
           'Observation archive seal changed after initial verification')
    if data.file_hash(root / 'seal.json') != initial_seal_hash:
        raise ValueError('Observation archive seal changed during verification')


def _protocol(path):
    value = data.read_json(path)
    if (not isinstance(value, dict) or set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
            or any(type(value.get(k)) is not type(v) or value[k] != v
                   for k, v in DEFAULT_PROTOCOL.items())):
        raise ValueError('Evaluator protocol must preserve the fixed whole-panel review prohibitions')
    return value


def _observer_identity(root, plan, evaluator_source):
    if not isinstance(plan, dict):
        raise ValueError('Invalid observation plan')
    protocol = qualification._protocol(root / 'protocol.json')
    model = qualification._model_identity(data.read_json(root / 'model_identity.json'))
    source = data.read_json(root / 'source_identity.json')
    if (not isinstance(source, dict) or not isinstance(source.get('source_sha256'), dict)
            or not source['source_sha256'] or any(not isinstance(k, str) or not k or not _sha(v)
                                                for k, v in source['source_sha256'].items())):
        raise ValueError('Missing or invalid archived observer source identity')
    if (plan.get('schema') != 'semantic-qualification-plan-v2'
            or plan.get('state') != 'observations_only'
            or plan.get('evaluation_material_accessed') is not False
            or plan.get('case_ids_are_model_inputs') is not False
            or plan.get('engineering_failure_is_semantic_refutation') is not False
            or not _sha(plan.get('protocol_sha256'))):
        raise ValueError('Invalid observation plan or observation-only isolation flags')
    _equal(plan.get('protocol'), protocol, 'Archived observer protocol differs from its plan')
    if (plan.get('model_identity_sha256') != _digest(model)
            or plan.get('source_identity_sha256') != _digest(source)):
        raise ValueError('Archived observer source/model identity differs from its plan')
    current = evaluator_source.get('source_sha256', {})
    if any(not _sha(source['source_sha256'].get(name))
           or source['source_sha256'][name] != current.get(name) for name in _OBSERVER_HELPERS):
        raise ValueError('Archived observation helpers are incompatible with this CPU reader; '
                         'restore a compatible CPU reader and retain the model observation archive')
    # Evaluation is a new release: only the reused helpers must match. The whole
    # evaluator identity may differ; archived model/source paths are never opened.
    return protocol, model, source


def _panel_inputs(panel, plan):
    for name in ('inputs.jsonl', 'manifest.json', 'seal.json'):
        _member(panel, name)
    manifest = data.read_json(panel / 'manifest.json')
    if not isinstance(manifest, dict) or manifest.get('schema') != PANEL_SCHEMA:
        raise ValueError('Only natural qualification-case-panel-v1 panels are supported; '
                         'constructed controls require a separate evaluation schema')
    seal = data.read_json(panel / 'seal.json')
    if (not isinstance(seal, dict) or set(seal) != _PANEL_FILES
            or any(not _sha(value) for value in seal.values())):
        raise ValueError('Invalid natural panel seal member set or identity')
    input_hash = data.file_hash(panel / 'inputs.jsonl')
    manifest_hash = data.file_hash(panel / 'manifest.json')
    if (input_hash != manifest.get('inputs_sha256') or input_hash != plan.get('inputs_sha256')
            or input_hash != seal['inputs.jsonl'] or manifest_hash != seal['manifest.json']
            or manifest.get('evaluation_only_sha256') != seal['evaluation_only.jsonl']):
        raise ValueError('Panel inputs or manifest identity changed')
    samples = list(read_samples(panel / 'inputs.jsonl'))
    if len(samples) != DEFAULT_PROTOCOL['required_panel_size']:
        raise ValueError('Evaluation requires the original whole 18-case population')
    rows = manifest.get('input_rows')
    if (not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows)
            or [r.get('id') for r in rows] != [s.id for s in samples]):
        raise ValueError('Panel manifest original ID roster/order changed')
    roster = [{'id': s.id, 'source_id': s.source_id,
               'response_sha256': data.text_hash(s.response)} for s in samples]
    _equal(plan.get('records'), roster, 'Observation plan original ID roster/order changed')
    for sample, row in zip(samples, rows, strict=True):
        expected = {'id': sample.id, 'source_id': sample.source_id,
                    'original_task': sample.task, 'generator': sample.generator,
                    **{name + '_sha256': data.text_hash(text) for name, text in sample.inputs.items()}}
        if any(row.get(k) != v for k, v in expected.items()) or not isinstance(row.get('cohort'), str):
            raise ValueError('Panel manifest does not bind the complete original inputs')
    counts = manifest.get('counts', {})
    if (not isinstance(counts, dict) or counts.get('total_inputs') != len(samples)
            or counts.get('cohorts') != dict(Counter(r['cohort'] for r in rows))):
        raise ValueError('Panel manifest population counts changed')
    if manifest.get('selection', {}).get('selected_id_order_sha256') != _digest([s.id for s in samples]):
        raise ValueError('Panel manifest original selection order changed')
    return samples, manifest, seal


def _requests(root, model, protocol):
    result = {}
    for child in sorted((root / 'requests').iterdir()):
        request = data.read_json(child / 'request.json')
        response = data.read_json(child / 'response.json')
        if not isinstance(request, dict) or _digest(request) != child.name or request.get('model_identity') != model:
            raise ValueError('Archived request-cache identity mismatch')
        decode = request.get('decode', {})
        if (set(request) != {'messages', 'model_identity', 'decode', 'admission'}
                or request.get('admission') != {'max_input_tokens': protocol['max_input_tokens']}
                or decode.get('do_sample') is not False or type(decode.get('num_beams')) is not int
                or decode.get('num_beams') != 1 or set(decode) != {'do_sample', 'num_beams', 'max_new_tokens'}
                or type(decode.get('max_new_tokens')) is not int
                or decode['max_new_tokens'] not in {protocol['proposer_max_new_tokens'],
                                                   protocol['observation_max_new_tokens']}):
            raise ValueError('Archived request decode/admission protocol mismatch')
        cap = decode['max_new_tokens']
        if (not isinstance(response, dict) or type(response.get('input_tokens')) is not int
                or response['input_tokens'] < 0 or type(response.get('output_tokens')) is not int
                or not 0 <= response['output_tokens'] <= cap
                or type(response.get('seconds')) not in {int, float}
                or not math.isfinite(response['seconds']) or response['seconds'] < 0
                or type(response.get('actual_model_calls')) is not int):
            raise ValueError('Invalid archived request execution metadata')
        if response.get('status') == 'input_rejected':
            expected = {'status': 'input_rejected', 'input_tokens': response['input_tokens'],
                        'output_tokens': 0, 'text': None, 'reason': 'input_token_limit',
                        'actual_model_calls': 0, 'seconds': 0.0}
            _equal(response, expected, 'Invalid archived input rejection')
            if response['input_tokens'] <= protocol['max_input_tokens'] or (child / 'backend_raw.json').exists():
                raise ValueError('Invalid archived input rejection/cache members')
        elif response.get('status') == 'generated':
            if (not isinstance(response.get('text'), str) or not isinstance(response.get('finish_reason'), str)
                    or response['actual_model_calls'] != 1
                    or response['input_tokens'] > protocol['max_input_tokens']
                    or not (child / 'backend_raw.json').is_file()):
                raise ValueError('Invalid archived generated response')
            raw = data.read_json(child / 'backend_raw.json')
            if not isinstance(raw, dict) or set(raw) != {'value'} or not isinstance(raw['value'], dict):
                raise ValueError('Invalid archived raw backend response')
            _equal(response, {**raw['value'], 'status': 'generated', 'actual_model_calls': 1,
                              'seconds': response['seconds']}, 'Archived backend response changed')
        else:
            raise ValueError('Incomplete or unsupported archived request status')
        result[child.name] = {'request': request, 'response': response}
    return result


def _event(event, request, protocol, model, requests, used):
    if not isinstance(event, dict):
        raise ValueError('Invalid archived observation event')
    cap = protocol['proposer_max_new_tokens'] if request['mode'] == 'proposal' else protocol['observation_max_new_tokens']
    execution = {'messages': request['messages'], 'model_identity': model,
                 'decode': {'do_sample': False, 'num_beams': 1, 'max_new_tokens': cap},
                 'admission': {'max_input_tokens': protocol['max_input_tokens']}}
    key = _digest(execution)
    if (event.get('mode') != request['mode'] or event.get('request_key') != key or key not in requests
            or type(event.get('cache_reused')) is not bool):
        raise ValueError('Observation event does not bind its original request')
    _equal(requests[key]['request'], execution, 'Observation request content changed')
    _equal(event.get('execution'), requests[key]['response'], 'Observation event response differs from cache')
    used.add(key)
    return event['execution']


def _case(sample, case, protocol, model, requests, used):
    if (not isinstance(case, dict) or case.get('schema') != 'semantic-qualification-case-v2' or case.get('state') != STATE
            or any(case.get(k) != getattr(sample, k) for k in ('id', 'source_id', 'task', 'generator'))):
        raise ValueError('Observation case original identity/state changed')
    _equal(case.get('input_text_sha256'), {k: data.text_hash(v) for k, v in sample.inputs.items()},
           'Observation case complete original input identity changed')
    context, transformation = qualification._context(sample.inputs)
    _equal(case.get('pre_generation_context_sha256'), data.text_hash(context),
           'Observation pre-generation context identity changed')
    _equal(case.get('context_transformation'), transformation, 'Observation context transformation changed')
    proposal = case.get('proposal', {})
    raw = _event(proposal, probe.build_proposer(sample.response, context=context), protocol, model, requests, used)
    if set(proposal) != {'mode', 'request_key', 'cache_reused', 'execution'}:
        raise ValueError('Invalid proposal event members')
    parsed = ({'frames': (), 'invalid': (), 'raw_claim_count': None,
               'envelope_error': {'code': 'input_token_limit', 'message': 'Proposal input rejected'}}
              if raw['status'] == 'input_rejected' else probe.parse_proposal_items(raw['text'], sample.response))
    frames = parsed['frames']
    limited = raw.get('finish_reason') == 'max_new_tokens'
    reasons = [raw['reason']] if raw['status'] == 'input_rejected' else []
    if limited:
        reasons.append('proposal_output_token_limit')
    if parsed['envelope_error'] is not None:
        reasons.append('proposal_envelope_invalid')
    if parsed['invalid']:
        reasons.append('invalid_proposed_frames')
    if any(not f.slots for f in frames):
        reasons.append('frames_without_slots')
    expected = {'raw_claim_count': parsed['raw_claim_count'], 'proposal_envelope_error': parsed['envelope_error'],
                'invalid_frames': parsed['invalid'], 'frames': [asdict(f) for f in frames],
                'unobserved_frames': [{'frame_index': i, 'status': 'unknown', 'reason': 'no_proposed_slots'}
                                      for i, f in enumerate(frames) if not f.slots],
                'coverage': probe.proposal_coverage(sample.response, frames),
                'proposal_output_limited': limited, 'coverage_limited_reasons': reasons}
    for key, value in expected.items():
        _equal(case.get(key), value, 'Observation case proposal/coverage structure changed: ' + key)
    events = case.get('observations')
    if not isinstance(events, list) or not all(isinstance(e, dict) for e in events):
        raise ValueError('Missing observation event population')
    expected_positions = [(i, j, mode) for i, frame in enumerate(frames)
                          for j, _ in enumerate(frame.slots) for mode in MODES]
    if [(e.get('frame_index'), e.get('slot_index'), e.get('mode')) for e in events] != expected_positions:
        raise ValueError('Incomplete, duplicated or reordered slot/mode observation population')
    for (i, j, mode), event in zip(expected_positions, events, strict=True):
        expected = {'frame_index': i, 'slot_index': j, 'mode': mode}
        try:
            request = getattr(probe, 'build_' + mode)(sample.source, sample.response, frames[i], frames[i].slots[j],
                                                     context=context)
        except probe.ProbeValidationError as error:
            expected.update(parse_status='request_invalid', error={'code': error.code, 'message': str(error)},
                            observation=asdict(probe.unknown_observation('request_invalid:' + error.code)))
        else:
            raw = _event(event, request, protocol, model, requests, used)
            expected.update({k: event[k] for k in ('request_key', 'cache_reused', 'execution')})
            if raw['status'] == 'input_rejected':
                expected.update(parse_status='input_rejected', observation=asdict(probe.unknown_observation('input_token_limit')))
            else:
                try:
                    observation = probe.parse_observation(raw['text'], sample.source, mode=mode)
                except probe.ProbeValidationError as error:
                    expected.update(parse_status='invalid', error={'code': error.code, 'message': str(error)},
                                    observation=asdict(probe.unknown_observation('invalid_output:' + error.code)))
                else:
                    expected.update(parse_status='valid', observation=asdict(observation))
        _equal(event, expected, 'Observation parsed output/status differs from its archived execution')
    return case


def _completion(cases, requests, records):
    events = [e for case in cases for e in [case['proposal'], *case['observations']]]
    observations = [e for e in events if e['mode'] != 'proposal']
    # Match the observer's summation order exactly, including floating seconds.
    costs = [requests[key]['response'] for key in sorted(requests)]
    coverage = {'cases': len(cases), 'raw_claims': sum(c['raw_claim_count'] or 0 for c in cases),
                'valid_frames': sum(c['coverage']['claims'] for c in cases),
                'valid_slots': sum(c['coverage']['slots'] for c in cases),
                'frames_without_slots': sum(c['coverage']['claims_without_slots'] for c in cases),
                'invalid_frames': sum(len(c['invalid_frames']) for c in cases),
                'proposal_envelope_failures': sum(c['proposal_envelope_error'] is not None for c in cases),
                'proposal_output_token_limits': sum(c['proposal_output_limited'] for c in cases),
                'cases_with_coverage_limitations': sum(bool(c['coverage_limited_reasons']) for c in cases),
                'cases_without_valid_frames': sum(not c['frames'] for c in cases),
                'semantic_exhaustiveness_verified': False,
                'observation_statuses': dict(Counter(e['parse_status'] for e in observations)),
                'observation_output_token_limits': sum(e.get('execution', {}).get('finish_reason') == 'max_new_tokens'
                                                       for e in observations),
                'unknown_observations': sum(e['observation']['status'] == 'unknown' for e in observations)}
    return {'schema': 'semantic-qualification-complete-v2', 'state': STATE, 'records': records,
            'input_id_order_sha256': _digest([c['id'] for c in cases]), 'coverage': coverage,
            'costs': {'logical_requests': sum('request_key' in e for e in events),
                      'scope': 'retained artifact lifetime; completed resume adds no model calls',
                      'unique_requests': len(requests), 'actual_model_calls': sum(c['actual_model_calls'] for c in costs),
                      'exact_request_reuses': sum('request_key' in e for e in events) - len(requests),
                      'input_tokens_generated': sum(c['input_tokens'] for c in costs if c['status'] == 'generated'),
                      'output_tokens_generated': sum(c['output_tokens'] for c in costs),
                      'input_rejections': sum(c['status'] == 'input_rejected' for c in costs),
                      'seconds_generation': sum(c['seconds'] for c in costs)},
            'evaluation_material_accessed': False, 'scalar_scores_created': False}


def _released_rows(panel, samples, manifest):
    rows = list(read_jsonl(panel / 'evaluation_only.jsonl'))
    if [row.get('id') for row in rows] != [s.id for s in samples]:
        raise ValueError('Released evaluation original ID roster/order changed')
    for row, sample, metadata in zip(rows, samples, manifest['input_rows'], strict=True):
        if (row.get('source_id') != sample.source_id or row.get('cohort') != metadata['cohort']
                or row.get('response_sha256') != data.text_hash(sample.response)):
            raise ValueError('Released evaluation ID/source/response join failed')
        expectation = row.get('human_mechanism_expectation')
        if (not isinstance(expectation, dict) or any(not isinstance(expectation.get(k), str)
                                                   for k in ('mechanism', 'expectation'))):
            raise ValueError('Missing natural-case human review expectations')
        annotation = row.get('released_annotation_record')
        if (not isinstance(annotation, dict) or annotation.get('id') != sample.id
                or annotation.get('response_sha256') != data.text_hash(sample.response)
                or annotation.get('granularity') not in {'token', 'sentence'}
                or annotation.get('coverage') not in {'complete', 'partial', 'annotated_sentences', 'unalignable'}
                or not isinstance(annotation.get('spans'), list)):
            raise ValueError('Invalid released natural annotation identity or scope')
        for span in annotation['spans']:
            if (not isinstance(span, dict) or type(span.get('start')) is not int
                    or type(span.get('end')) is not int
                    or not 0 <= span['start'] < span['end'] <= len(sample.response)
                    or 'label' not in span or (span['label'] is not None and
                        (type(span['label']) is not int or span['label'] not in (0, 1)))):
                raise ValueError('Invalid original released annotation span')
        # original_annotation.path is retained as opaque provenance, never opened.
    return rows


def _review(sample, case, released):
    paired = []
    for i, frame in enumerate(case['frames']):
        for j, slot in enumerate(frame['slots']):
            paired.append({'frame_index': i, 'slot_index': j, 'claim': frame['text'], 'slot': slot,
                           'modes': {event['mode']: event for event in case['observations']
                                     if event['frame_index'] == i and event['slot_index'] == j}})
    return {'schema': 'semantic-qualification-review-case-v1', 'id': sample.id,
            'source_id': sample.source_id, 'task': sample.task, 'generator': sample.generator,
            'original_inputs': sample.inputs, 'input_text_sha256': case['input_text_sha256'],
            'pre_generation_context': qualification._context(sample.inputs)[0],
            'pre_generation_context_sha256': case['pre_generation_context_sha256'],
            'context_transformation': case['context_transformation'],
            'proposal': case['proposal'], 'automatic_frames': case['frames'],
            'raw_claim_count': case['raw_claim_count'], 'proposal_envelope_error': case['proposal_envelope_error'],
            'proposal_output_limited': case['proposal_output_limited'],
            'invalid_frames': case['invalid_frames'], 'paired_observations': paired,
            'coverage': case['coverage'], 'coverage_limited_reasons': case['coverage_limited_reasons'],
            'unobserved_frames': case['unobserved_frames'],
            'unobserved_case_count': int(not paired), 'unproposed_facts_count': None,
            'released_evaluation_record': released,
            'annotation_scope': {'granularity': released['released_annotation_record']['granularity'],
                                 'native_units_preserved': True, 'per_token_fact_truth_created': False,
                                 'note': 'Released annotation spans retain their original scope; broad spans and '
                                         'sentence labels do not establish that every contained word is false.'},
            'human_semantic_review': {'state': 'pending', 'automatic_correctness_judgment': False,
                                      'literal_answer_inequality_is_semantic_conflict': False}}


def _summary(cases, released):
    slots = sum(c['coverage']['slots'] for c in cases)
    per_mode = {}
    for mode in MODES:
        events = [e for c in cases for e in c['observations'] if e['mode'] == mode]
        per_mode[mode] = {'whole_case_denominator': len(cases), 'proposed_slot_denominator': slots,
                          'retained_observations': len(events),
                          'parse_statuses': dict(Counter(e['parse_status'] for e in events)),
                          'answerable': sum(e['observation']['status'] == 'answerable' for e in events),
                          'unknown': sum(e['observation']['status'] == 'unknown' for e in events),
                          'output_limited': sum(e.get('execution', {}).get('finish_reason') == 'max_new_tokens'
                                                for e in events),
                          'unobserved_cases': sum(not c['coverage']['slots'] for c in cases)}
    return {'whole_case_denominator': len(cases), 'review_cases': len(cases),
            'cases_without_valid_frames': sum(not c['frames'] for c in cases),
            'unobserved_case_count': sum(not c['coverage']['slots'] for c in cases),
            'frames_without_slots': sum(c['coverage']['claims_without_slots'] for c in cases),
            'invalid_frames': sum(len(c['invalid_frames']) for c in cases),
            'proposal_envelope_failures': sum(c['proposal_envelope_error'] is not None for c in cases),
            'proposal_output_limited_cases': sum(c['proposal_output_limited'] for c in cases),
            'cases_with_coverage_limitations': sum(bool(c['coverage_limited_reasons']) for c in cases),
            'valid_frames': sum(c['coverage']['claims'] for c in cases), 'proposed_slots': slots,
            'slot_modes': per_mode, 'unproposed_facts_count': None,
            'released_annotation_granularities': dict(Counter(r['released_annotation_record']['granularity']
                                                             for r in released)),
            'semantic_exhaustiveness_verified': False, 'automatic_accuracy_created': False,
            'scalar_scores_created': False, 'auroc_created': False, 'model_calls': 0,
            'scope': 'Whole exploratory panel and proposed-slot coverage only; human semantic review pending.'}


def evaluate(observations, panel, output, protocol):
    """Validate a complete sealed natural panel, then attach released review data.

    The output must be new and disjoint from both input trees. A failed release
    retains its frozen evaluator plan; it is never overwritten or model-rerun.
    """
    observations, panel, output, protocol = (Path(p).resolve() for p in (observations, panel, output, protocol))
    if (observations.is_relative_to(panel) or panel.is_relative_to(observations)
            or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (observations, panel))):
        raise ValueError('Evaluation output and observation/panel inputs must be disjoint artifact trees')
    if protocol.is_relative_to(panel) or protocol.is_relative_to(observations) or protocol.name.casefold() in qualification.ANNOTATION_NAMES:
        raise ValueError('Evaluation protocol must be independent of observation/panel artifacts')
    if output.exists():
        raise FileExistsError('Evaluation output already exists; retained results must not be overwritten')
    frozen_protocol = _protocol(protocol)
    protocol_hash, evaluator_source = data.file_hash(protocol), research_identity()
    members, observation_seal, observation_seal_hash = _observation_members(observations)
    plan = data.read_json(observations / 'plan.json')
    complete = data.read_json(observations / 'complete.json')
    if not isinstance(complete, dict) or complete.get('state') != STATE:
        raise ValueError('Evaluation requires observations_sealed_not_evaluated completion state')
    observer_protocol, model, observer_source = _observer_identity(observations, plan, evaluator_source)
    samples, manifest, panel_seal = _panel_inputs(panel, plan)
    requests, used, cases, records = _requests(observations, model, observer_protocol), set(), [], []
    if {p.name for p in (observations / 'cases').iterdir()} != {data.text_hash(s.id) for s in samples}:
        raise ValueError('Observation case population differs from original inputs')
    for sample in samples:
        path = observations / 'cases' / data.text_hash(sample.id) / 'case.json'
        cases.append(_case(sample, data.read_json(path), observer_protocol, model, requests, used))
        records.append({'id': sample.id, 'file': str(path.relative_to(observations)), 'sha256': data.file_hash(path)})
    if used != set(requests):
        raise ValueError('Unreferenced or missing archived request population')
    _equal(complete, _completion(cases, requests, records), 'Observation completion/coverage/cost structure changed')
    # Finish every observation/input check before evaluating, including a second
    # seal pass to catch changes made while the archive was being interpreted.
    _verify_observation_snapshot(observations, members, observation_seal, observation_seal_hash)
    for name in ('inputs.jsonl', 'manifest.json', 'seal.json'):
        _member(panel, name)
    _member(protocol.parent, protocol.name)
    if (data.file_hash(panel / 'inputs.jsonl') != plan['inputs_sha256']
            or data.file_hash(panel / 'manifest.json') != panel_seal['manifest.json']
            or data.read_json(panel / 'seal.json') != panel_seal
            or data.file_hash(protocol) != protocol_hash):
        raise ValueError('Frozen inputs, manifest, panel seal or evaluator protocol changed during preflight')
    completed_identity(evaluator_source)
    observation_identity = {'observation_root_seal_sha256': observation_seal_hash,
                            'observer_source_identity_sha256': _digest(observer_source),
                            'observer_model_identity_sha256': _digest(model),
                            'observer_protocol': observer_protocol,
                            'observer_original_protocol_sha256': plan['protocol_sha256']}
    evaluation_plan = {'schema': 'semantic-qualification-evaluation-plan-v1', 'state': 'evaluator_frozen_before_release',
                       'observations': str(observations), 'panel': str(panel),
                       'original_id_order': [s.id for s in samples],
                       'inputs_sha256': plan['inputs_sha256'], 'manifest_sha256': panel_seal['manifest.json'],
                       'observation_identity': observation_identity,
                       'evaluator_protocol_sha256': protocol_hash,
                       'evaluator_source_identity_sha256': _digest(evaluator_source),
                       'evaluation_material_accessed': False, 'model_calls_allowed': False}
    output.mkdir(parents=True, exist_ok=False)
    data.write_json(output / 'plan.json', evaluation_plan)
    data.write_json(output / 'protocol.json', frozen_protocol)
    data.write_json(output / 'evaluator_source_identity.json', evaluator_source)
    data.write_json(output / 'observer_identity.json', {'source_identity': observer_source, 'model_identity': model,
                                                       **observation_identity})
    # RELEASE GATE: no evaluation_only path has been read, hashed, resolved, or
    # opened above. Panel-wide verification is intentionally deferred until now.
    if {p.name for p in panel.iterdir()} != _PANEL_FILES | {'seal.json'}:
        raise ValueError('Unexpected natural panel artifact members')
    for name in _PANEL_FILES | {'seal.json'}:
        _member(panel, name)
    _equal(data.verify_seal(panel, expected=_PANEL_FILES), panel_seal, 'Panel identity changed at release')
    released = _released_rows(panel, samples, manifest)
    review_root = output / 'cases'
    review_root.mkdir()
    review_records = []
    for sample, case, annotation in zip(samples, cases, released, strict=True):
        path = review_root / (data.text_hash(sample.id) + '.json')
        data.write_json(path, _review(sample, case, annotation))
        review_records.append({'id': sample.id, 'file': str(path.relative_to(output)), 'sha256': data.file_hash(path)})
    summary = _summary(cases, released)
    data.write_json(output / 'summary.json', summary)
    data.write_json(output / 'panel_identity.json', {'schema': PANEL_SCHEMA, 'seal': panel_seal,
                                                   'panel_seal_sha256': data.file_hash(panel / 'seal.json'),
                                                   'manifest': manifest})
    completed_identity(evaluator_source)
    _verify_observation_snapshot(observations, members, observation_seal, observation_seal_hash)
    _equal(data.verify_seal(panel, expected=_PANEL_FILES), panel_seal, 'Panel changed during review export')
    if data.file_hash(protocol) != protocol_hash:
        raise ValueError('Evaluator protocol changed during review export')
    result = {'schema': 'semantic-qualification-evaluation-complete-v1', 'state': 'review_sealed_human_review_pending',
              'records': review_records, 'coverage': summary, 'evaluation_material_accessed': True,
              'scalar_scores_created': False, 'model_calls': 0,
              'evaluator_source_identity_sha256': _digest(evaluator_source),
              'observer_source_identity_sha256': _digest(observer_source)}
    data.write_json(output / 'complete.json', result)
    names = {'plan.json', 'protocol.json', 'evaluator_source_identity.json', 'observer_identity.json',
             'summary.json', 'panel_identity.json', 'complete.json'} | {r['file'] for r in review_records}
    data.write_json(output / 'seal.json', {name: data.file_hash(output / name) for name in sorted(names)})
    data.verify_seal(output, expected=names)
    return result

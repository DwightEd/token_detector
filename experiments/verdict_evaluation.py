"""Read sealed H003 observations on CPU and export whole-case human review.

The full observation archive is checked before manifest/annotation access. This
reader never calls an instrument, fits a rule, projects labels onto tokens, or
turns conditional label diagnostics into factual correctness judgments.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from . import data, verdict_qualification as qualification
from .datasets import read_jsonl, read_samples
from .provenance import completed_identity, research_identity
from .research import semantic_verdict as verdict


DEFAULT_PROTOCOL = {
    'schema': 'verdict-qualification-evaluation-v1', 'purpose': 'posthoc_review_only',
    'required_panel_size': 18, 'whole_case_population': True,
    'model_calls_allowed': False, 'target_label_tuning_allowed': False,
    'automatic_correctness_judgment': False, 'token_score_projection_allowed': False,
    'auroc_allowed': False, 'fitted_calibration_allowed': False,
    'gold_based_threshold_allowed': False,
}
PANEL_SCHEMA = 'qualification-case-panel-v1'
STATE = 'observations_sealed_not_evaluated'
_METADATA = {'plan.json', 'protocol.json', 'model_identity.json',
             'source_identity.json', 'complete.json'}
_PANEL_FILES = {'inputs.jsonl', 'manifest.json', 'evaluation_only.jsonl'}
_OBSERVER_HELPERS = (
    'experiments/verdict_qualification.py', 'experiments.research/semantic_verdict.py',
    'experiments/semantic_qualification.py', 'experiments.research/semantic_probe.py',
)


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


def _member(root, name, directory=False):
    path = root / name
    if path.resolve() != path or not (path.is_dir() if directory else path.is_file()):
        raise ValueError('Redirected or missing evaluation-stage artifact: ' + name)
    return path


def _seal_content(root, names):
    """Admit every seal key/path before any archive member can be hashed."""
    raw = _member(root, 'seal.json').read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError('Sealed artifact member set changed')
    for name, digest in value.items():
        if (not isinstance(name, str) or not name or Path(name).is_absolute()
                or str(Path(name)) != name or any(p in {'.', '..'} for p in Path(name).parts)
                or not _sha(digest)):
            raise ValueError('Unsafe sealed artifact member path or digest')
        _member(root, name)
    return value, raw


def _observation_inventory(root):
    if {p.name for p in root.iterdir()} != _METADATA | {'cases', 'requests', 'seal.json'}:
        raise ValueError('Unexpected or incomplete observation root members')
    members, children = set(_METADATA), []
    for name in _METADATA | {'seal.json'}:
        _member(root, name)
    for name in ('cases', 'requests'):
        parent = _member(root, name, directory=True)
        for entry in parent.iterdir():
            if not _sha(entry.name):
                raise ValueError('Unsafe observation artifact directory name')
            child = _member(parent, entry.name, directory=True)
            expected = {'case.json'} if name == 'cases' else {'request.json', 'response.json'}
            actual = {p.name for p in child.iterdir()}
            if name == 'requests' and 'backend_raw.json' in actual:
                expected.add('backend_raw.json')
            if actual != expected | {'seal.json'}:
                raise ValueError('Unexpected or incomplete observation artifact members')
            for member in expected | {'seal.json'}:
                members.add(str(_member(child, member).relative_to(root)))
            children.append((child, expected))
    initial_seal, initial_seal_bytes = _seal_content(root, members)
    for child, expected in children:
        _seal_content(child, expected)
    return members, initial_seal, initial_seal_bytes, children


def _verify_observation_snapshot(root, members, initial_seal, initial_hash, children):
    for name in members:
        _member(root, name)
    if data.file_hash(_member(root, 'seal.json')) != initial_hash:
        raise ValueError('Observation archive seal changed after initial verification')
    _equal(data.verify_seal(root, expected=members), initial_seal,
           'Observation archive seal changed after initial verification')
    for child, expected in children:
        data.verify_seal(child, expected=expected)
    if data.file_hash(root / 'seal.json') != initial_hash:
        raise ValueError('Observation archive seal changed during verification')


def _protocol(path):
    value = data.read_json(path)
    if (not isinstance(value, dict) or set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
            or any(type(value.get(k)) is not type(v) or value.get(k) != v
                   for k, v in DEFAULT_PROTOCOL.items())):
        raise ValueError('Evaluator protocol must preserve whole-panel review prohibitions')
    return value


def _observer_identity(root, plan, evaluator_source):
    protocol = qualification._protocol(_member(root, 'protocol.json'))
    model = qualification._model_identity(data.read_json(_member(root, 'model_identity.json')))
    source = data.read_json(_member(root, 'source_identity.json'))
    if (not isinstance(source, dict) or not isinstance(source.get('source_sha256'), dict)
            or not source['source_sha256'] or any(not isinstance(k, str) or not k or not _sha(v)
                                                for k, v in source['source_sha256'].items())):
        raise ValueError('Missing or invalid archived observer source identity')
    fields = {'schema', 'state', 'inputs_path', 'inputs_sha256', 'protocol_sha256', 'protocol',
              'model_identity_sha256', 'source_identity_sha256', 'records',
              'evaluation_material_accessed', 'case_ids_are_model_inputs',
              'engineering_failure_is_semantic_refutation'}
    if (not isinstance(plan, dict) or set(plan) != fields
            or plan.get('schema') != 'verdict-qualification-plan-v1'
            or plan.get('state') != 'observations_only'
            or plan.get('evaluation_material_accessed') is not False
            or plan.get('case_ids_are_model_inputs') is not False
            or plan.get('engineering_failure_is_semantic_refutation') is not False
            or not _sha(plan.get('protocol_sha256')) or not _sha(plan.get('inputs_sha256'))
            or not isinstance(plan.get('inputs_path'), str)):
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
    return protocol, model, source


def _roster(samples, protocol, model, plan):
    """Rebuild the complete logical request manifest using original strings."""
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
            key = _digest(execution)
            executions[key] = execution
            requests.append({'request_key': key, 'label_assignment': assignment})
        rows.append({'id': sample.id, 'source_id': sample.source_id,
                     'input_text_sha256': {k: data.text_hash(v) for k, v in sample.inputs.items()},
                     'pre_generation_context_sha256': data.text_hash(context),
                     'context_transformation': transformation, 'requests': requests})
    _equal(plan.get('records'), rows, 'Observation full-input/order/request manifest changed')
    if (len(rows) != 18 or sum(len(row['requests']) for row in rows) != 108
            or not 0 < len(executions) <= protocol['max_forward_passes']):
        raise ValueError('Whole 18-case/108-logical-request population is required')
    return rows, executions


def _case(sample, row, root, responses, protocol):
    path = root / 'cases' / data.text_hash(sample.id) / 'case.json'
    value = data.read_json(_member(path.parent, path.name))
    # _case is the collector's pure cache-to-case reduction; it never observes a
    # model. Reusing it also preserves the distinction between unscored unknown
    # and the model semantic class named unresolved.
    _equal(value, qualification._case(sample, row, responses, protocol),
           'Observation case differs from its exact requests/reduction')
    return value


def _completion(cases, responses, records, protocol):
    costs = list(responses.values())  # collector insertion order, never directory order
    return {'schema': 'verdict-qualification-complete-v1', 'state': STATE, 'records': records,
            'input_id_order_sha256': _digest([c['id'] for c in cases]),
            'coverage': {'cases': len(cases), 'permutations_per_case': protocol['permutations'],
                         'scored_cases': sum(c['verdict'].get('status') != 'unknown' for c in cases),
                         'unknown_cases': sum(c['verdict'].get('status') == 'unknown' for c in cases)},
            'costs': {'scope': 'retained artifact lifetime; resume adds no repeated forwards',
                      'logical_requests': len(cases) * protocol['permutations'],
                      'unique_requests': len(costs),
                      'exact_request_reuses': len(cases) * protocol['permutations'] - len(costs),
                      'actual_forward_passes': sum(c['actual_forward_passes'] for c in costs),
                      'input_tokens_scored': sum(c['input_tokens'] for c in costs if c['status'] == 'scored'),
                      'output_tokens': 0, 'input_rejections': sum(c['status'] == 'input_rejected' for c in costs),
                      'seconds_forward': sum(c['seconds'] for c in costs)},
            'evaluation_material_accessed': False, 'scalar_scores_created': True,
            'scores_are_ground_truth': False, 'auroc_computed': False}


def _manifest(panel, samples, seal):
    manifest = data.read_json(panel / 'manifest.json')
    if not isinstance(manifest, dict) or manifest.get('schema') != PANEL_SCHEMA:
        raise ValueError('Only natural qualification-case-panel-v1 panels are supported')
    if (manifest.get('inputs_sha256') != seal['inputs.jsonl']
            or manifest.get('evaluation_only_sha256') != seal['evaluation_only.jsonl']):
        raise ValueError('Panel manifest input/released-material identity changed')
    rows = manifest.get('input_rows')
    if (not isinstance(rows, list) or not all(isinstance(r, dict) for r in rows)
            or [r.get('id') for r in rows] != [s.id for s in samples]):
        raise ValueError('Panel manifest original ID roster/order changed')
    for sample, row in zip(samples, rows, strict=True):
        expected = {'id': sample.id, 'source_id': sample.source_id,
                    'original_task': sample.task, 'generator': sample.generator,
                    **{name + '_sha256': data.text_hash(text) for name, text in sample.inputs.items()}}
        if any(row.get(k) != v for k, v in expected.items()) or not isinstance(row.get('cohort'), str):
            raise ValueError('Panel manifest does not bind the complete original inputs')
    counts = manifest.get('counts', {})
    if (not isinstance(counts, dict) or counts.get('total_inputs') != len(samples)
            or counts.get('cohorts') != dict(Counter(r['cohort'] for r in rows))
            or manifest.get('selection', {}).get('selected_id_order_sha256') != _digest([s.id for s in samples])):
        raise ValueError('Panel manifest population counts/selection order changed')
    return manifest


def _released_rows(panel, samples, manifest):
    rows = list(read_jsonl(panel / 'evaluation_only.jsonl'))
    if [row.get('id') for row in rows] != [s.id for s in samples]:
        raise ValueError('Released evaluation original ID roster/order changed')
    for row, sample, metadata in zip(rows, samples, manifest['input_rows'], strict=True):
        if (row.get('source_id') != sample.source_id or row.get('cohort') != metadata['cohort']
                or row.get('response_sha256') != data.text_hash(sample.response)):
            raise ValueError('Released evaluation ID/source/response join failed')
        expectation = row.get('human_mechanism_expectation')
        annotation = row.get('released_annotation_record')
        if (not isinstance(expectation, dict) or any(not isinstance(expectation.get(k), str)
                                                   for k in ('mechanism', 'expectation'))
                or not isinstance(annotation, dict) or annotation.get('id') != sample.id
                or annotation.get('response_sha256') != data.text_hash(sample.response)
                or annotation.get('granularity') not in {'token', 'sentence'}
                or annotation.get('coverage') not in {'complete', 'partial', 'annotated_sentences', 'unalignable'}
                or not isinstance(annotation.get('spans'), list)):
            raise ValueError('Invalid released natural annotation identity, expectation or scope')
        for span in annotation['spans']:
            if (not isinstance(span, dict) or type(span.get('start')) is not int
                    or type(span.get('end')) is not int
                    or not 0 <= span['start'] < span['end'] <= len(sample.response)
                    or 'label' not in span or (span['label'] is not None and
                        (type(span['label']) is not int or span['label'] not in (0, 1)))):
                raise ValueError('Invalid original released annotation span')
        # original_annotation paths remain opaque provenance, never followed.
    return rows


def _review(sample, case, released):
    return {'schema': 'verdict-qualification-review-case-v1', 'id': sample.id,
            'source_id': sample.source_id, 'task': sample.task, 'generator': sample.generator,
            'original_inputs': sample.inputs, 'input_text_sha256': case['input_text_sha256'],
            'pre_generation_context': qualification._context(sample.inputs)[0],
            'pre_generation_context_sha256': case['pre_generation_context_sha256'],
            'context_transformation': case['context_transformation'],
            'observations': case['observations'], 'verdict': case['verdict'],
            'unscored_unknown': case['verdict'].get('status') == 'unknown',
            'released_evaluation_record': released,
            'annotation_scope': {'granularity': released['released_annotation_record']['granularity'],
                                 'coverage': released['released_annotation_record']['coverage'],
                                 'native_units_preserved': True, 'per_token_fact_truth_created': False,
                                 'note': 'Whole-answer classes and native annotation units do not establish '
                                         'the truth of each contained token.'},
            'human_semantic_review': {'state': 'pending', 'automatic_correctness_judgment': False},
            'limitations': list(verdict.LIMITATIONS)}


def _diagnostics(cases):
    scored = [c['verdict'] for c in cases if c['verdict'].get('status') != 'unknown']
    permutations = [p for v in scored for p in v['permutations']]
    masses = [p['off_label_mass'] for p in permutations]
    events = [event for case in cases for event in case['observations']]
    return {'cases': len(cases), 'scored_cases': len(scored), 'unscored_unknown_cases': len(cases) - len(scored),
            'retained_label_requests': len(events),
            'scored_label_requests': sum(event['execution']['status'] == 'scored' for event in events),
            'input_rejected_label_requests': sum(event['execution']['status'] == 'input_rejected' for event in events),
            'aggregate_score_ties': sum(sum(value == max(v['class_scores'].values())
                                             for value in v['class_scores'].values()) > 1 for v in scored),
            'unanimous_unique_argmax_cases': sum(v['permutation_agreement']['unanimous_unique_argmax'] for v in scored),
            'cases_with_tied_permutations': sum(v['permutation_agreement']['tied_permutations'] > 0 for v in scored),
            'tied_permutations': sum(v['permutation_agreement']['tied_permutations'] for v in scored),
            'unique_permutation_argmax_counts': dict(Counter(p['top_classes'][0] for p in permutations
                                                            if len(p['top_classes']) == 1)),
            'scored_permutations': len(permutations),
            'permutation_diagnostic_scope': 'Fully scored cases; all raw events, including partial cases, remain in review packets.',
            'off_label_mass': {'count': len(masses), 'min': min(masses) if masses else None,
                               'max': max(masses) if masses else None,
                               'mean': math.fsum(masses) / len(masses) if masses else None},
            'full_vocab_class_mass_means': {k: math.fsum(p['full_vocab_class_masses'][k] for p in permutations)
                                           / len(permutations) if permutations else None
                                           for k in verdict.SEMANTIC_CLASSES}}


def _summary(cases, released):
    cohorts = {}
    for name in dict.fromkeys(r['cohort'] for r in released):
        cohorts[name] = _diagnostics([case for case, row in zip(cases, released, strict=True) if row['cohort'] == name])
    return {'whole_case_denominator': len(cases), 'review_cases': len(cases),
            **_diagnostics(cases), 'cohorts': cohorts,
            'released_annotation_granularities': dict(Counter(r['released_annotation_record']['granularity'] for r in released)),
            'automatic_accuracy_created': False, 'new_scalar_scores_created': False,
            'token_score_projection_created': False, 'auroc_created': False,
            'fitted_calibration_created': False, 'model_calls': 0,
            'scope': 'Whole exploratory panel diagnostics; human semantic review pending. '
                     'Model unresolved is a semantic class; unknown means input was not fully scored.'}


def evaluate(observations, panel, output, protocol):
    """Validate the entire frozen archive, then release annotation review data.

    A failed release retains its evaluator plan and is never overwritten. Source
    compatibility is limited to actual reused observer helpers, so changing a
    reader or CLI does not require repeating any archived model observations.
    """
    observations, panel, output, protocol = (Path(p).resolve() for p in (observations, panel, output, protocol))
    if (observations.is_relative_to(panel) or panel.is_relative_to(observations)
            or any(output.is_relative_to(p) or p.is_relative_to(output) for p in (observations, panel))):
        raise ValueError('Evaluation output and observation/panel inputs must be disjoint artifact trees')
    if (protocol.is_relative_to(panel) or protocol.is_relative_to(observations)
            or protocol.is_relative_to(output) or protocol.name.casefold() in qualification.ANNOTATION_NAMES):
        raise ValueError('Evaluation protocol must be independent of artifact trees')
    if output.exists():
        raise FileExistsError('Evaluation output already exists; retained results must not be overwritten')
    # Inventory and seal-key admission precede every file hash, including source
    # identity hashing. evaluation_only is deliberately untouched until release.
    members, observation_seal, observation_seal_bytes, children = _observation_inventory(observations)
    for name in ('inputs.jsonl', 'manifest.json', 'seal.json'):
        _member(panel, name)
    _member(protocol.parent, protocol.name)
    panel_seal_bytes = (panel / 'seal.json').read_bytes()
    panel_seal = json.loads(panel_seal_bytes)
    if (not isinstance(panel_seal, dict) or set(panel_seal) != _PANEL_FILES
            or any(not _sha(v) for v in panel_seal.values())):
        raise ValueError('Invalid natural panel seal member set or identity')
    observation_seal_hash = hashlib.sha256(observation_seal_bytes).hexdigest()
    _verify_observation_snapshot(observations, members, observation_seal, observation_seal_hash, children)
    frozen_protocol = _protocol(protocol)
    protocol_hash, evaluator_source = data.file_hash(protocol), research_identity()
    panel_seal_hash = hashlib.sha256(panel_seal_bytes).hexdigest()
    plan = data.read_json(_member(observations, 'plan.json'))
    observer_protocol, model, observer_source = _observer_identity(observations, plan, evaluator_source)
    if (data.file_hash(_member(panel, 'inputs.jsonl')) != plan['inputs_sha256']
            or plan['inputs_sha256'] != panel_seal['inputs.jsonl']):
        raise ValueError('Original input file identity changed')
    samples = list(read_samples(_member(panel, 'inputs.jsonl')))
    if len(samples) != frozen_protocol['required_panel_size']:
        raise ValueError('Evaluation requires the original whole 18-case population')
    roster, executions = _roster(samples, observer_protocol, model, plan)
    if ({p.name for p in (observations / 'cases').iterdir()} != {data.text_hash(s.id) for s in samples}
            or {p.name for p in (observations / 'requests').iterdir()} != set(executions)):
        raise ValueError('Observation case/request population differs from complete original ownership')
    responses = {key: qualification._cached(observations / 'requests', key, request)
                 for key, request in executions.items()}
    if any(value is None for value in responses.values()):
        raise ValueError('Missing archived label response')
    cases, records = [], []
    for sample, row in zip(samples, roster, strict=True):
        cases.append(_case(sample, row, observations, responses, observer_protocol))
        path = observations / 'cases' / data.text_hash(sample.id) / 'case.json'
        records.append({'id': sample.id, 'file': str(path.relative_to(observations)), 'sha256': data.file_hash(path)})
    _equal(data.read_json(_member(observations, 'complete.json')), _completion(cases, responses, records, observer_protocol),
           'Observation completion/coverage/cost structure changed')
    _verify_observation_snapshot(observations, members, observation_seal, observation_seal_hash, children)
    if (data.file_hash(_member(panel, 'inputs.jsonl')) != plan['inputs_sha256']
            or data.file_hash(_member(panel, 'seal.json')) != panel_seal_hash
            or data.read_json(_member(panel, 'seal.json')) != panel_seal
            or data.file_hash(_member(protocol.parent, protocol.name)) != protocol_hash):
        raise ValueError('Frozen inputs, panel seal or evaluator protocol changed during preflight')
    completed_identity(evaluator_source)
    observation_identity = {'observation_root_seal_sha256': observation_seal_hash,
                            'observer_source_identity_sha256': _digest(observer_source),
                            'observer_model_identity_sha256': _digest(model),
                            'observer_protocol': observer_protocol,
                            'observer_original_protocol_sha256': plan['protocol_sha256']}
    evaluation_plan = {'schema': 'verdict-qualification-evaluation-plan-v1', 'state': 'evaluator_frozen_before_release',
                       'observations': str(observations), 'panel': str(panel),
                       'original_id_order': [s.id for s in samples], 'inputs_sha256': plan['inputs_sha256'],
                       'panel_seal_sha256': panel_seal_hash, 'manifest_sha256': panel_seal['manifest.json'],
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
    # RELEASE GATE: all observations/inputs/identities/totals are now fixed and
    # checked. Neither manifest nor evaluation_only has been opened or hashed.
    _verify_observation_snapshot(observations, members, observation_seal, observation_seal_hash, children)
    if (data.file_hash(_member(panel, 'seal.json')) != panel_seal_hash
            or data.file_hash(_member(panel, 'inputs.jsonl')) != plan['inputs_sha256']
            or data.file_hash(_member(protocol.parent, protocol.name)) != protocol_hash):
        raise ValueError('Frozen inputs, panel seal or evaluator protocol changed at release')
    completed_identity(evaluator_source)
    if {p.name for p in panel.iterdir()} != _PANEL_FILES | {'seal.json'}:
        raise ValueError('Unexpected natural panel artifact members')
    for name in _PANEL_FILES | {'seal.json'}:
        _member(panel, name)
    _equal(data.verify_seal(panel, expected=_PANEL_FILES), panel_seal, 'Panel identity changed at release')
    manifest = _manifest(panel, samples, panel_seal)
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
                                                   'panel_seal_sha256': panel_seal_hash, 'manifest': manifest})
    completed_identity(evaluator_source)
    _verify_observation_snapshot(observations, members, observation_seal, observation_seal_hash, children)
    _equal(data.verify_seal(panel, expected=_PANEL_FILES), panel_seal, 'Panel changed during review export')
    if data.file_hash(panel / 'seal.json') != panel_seal_hash or data.file_hash(protocol) != protocol_hash:
        raise ValueError('Panel seal or evaluator protocol changed during review export')
    result = {'schema': 'verdict-qualification-evaluation-complete-v1', 'state': 'review_sealed_human_review_pending',
              'records': review_records, 'coverage': summary, 'evaluation_material_accessed': True,
              'new_scalar_scores_created': False, 'scalar_scores_created': False, 'model_calls': 0,
              'evaluator_source_identity_sha256': _digest(evaluator_source),
              'observer_source_identity_sha256': _digest(observer_source)}
    data.write_json(output / 'complete.json', result)
    names = {'plan.json', 'protocol.json', 'evaluator_source_identity.json', 'observer_identity.json',
             'summary.json', 'panel_identity.json', 'complete.json'} | {r['file'] for r in review_records}
    data.write_json(output / 'seal.json', {name: data.file_hash(output / name) for name in sorted(names)})
    data.verify_seal(output, expected=names)
    try:
        _verify_observation_snapshot(observations, members, observation_seal, observation_seal_hash, children)
        if data.file_hash(_member(panel, 'seal.json')) != panel_seal_hash or data.file_hash(protocol) != protocol_hash:
            raise ValueError('Panel seal or evaluator protocol changed during final publication')
        _equal(data.verify_seal(panel, expected=_PANEL_FILES), panel_seal, 'Panel changed during final publication')
        completed_identity(evaluator_source)
    except Exception:
        # Retain the failed review files and frozen plan, without success markers.
        (output / 'complete.json').unlink(missing_ok=True)
        (output / 'seal.json').unlink(missing_ok=True)
        raise
    return result

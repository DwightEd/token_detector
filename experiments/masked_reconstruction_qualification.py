"""Fixed response-masking bridge; known toy slots isolate the question compiler.

Source-only names form the complete candidate pool. Response values are retained
only for exact erasure/alignment; neither original person is copied into Q.
"""
from copy import deepcopy
from dataclasses import asdict
import re

from . import data, reconstruction_qualification as collector
from .research import masked_reconstruction as kernel

DEFAULT_PROTOCOL = dict(collector.DEFAULT_PROTOCOL,
    schema='masked-reconstruction-qualification-v1',
    purpose='response_masked_question_bridge_observations',
    expected_unique_queries=48, expected_completions=96,
    query_policy='erase_all_two_person_literals_from_entire_response_fixed_question_v1',
    masked_question_contract='h007_known_person_slots_events_correct_explicit_bijection_v1',
    question_template=kernel.QUESTION_TEMPLATE,
    logical_completion_requests=192, fixed_question_template=True,
    natural_frontend_claim=False, response_suffix_preserved=True,
    candidate_pool_depends_only_on_source=True)
_require = collector._require


def _protocol(path):
    value = data.read_json(path)
    _require(isinstance(value, dict)
        and not set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
        and all(type(value.get(key)) is type(expected) and value[key] == expected
            for key, expected in DEFAULT_PROTOCOL.items() if key != 'original_inputs_sha256')
        and isinstance(value.get('original_inputs_sha256'), str)
        and re.fullmatch('[0-9a-f]{64}', value['original_inputs_sha256']) is not None,
        'Exact fixed masked-question bridge protocol required')
    return value


def _prepare(samples):
    """Adapt the pure compiler to the shared completion collector contract."""
    _require(len(samples) == 48 and len({sample.id for sample in samples}) == 48,
             'All48 originals must be retained before masked-question admission')
    prepared = kernel.prepare_masked_queries(samples)
    _require(len(prepared['originals']) == 48 and len(prepared['sources']) == 8
        and len(prepared['queries']) == 48 and len(prepared['requests']) == 96,
        'Complete8-source/48-question/96-completion bridge required; no selection')
    by_id = {sample.id: sample for sample in samples}
    originals, references = [], {}
    for original in prepared['originals']:
        sample = by_id[original['id']]
        _require(original['status'] == 'ok' and len(original['occurrences']) == 2,
                 'Unknown compiler slots are retained but forbid this complete controlled qualification')
        slots = []
        for slot in original['occurrences']:
            key = slot['query_key']
            _require(key in prepared['queries'] and sample.response[slot['start']:slot['end']] == slot['text'],
                     'Original exact occurrence or masked question changed')
            references[key] = references.get(key, 0) + 1
            slots.append(deepcopy(slot))
        originals.append(dict(sample=asdict(sample), source_key=original['source_sha256'],
            alignment=dict(status='ok', reason=None, slots=slots, target_span=original['target_span']),
            input_text_sha256={key: data.text_hash(value) for key, value in sample.inputs.items()}))
    _require(set(references) == set(prepared['queries']) and set(references.values()) == {2},
             'Both response directions must collapse to each exact masked query')
    candidates = [dict(query_key=row['query_key'], candidate=row['candidate'],
                       messages=deepcopy(row['messages']))
                  for key, row in sorted(prepared['requests'].items())]
    _require(len(candidates) * 2 == DEFAULT_PROTOCOL['logical_completion_requests'],
             'All192 original-slot-candidate instances must retain their96 exact observations')
    return dict(originals=originals, sources=deepcopy(prepared['sources']),
                queries=deepcopy(prepared['queries']), candidates=candidates)


def _plan(*args):
    result = collector._plan(*args)
    result.update(schema='masked-reconstruction-plan-v1',
        response_used_in_model_requests=True, response_person_values_exposed_to_qa=False,
        response_request_information='fixed_literal_erasure_with_full_suffix',
        known_person_slots_and_event_anchors_are_toy_design_assumptions=True,
        natural_data_coverage_established=False)
    return result


def _completion(prepared, executions, responses, frozen):
    result = collector._completion(prepared, executions, responses, frozen)
    result.update(schema='masked-reconstruction-complete-v1',
        response_values_exposed_to_qa=False, masked_response_used_in_qa=True,
        source_defined_question_control='sealed_H008_reused_only_during_separate_CPU_review',
        known_person_slots_and_event_anchors_are_toy_design_assumptions=True)
    result['coverage']['logical_completion_requests'] = frozen['logical_completion_requests']
    return result


def qualify(inputs, protocol, output, backend, model_identity):
    return collector._collect(inputs, protocol, output, backend, model_identity,
        protocol_loader=_protocol, prepare=_prepare, plan_builder=_plan, completion_builder=_completion)

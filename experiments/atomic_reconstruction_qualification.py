"""Fixed H013 one-conjunct profile for the shared complete-candidate collector."""
from copy import deepcopy
from dataclasses import asdict
import re
from . import data, reconstruction_qualification as collector
from . import masked_reconstruction_qualification as prior
from .research import atomic_reconstruction as kernel

DEFAULT_PROTOCOL = dict(prior.DEFAULT_PROTOCOL,
    schema='atomic-reconstruction-qualification-v1', purpose='one_conjunct_masked_statement_bridge',
    expected_unique_queries=16, expected_completions=32,
    query_policy='known_affirmative_R_conjunct_only_same_H011_question_template',
    response_suffix_preserved=False, primary_control='sealed_H011_burden_zero_only',
    non_target_conjunct_bundle_removed=True, dangling_other_instruction_retained=True,
    canonical_first_conjunct_terminal='. ', empirical_suffix_invariance_claim=False)


def _protocol(path):
    value = data.read_json(path)
    collector._require(isinstance(value, dict)
        and not set(value) - set(DEFAULT_PROTOCOL) - {'notes', 'hypothesis_id'}
        and all(type(value.get(k)) is type(v) and value[k] == v for k, v in DEFAULT_PROTOCOL.items()
                if k != 'original_inputs_sha256')
        and isinstance(value.get('original_inputs_sha256'), str)
        and re.fullmatch('[0-9a-f]{64}', value['original_inputs_sha256']) is not None,
        'Exact predeclared one-conjunct bundle required')
    return value


def _prepare(samples):
    collector._require(len(samples) == 48 and len({s.id for s in samples}) == 48, 'All48 originals required')
    parsed = kernel.prepare_atomic_queries(samples)
    collector._require(parsed['counts'] == dict(originals=48, sources=8, logical_queries=96,
        unique_queries=16, logical_requests=192, unique_requests=32, unknown_occurrences=0),
        'Complete one-conjunct population required without dropping unknowns')
    lookup = {s.id: s for s in samples}
    originals, references = [], {}
    for row in parsed['originals']:
        sample = lookup[row['id']]
        for slot in row['occurrences']:
            collector._require(sample.response[slot['start']:slot['end']] == slot['text'], 'Exact original slot changed')
            references[slot['query_key']] = references.get(slot['query_key'], 0) + 1
        originals.append(dict(sample=asdict(sample), source_key=row['source_sha256'],
            alignment=dict(status='ok', reason=None, target_span=row['target_span'],
                target_text=sample.response[row['target_span'][0]:row['target_span'][1]],
                slots=deepcopy(row['occurrences'])),
            input_text_sha256={k: data.text_hash(v) for k, v in sample.inputs.items()}))
    collector._require(set(references) == set(parsed['queries']) and set(references.values()) == {6},
        'Both response directions and all three burdens must map to each exact atomic question')
    return dict(originals=originals, sources=deepcopy(parsed['sources']), queries=deepcopy(parsed['queries']),
        candidates=[dict(query_key=row['query_key'], candidate=row['candidate'], messages=deepcopy(row['messages']))
                    for key, row in sorted(parsed['requests'].items())])


def _plan(*args):
    result = prior._plan(*args)
    result.update(schema='atomic-reconstruction-plan-v1',
        interpretation='known affirmative conjunct bundle; not a unique attention or predicate-count cause',
        primary_control='sealed_H011_burden_zero_only', response_suffix_removed_by_design=True,
        response_request_information='select_known_R_conjunct_then_literal_erasure_and_canonical_terminal')
    return result


def _completion(prepared, executions, responses, frozen):
    result = prior._completion(prepared, executions, responses, frozen)
    result.update(schema='atomic-reconstruction-complete-v1',
        primary_control='sealed_H011_burden_zero_only', suffix_invariance_is_by_construction=True,
        method_promoted=False, natural_task_performance_claim=False)
    return result


def qualify(inputs, protocol, output, backend, model_identity):
    return collector._collect(inputs, protocol, output, backend, model_identity,
        protocol_loader=_protocol, prepare=_prepare, plan_builder=_plan, completion_builder=_completion)

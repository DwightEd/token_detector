"""Pure H011 masked-response QA construction for the exact H007 toy contract.

Known person slots are an admitted toy frontend, not natural-text coverage or a
unique edit-localization certificate. Source parsing validates the grammar but
never exports its relation bindings as answers. No model, files, gold, or scores.
"""
from collections.abc import Mapping
import hashlib
import json
import re

from .source_reconstruction import EOS_POLICY, FILLERS, align_response, source_queries

CONTRACT_VERSION = 'masked-response-reconstruction-v1'
QUESTION_TEMPLATE = ('Using only the source, give the full person name for [TARGET] in the masked statement. '
                     '[OTHER] denotes a different source person.\nMasked statement:\n{masked_response}')
_PLACEHOLDERS = ('[TARGET]', '[OTHER]')
_SUFFIXES = {' '.join(FILLERS[:burden]) for burden in (0, 4, 12)}


def _key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()


def _text_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest() if isinstance(text, str) else None


def _mask_person_values(response, target, other):
    """Erase every literal occurrence; overlapping values cannot be guessed."""
    if (not isinstance(response, str) or not all(isinstance(v, str) and v for v in (target, other))
            or target == other):
        return dict(status='unknown', reason='invalid_person_mask_values', mask_spans=[])
    if any(marker in response for marker in _PLACEHOLDERS):
        return dict(status='unknown', reason='placeholder_collision', mask_spans=[])
    spans = sorted((dict(start=match.start(), end=match.start() + len(value),
                         text=value, replacement=replacement)
                    for value, replacement in ((target, '[TARGET]'), (other, '[OTHER]'))
                    for match in re.finditer('(?=' + re.escape(value) + ')', response)),
                   key=lambda span: (span['start'], span['end']))
    if {span['replacement'] for span in spans} != set(_PLACEHOLDERS):
        return dict(status='unknown', reason='missing_literal_person_value', mask_spans=spans)
    if any(left['end'] > right['start'] for left, right in zip(spans, spans[1:])):
        return dict(status='unknown', reason='overlapping_literal_person_values', mask_spans=spans)
    parts, cursor = [], 0
    for span in spans:
        assert response[span['start']:span['end']] == span['text']
        parts.extend((response[cursor:span['start']], span['replacement']))
        cursor = span['end']
    masked = ''.join([*parts, response[cursor:]])
    if any(value in masked for value in (target, other)):
        return dict(status='unknown', reason='literal_person_value_remains', mask_spans=spans)
    return dict(status='ok', reason=None, masked_response=masked, mask_spans=spans)


def _unknown_slots(reason, slots=()):
    by_index = {slot['index']: slot for slot in slots}
    return [dict(index=index, start=None, end=None, text=None, event=None)
            | {key: value for key, value in by_index.get(index, {}).items() if key != 'index'}
            | dict(status='unknown', reason=reason, candidate_index=None,
                   query_key=None, observation_keys=[], mask_spans=[])
            for index in range(2)]


def prepare_masked_queries(samples):
    """Return deduplicated source/query/completion payloads and all original slots.

Each sample supplies id/source/response, either mapping keys or attributes.
Candidate values/order depend only on source. Original target values, offsets,
candidate indices and mask maps live only in originals, never in QA prefixes.
Malformed toy cases retain two explicit unknown occurrences; callers must admit
the complete cohort rather than silently filtering unknown cases.
"""
    result = dict(schema=CONTRACT_VERSION, sources={}, queries={}, requests={}, originals=[])
    seen = set()
    for sample in samples:
        get = sample.get if isinstance(sample, Mapping) else lambda key: getattr(sample, key, None)
        identifier, source, response = (get(key) for key in ('id', 'source', 'response'))
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('sample IDs must be unique nonempty strings')
        seen.add(identifier)
        source_hash = _text_hash(source)
        original = dict(id=identifier, source_sha256=source_hash, response_sha256=_text_hash(response),
                        status='unknown', reason=None, target_span=None, occurrences=[])
        result['originals'].append(original)
        prepared = source_queries(source)
        if source_hash is not None:
            metadata = {key: prepared[key] for key in ('status', 'reason', 'people', 'events', 'source_sha256')}
            metadata.update(source=source, candidates=list(prepared['people']))
            if prepared['status'] == 'ok':
                metadata['relation_family'] = prepared['relation_family']
            if source_hash in result['sources'] and result['sources'][source_hash] != metadata:
                raise ValueError('source identity collision')
            result['sources'][source_hash] = metadata
        if prepared['status'] != 'ok':
            reason = 'source:' + prepared['reason']
            original.update(reason=reason, occurrences=_unknown_slots(reason))
            continue
        aligned = align_response(source, response)
        original['target_span'] = aligned['target_span']
        if aligned['status'] != 'ok':
            reason = 'response:' + aligned['reason']
            original.update(reason=reason, occurrences=_unknown_slots(reason))
            continue
        slots = aligned['slots']
        reason = None
        if response[aligned['target_span'][1]:] not in _SUFFIXES:
            reason = 'response_filler_suffix_outside_toy_contract'
        elif {slot['text'] for slot in slots} != set(prepared['people']):
            reason = 'response_slots_not_two_distinct_source_people'
        masked = [_mask_person_values(response, slot['text'], next(
            person for person in prepared['people'] if person != slot['text'])) for slot in slots]
        reason = reason or next((item['reason'] for item in masked if item['status'] != 'ok'), None)
        if reason:
            original.update(reason=reason, occurrences=_unknown_slots(reason, slots))
            continue
        original.update(status='ok', reason=None)
        for slot, mask in zip(slots, masked):
            question = QUESTION_TEMPLATE.format(masked_response=mask['masked_response'])
            query_key = _key(dict(contract=CONTRACT_VERSION, source=source, question=question))
            messages = [dict(prepared['queries'][0]['messages'][0]), dict(role='user', content=json.dumps(
                dict(source=source, question=question), ensure_ascii=False, separators=(',', ':')))]
            query = dict(query_key=query_key, source_sha256=source_hash, source=source,
                         event=slot['event'], masked_response=mask['masked_response'], question=question,
                         candidates=list(prepared['people']), messages=messages, eos_policy=dict(EOS_POLICY))
            if query_key in result['queries'] and result['queries'][query_key] != query:
                raise ValueError('query identity collision')
            result['queries'][query_key] = query
            observation_keys = []
            for candidate in query['candidates']:
                observation_key = _key(dict(query_key=query_key, candidate=candidate, eos_policy=EOS_POLICY))
                result['requests'][observation_key] = dict(observation_key=observation_key, query_key=query_key,
                    messages=[dict(message) for message in messages], candidate=candidate, eos_policy=dict(EOS_POLICY))
                observation_keys.append(observation_key)
            original['occurrences'].append(dict(**{key: slot[key] for key in ('index', 'start', 'end', 'text', 'event')},
                status='ok', reason=None, candidate_index=prepared['people'].index(slot['text']),
                query_key=query_key, observation_keys=observation_keys, mask_spans=mask['mask_spans']))
    occurrences = [slot for original in result['originals'] for slot in original['occurrences']]
    result['counts'] = dict(originals=len(result['originals']), sources=len(result['sources']),
        logical_queries=len(occurrences), unique_queries=len(result['queries']),
        logical_requests=sum(len(slot['observation_keys']) for slot in occurrences),
        unique_requests=len(result['requests']), unknown_occurrences=sum(slot['status'] != 'ok' for slot in occurrences))
    return result

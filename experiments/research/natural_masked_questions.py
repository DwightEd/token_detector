"""Pure H012 cloze artifacts from frozen NER spans; never QA or risk scores.

All source pools precede response-value inspection. Requested native labels may
allocate empty pools; candidate members/order never depend on response values.
H011's failed toy masked QA remains unresolved: compilation is not answerability.
"""
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import re

CONTRACT_VERSION = 'natural-masked-cloze-inventory-v1'
OFFSET_UNIT = 'Python Unicode code points; half-open'
UNKNOWN = 'unknown_not_assessed'
_RESERVED = re.compile(r'\[(?:TARGET|OTHER_[0-9]+)\]')
_SEMANTIC = {key: UNKNOWN for key in ('anchor', 'candidate_domain_completeness', 'alias',
                                    'arithmetic', 'assertion', 'factuality', 'answerability')}


def _hash(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest() if isinstance(value, str) else None


def _key(value):
    return _hash(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False))


def _get(item, key):
    return item.get(key) if isinstance(item, Mapping) else getattr(item, key, None)


def _label(entity):
    value = entity.get('type') if isinstance(entity, Mapping) else None
    return value if isinstance(value, str) and value else None


def _entities(field):
    value = field.get('entities') if isinstance(field, Mapping) else None
    return value if isinstance(value, list) else []


def _field_reason(field, text):
    if not isinstance(text, str) or not isinstance(field, Mapping):
        return 'invalid_text_or_field_inventory'
    if (field.get('status') != 'processed' or field.get('text_sha256') != _hash(text)
            or type(field.get('characters')) is not int or field['characters'] != len(text)
            or field.get('offset_unit') != OFFSET_UNIT or not isinstance(field.get('entities'), list)):
        return 'untrusted_field_identity_or_status'
    ids = [e.get('occurrence_id') if isinstance(e, Mapping) else None for e in field['entities']]
    if any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids):
        return 'invalid_or_duplicate_occurrence_id'
    return None


def _entity_reason(entity, text):
    if not isinstance(entity, Mapping) or _label(entity) is None:
        return 'invalid_entity_structure_or_native_label'
    literal, start, end = (entity.get(k) for k in ('text', 'start_char', 'end_char'))
    if not isinstance(literal, str) or not literal:
        return 'empty_or_invalid_entity_literal'
    if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
        return 'invalid_entity_bounds'
    if text[start:end] != literal:
        return 'entity_slice_mismatch'
    return None


def _provenance(entity, owner, index):
    result = {key: entity.get(key) for key in ('occurrence_id', 'text', 'type', 'start_char', 'end_char')} if isinstance(entity, Mapping) else {}
    return dict(result, owner_id=owner, entity_index=index)


def _mask_integrity(response, literal, masked, mappings):
    """Require full original-literal erasure, rather than marker substring bans."""
    required = {(m.start(), m.start() + len(literal)) for m in re.finditer('(?=' + re.escape(literal) + ')', response)}
    targets = {(m['start_char'], m['end_char']) for m in mappings if m['replacement'] == '[TARGET]' and m['text'] == literal}
    if required != targets:
        return 'unmasked_original_literal_occurrence'
    parts, cursor = [], 0
    for mapping in mappings:
        start, end = mapping['start_char'], mapping['end_char']
        if (type(start) is not int or type(end) is not int or not cursor <= start < end <= len(response)
                or response[start:end] != mapping['text'] or literal in response[cursor:start]):
            return 'invalid_mask_provenance'
        parts.extend((response[cursor:start], mapping['replacement']))
        cursor = end
    if literal in response[cursor:] or ''.join([*parts, response[cursor:]]) != masked:
        return 'invalid_mask_provenance'
    return None


def _mask(response, entities, target):
    reason = _entity_reason(target, response)
    if reason:
        return reason, None, [], 0
    if _RESERVED.search(response):
        return 'reserved_placeholder_collision', None, [], 0
    literal, label = target['text'], target['type']
    spans = {}
    literal_matches = list(re.finditer('(?=' + re.escape(literal) + ')', response))
    for index, match in enumerate(literal_matches):
        coordinate = match.start(), match.start() + len(literal)
        spans[coordinate] = dict(target=True, provenance=[dict(kind='target_literal', literal_match_index=index)])
    for index, entity in enumerate(entities):
        if _label(entity) != label:
            continue
        reason = _entity_reason(entity, response)
        if reason:
            return reason, None, [], len(literal_matches)
        coordinate = entity['start_char'], entity['end_char']
        item = spans.setdefault(coordinate, dict(target=False, provenance=[]))
        item['provenance'].append(dict(kind='ner_occurrence', entity_index=index, occurrence_id=entity['occurrence_id']))
    coordinates = sorted(spans)
    if any(left[1] > right[0] for left, right in zip(coordinates, coordinates[1:])):
        return 'overlapping_mask_spans', None, [], len(literal_matches)
    mappings, parts, cursor, other = [], [], 0, 0
    for start, end in coordinates:
        item = spans[start, end]
        if item['target']:
            replacement = '[TARGET]'
        else:
            other += 1
            replacement = f'[OTHER_{other:04d}]'
        parts.extend((response[cursor:start], replacement))
        mappings.append(dict(start_char=start, end_char=end, text=response[start:end], replacement=replacement,
                             provenance=item['provenance']))
        cursor = end
    masked = ''.join([*parts, response[cursor:]])
    reason = _mask_integrity(response, literal, masked, mappings)
    if reason:
        return reason, None, mappings, len(literal_matches)
    return None, masked, mappings, len(literal_matches)


def prepare_natural_questions(samples, inventories):
    """Retain every case/NER occurrence as an unscored cloze or explicit unknown.

Inventories may be ordered H010 JSONL-shaped rows or an exact id->row mapping.
No input mutation, files, NLP rerun, tokenizer, instructions, answers or scores.
"""
    samples = list(samples)
    ids = [_get(sample, 'id') for sample in samples]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError('Unique nonempty original IDs required')
    if isinstance(inventories, Mapping):
        if set(inventories) != set(ids):
            raise ValueError('Complete matching inventory ID population required')
        rows = [inventories[i] for i in ids]
        if any(not isinstance(row, Mapping) or row.get('id', owner) != owner for owner, row in zip(ids, rows)):
            raise ValueError('Inventory owner identity changed')
    else:
        rows = list(inventories)
        if len(rows) != len(ids) or [_get(row, 'id') for row in rows] != ids:
            raise ValueError('Complete ordered inventory ID population required')
    result = dict(schema=CONTRACT_VERSION, originals=[], source_pools={}, proposals=[], counts={}, limitations=[
        'Frozen external-supervision NER spans are emitted occurrences, not semantic coverage.',
        'H011 masked toy QA failed; this compiler creates no executable QA or repaired method.',
        'Anchor, source-domain completeness, aliases, arithmetic and assertion/factuality remain unknown.',
        'Unproposed response regions are unscored; empty/single/multiple pools confer no truth judgment.',
        'Requested native labels may allocate empty pools; candidate members/order never depend on R values.',
        'Other-context values remain; literal erasure is not proof of semantic absence of leakage.'])
    cases = []
    # Phase one: every S pool is complete before any response target value is read.
    for sample, row, owner in zip(samples, rows, ids, strict=True):
        source, field = _get(sample, 'source'), row.get('source')
        entities = _entities(field)
        source_reason = _field_reason(field, source)
        if source_reason is None:
            source_reason = next((_entity_reason(e, source) for e in entities if _entity_reason(e, source)), None)
        response_field = row.get('response')
        # Types are metadata only: no response entity text/coordinates are used.
        labels = {_label(e) for e in entities} | {_label(e) for e in _entities(response_field)}
        pools = {}
        for label in sorted(labels, key=lambda value: (value is None, value or '')):
            occurrences = [_provenance(e, owner, i) for i, e in enumerate(entities) if _label(e) == label]
            reason = source_reason or ('invalid_native_label' if label is None else None)
            candidates = sorted({e['text'] for e in occurrences}) if reason is None else []
            key = _key(dict(schema=CONTRACT_VERSION, owner_id=owner, native_label=label))
            pools[label] = key
            result['source_pools'][key] = dict(source_pool_id=key, owner_id=owner, native_label=label,
                source_sha256=_hash(source), compiler_status='unknown' if reason else 'compiled', compiler_reason=reason,
                candidates=candidates, occurrences=occurrences, raw_occurrence_count=len(occurrences),
                unique_candidate_count=None if reason else len(candidates),
                cardinality='unknown' if reason else ('empty' if not candidates else 'single' if len(candidates) == 1 else 'multiple'),
                semantic_status=UNKNOWN, qa_admitted=False, risk=None)
        cases.append((sample, row, owner, source_reason, pools))
    # Phase two: compile all R occurrences against the already frozen S inventories.
    for sample, row, owner, source_reason, pools in cases:
        response, field = _get(sample, 'response'), row.get('response')
        entities = _entities(field)
        field_reason = _field_reason(field, response)
        original = dict(id=owner, source_sha256=_hash(_get(sample, 'source')), response_sha256=_hash(response),
            source_inventory_status='unknown' if source_reason else 'validated',
            source_inventory_reason=source_reason, response_inventory_status='unknown' if field_reason else 'validated',
            response_inventory_reason=field_reason, proposal_ids=[], unproposed_response_regions=UNKNOWN,
            qa_admitted=False, risk=None)
        result['originals'].append(original)
        for index, target in enumerate(entities):
            label = _label(target)
            pool_id = pools.get(label)
            reason, masked, mappings, repeats = field_reason or source_reason, None, [], 0
            if reason is None:
                reason, masked, mappings, repeats = _mask(response, entities, target)
            proposal_id = _key(dict(schema=CONTRACT_VERSION, owner_id=owner, response_entity_index=index))
            readout = _provenance(target, owner, index)
            proposal = dict(proposal_id=proposal_id, owner_id=owner, response_entity_index=index,
                compiler_status='unknown' if reason else 'compiled', compiler_reason=reason,
                cloze_artifact=None if reason else dict(masked_response=masked, target_placeholder='[TARGET]', native_label=label),
                target_readout=readout, mask_spans=mappings, target_literal_occurrences=repeats,
                diagnostics=dict(placeholder_constant_contains_target_literal=any(
                    isinstance(readout.get('text'), str) and readout['text'] in m['replacement'] for m in mappings)),
                source_pool_id=pool_id, masked_characters=None if masked is None else len(masked),
                semantic_status=dict(_SEMANTIC), qa_admitted=False, risk=None)
            result['proposals'].append(proposal)
            original['proposal_ids'].append(proposal_id)
    # Last, readout only: membership never changes pools, artifacts or admission.
    for proposal in result['proposals']:
        pool = result['source_pools'].get(proposal['source_pool_id'])
        literal = proposal['target_readout'].get('text')
        proposal['target_readout']['original_value_exactly_in_source_pool'] = (
            literal in pool['candidates'] if pool and pool['compiler_status'] == 'compiled' and isinstance(literal, str) and literal else None)
    proposals = result['proposals']
    result['counts'] = dict(originals=len(result['originals']), response_occurrences=len(proposals),
        source_occurrences=sum(p['raw_occurrence_count'] for p in result['source_pools'].values()),
        source_pools=len(result['source_pools']), compiled=sum(p['compiler_status'] == 'compiled' for p in proposals),
        unknown=sum(p['compiler_status'] == 'unknown' for p in proposals),
        labels=dict(Counter(_label(p['target_readout']) for p in proposals if _label(p['target_readout']) is not None)),
        invalid_native_label_occurrences=sum(_label(p['target_readout']) is None for p in proposals),
        unknown_reasons=dict(Counter(p['compiler_reason'] for p in proposals if p['compiler_reason'])),
        potential_completion_pairs_if_scored=sum(len(result['source_pools'][p['source_pool_id']]['candidates'])
            for p in proposals if p['source_pool_id'] in result['source_pools']),
        qa_admitted=0, model_calls=0, tokenizer_calls=0, nlp_calls=0)
    return result

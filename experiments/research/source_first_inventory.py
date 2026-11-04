"""Pure source-only inventory/controls and literal proposal validation for H020.

No R interface, file I/O, tokenizer/model, risk, or semantic licence is provided.
The caller freezes source inputs and retains backend raw output before validation.
"""
from collections import Counter
import hashlib
import json
import math
import re


REQUESTS = [(s, 'native') for s in ('14304', '14315', '14325', '14375')] + [
    ('14375', 'passage_reorder_312'), ('14375', 'duration_literal_swap')]
NUMBER_WORDS = ('zero one two three four five six seven eight nine ten eleven twelve thirteen '
    'fourteen fifteen sixteen seventeen eighteen nineteen twenty hundred thousand million billion '
    'half quarter third once twice thrice').split()
UNITS = ('second seconds sec secs minute minutes min mins hour hours hr hrs day days week weeks '
    'month months year years degree degrees pound pounds lb lbs percent percentage dollar dollars '
    'cent cents mile miles meter meters metre metres kilometer kilometers kilometre kilometres '
    'inch inches foot feet kg g mg cm mm').split()
SCOPE_CUES = ('each every all both either neither none no any some most least more less fewer '
    'only always never exactly about another additional per between during before after afterwards '
    'then when until if unless not without').split()
PATTERNS = [('number', re.compile(r'\d+(?:[.,]\d+)*'))] + [
    (kind, re.compile(r'\b(?:'+'|'.join(map(re.escape, words))+r')\b', re.I))
    for kind, words in [('number_word', NUMBER_WORDS), ('unit', UNITS), ('scope_cue', SCOPE_CUES)]]
RECORD_KEYS = {'id', 'status', 'speech', 'anchors', 'cite', 'pred', 'args', 'ops', 'links', 'unknown'}
ROLES = {'subject', 'object', 'value', 'recipient', 'location', 'other'}
OPS = {'negation', 'quantifier', 'condition', 'time', 'unit', 'rate', 'comparison', 'modality',
       'quotation', 'attribution', 'sequence', 'alternative', 'method', 'other'}
LINKS = {'before', 'after', 'condition', 'reported_by', 'same_event', 'alternative', 'coreference', 'unknown'}
SPEECH = {'asserted', 'instruction', 'reported', 'hypothetical', 'unknown'}


def _require(value, reason):
    if not value:
        raise ValueError(reason)


def _sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _integer(value):
    return type(value) is int


def _matches(text, quote):
    result, start = [], 0
    while quote:
        position = text.find(quote, start)
        if position < 0:
            break
        result.append([position, position+len(quote)])
        start = position+1  # Overlapping exact occurrences are distinct.
    return result


def _split(prompt):
    markers = list(re.finditer(r'(?m)^passage ([123]):', prompt))
    endings = list(re.finditer(r'(?m)^In case the passages do not contain', prompt))
    _require(len(markers) == 3 and {int(m[1]) for m in markers} == {1, 2, 3}
             and len(endings) == 1 and endings[0].start() > markers[-1].end(),
             'Ambiguous source passage/footer boundaries')
    footer = endings[0].start()
    passages = []
    for index, marker in enumerate(markers):
        stop = markers[index+1].start() if index < 2 else footer
        body_stop = stop-2 if prompt[stop-2:stop] == '\n\n' else stop
        passages.append(dict(id=int(marker[1]), text=prompt[marker.end():body_stop],
            prompt_span=[marker.end(), body_stop], block_span=[marker.start(), stop],
            separator=prompt[body_stop:stop]))
    return prompt[:markers[0].start()], prompt[footer:], passages


def _origin(span, segments):
    result = []
    for segment in segments:
        a, b = segment['world_span']; left, right = max(a, span[0]), min(b, span[1])
        if left < right:
            exact = segment['kind'] != 'replacement'
            original = ([segment['original_span'][0]+left-a, segment['original_span'][0]+right-a]
                        if exact else segment['original_span'])
            result.append(dict(world_span=[left, right], original_span=original, exact=exact,
                               kind=segment['kind']))
    return result


def _case(source_id, world, prompt, native, segments, edits):
    prefix, suffix, passages = _split(prompt)
    _, _, original_passages = _split(native)
    originals = {p['id']: p for p in original_passages}
    anchors = []
    for passage in sorted(passages, key=lambda p: p['id']):
        passage['original_prompt_span'] = originals[passage['id']]['prompt_span']
        passage['original_block_span'] = originals[passage['id']]['block_span']
        passage['text_sha256'] = _sha(passage['text'])
        found = [(m.start(), m.end(), kind, m.group()) for kind, pattern in PATTERNS
                 for m in pattern.finditer(passage['text'])]
        for ordinal, (a, b, kind, literal) in enumerate(sorted(found)):
            physical = [passage['prompt_span'][0]+a, passage['prompt_span'][0]+b]
            anchors.append(dict(id=f"p{passage['id']}a{ordinal}", passage=passage['id'], kind=kind,
                literal=literal, span=[a, b], prompt_span=physical, origin_parts=_origin(physical, segments)))
    return dict(world_key=source_id+'--'+world, source_id=source_id, world=world,
        constructed_counterfactual=world != 'native',
        prompt=prompt, prompt_sha256=_sha(prompt), native_prompt_sha256=_sha(native),
        passages=passages, anchors=anchors,
        transform=dict(kind=world, segments=segments, edits=edits,
            utf8_byte_multiset_preserved=Counter(prompt.encode()) == Counter(native.encode()),
            semantic_equivalence_claim=False),
        model_data=dict(task_prefix=prefix, task_suffix=suffix,
            passages=[dict(id=p['id'], text=p['text']) for p in passages],
            anchors=[[a['id'], a['kind'], a['literal'], a['span']] for a in anchors]))


def prepare_inventory(source_records, protocol):
    """Return six fixed source-only cases; inputs contain only source_id/prompt.

    No paths are followed. Controls consume the supplied, separately frozen
    protocol's exact source hash/spans. M0 token counts and semantic review belong
    to the caller, not to this literal inventory.
    """
    _require(protocol.get('schema') == 'h020-source-first-record-proposal-v1'
             and [(r['source_id'], r['world']) for r in protocol['requests']] == REQUESTS
             and protocol.get('input_whitelist') == ['source_id', 'prompt'], 'Finite protocol differs')
    sources = {}
    for item in source_records:
        _require(type(item) is dict and set(item) == {'source_id', 'prompt'}, 'Source-only input keys required; R is forbidden')
        identity, prompt = item['source_id'], item['prompt']
        _require(type(identity) is str and type(prompt) is str and prompt and identity not in sources,
                 'Invalid or duplicate source input')
        _sha(prompt)
        _require([p['id'] for p in _split(prompt)[2]] == [1, 2, 3], 'Native passage identity/order differs')
        sources[identity] = prompt
    _require(set(sources) == {s for s, world in REQUESTS if world == 'native'}, 'Source roster differs')
    native = sources['14375']; spec = protocol['source14375']
    prefix, suffix, passages = _split(native)
    blocks = {p['id']: native[slice(*p['block_span'])] for p in passages}
    _require(_sha(native) == protocol['control_source_sha256'] and len(native) == spec['characters']
             and spec['header'] == [0, len(prefix)]
             and spec['passage_blocks'] == [p['block_span'] for p in passages]
             and spec['footer'] == [len(native)-len(suffix), len(native)]
             and spec['reorder_original_passage_ids'] == [3, 1, 2], 'Control source/span identity differs')
    cases = []
    for identity, world in REQUESTS:
        original = sources[identity]
        if world == 'native':
            text, edits = original, []
            segments = [dict(original_span=[0, len(text)], world_span=[0, len(text)], kind='unchanged')]
        elif world == 'passage_reorder_312':
            text = prefix+''.join(blocks[p] for p in (3, 1, 2))+suffix
            pieces = [([0, len(prefix)], 'unchanged')] + [(p['block_span'], 'moved')
                for pid in (3, 1, 2) for p in passages if p['id'] == pid] + [(spec['footer'], 'unchanged')]
            segments, position = [], 0
            for span, kind in pieces:
                size = span[1]-span[0]
                segments.append(dict(original_span=span, world_span=[position, position+size], kind=kind))
                position += size
            edits = []
        else:
            replacements = sorted(spec['swap_literals'], key=lambda e: e['span'][0])
            _require(len(replacements) == 2 and replacements[0]['span'][1] <= replacements[1]['span'][0]
                     and Counter(e['old'] for e in replacements) == Counter(e['new'] for e in replacements),
                     'Swap must preserve the complete literal multiset')
            text = original
            for edit in reversed(replacements):
                a, b = edit['span']
                _require(_integer(a) and _integer(b) and 0 <= a < b <= len(original)
                         and original[a:b] == edit['old'] and type(edit['new']) is str and edit['new'],
                         'Literal swap source differs')
                text = text[:a]+edit['new']+text[b:]
            segments, edits, old, new = [], [], 0, 0
            for edit in replacements:
                a, b = edit['span']
                if a > old:
                    segments.append(dict(original_span=[old, a], world_span=[new, new+a-old], kind='unchanged'))
                    new += a-old
                end = new+len(edit['new'])
                segments.append(dict(original_span=[a, b], world_span=[new, end], kind='replacement'))
                edits.append(dict(original_span=[a, b], world_span=[new, end], old=edit['old'], new=edit['new']))
                old, new = b, end
            segments.append(dict(original_span=[old, len(original)], world_span=[new, new+len(original)-old], kind='unchanged'))
        item = _case(identity, world, text, original, segments, edits)
        _require(item['transform']['utf8_byte_multiset_preserved'], 'Control changed the source UTF8 byte multiset')
        cases.append(item)
    return dict(schema='h020-source-first-inventory-v1', cases=cases,
        source_count=4, request_count=6, anchors_per_case=[len(c['anchors']) for c in cases],
        semantic_licensed=False, token_counts_measured=False, model_calls=0, tokenizer_calls=0, GT_reads=0)


def _strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate_json_key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite_json')
    def number(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError('nonfinite_json')
        return result
    def unicode_tree(value):
        if isinstance(value, str): value.encode('utf-8')
        elif isinstance(value, list):
            for item in value: unicode_tree(item)
        elif isinstance(value, dict):
            for key, item in value.items(): unicode_tree(key); unicode_tree(item)
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)
    unicode_tree(value)
    return value


def _citation(item, passages, segments):
    _require(type(item) is list and len(item) == 3, 'citation_shape')
    passage, quote, ordinal = item
    _require(_integer(passage) and passage in passages and type(quote) is str and quote
             and _integer(ordinal) and ordinal >= 0, 'citation_reference')
    source = passages[passage]
    matches = _matches(source['text'], quote)
    _require(ordinal < len(matches), 'citation_not_exact_or_occurrence_out_of_range')
    a, b = matches[ordinal]; physical = [source['prompt_span'][0]+a, source['prompt_span'][0]+b]
    return dict(passage=passage, quote=quote, occurrence=ordinal, body_span=[a, b],
        prompt_span=physical, quote_sha256=_sha(quote), origin_parts=_origin(physical, segments))


def validate_proposal(raw_text, inventory, finish_reason):
    """Validate literals/coverage; always leaves semantic_licensed=False.

    Invalid/truncated envelopes retain no salvaged records. A valid envelope
    retains each raw record and its issues, including every ambiguous boundary.
    Backend raw text/token IDs must already be sealed by the caller.
    """
    anchors = {a['id']: a for a in inventory['anchors']}
    passages = {p['id']: p for p in inventory['passages']}
    segments = inventory['transform']['segments']
    def refused(reason):
        return dict(request_status='unknown', envelope_error=reason, records=[], coverage_errors=[],
            anchor_accounting=[dict(id=key, category='request_unknown', reason=reason, model_explicit=False) for key in anchors],
            raw_model_mapped=[], valid_mapped=[], explicit_unknown=[], nonrelational=[],
            mechanical_gate_passed=False, semantic_licensed=False, raw_retention_required=True)
    if finish_reason != 'eos':
        return refused('output_token_limit' if finish_reason == 'max_new_tokens' else 'unsupported_finish_reason')
    try:
        _require(type(raw_text) is str, 'output_is_not_text')
        value = _strict_json(raw_text)
        _require(type(value) is dict and set(value) == {'records', 'coverage'}
                 and type(value['records']) is list and type(value['coverage']) is list, 'envelope_schema')
        coverage = value['coverage']
        _require(len(coverage) == 3 and all(type(c) is dict and set(c) == {'passage', 'unknown', 'nonrelational', 'scope_unknown'}
                 and _integer(c['passage']) and c['passage'] in passages
                 and all(type(c[k]) is list for k in ('unknown', 'nonrelational', 'scope_unknown')) for c in coverage)
                 and {c['passage'] for c in coverage} == set(passages), 'coverage_envelope_schema')
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        return refused(str(error))
    ids = Counter(r.get('id') for r in value['records'] if type(r) is dict and type(r.get('id')) is str)
    records, raw_mapped, valid_mapped = [], {}, {}
    def issue(result, code, field):
        result['issues'].append(dict(code=code, field=field))
    for index, raw in enumerate(value['records']):
        result = dict(index=index, raw=raw, id=raw.get('id') if type(raw) is dict else None,
            valid=False, citations=[], literal_boundaries=[], unresolved=[], issues=[], semantic_licensed=False,
            mechanically_complete_candidate=False)
        records.append(result)
        if type(raw) is dict and type(raw.get('anchors')) is list:
            for key in raw['anchors']:
                if type(key) is str and key in anchors: raw_mapped.setdefault(key, []).append(index)
        if not (type(raw) is dict and set(raw) == RECORD_KEYS
                and type(raw['id']) is str and raw['id'].strip()
                and type(raw['status']) is str and raw['status'] in ('complete', 'partial', 'unknown')
                and type(raw['speech']) is str and raw['speech'] in SPEECH
                and all(type(raw[k]) is list for k in ('anchors', 'cite', 'args', 'ops', 'links', 'unknown'))):
            issue(result, 'record_schema', 'record'); continue
        if ids[raw['id']] != 1: issue(result, 'duplicate_record_id', 'id')
        if not all(type(x) is str and x.strip() for x in raw['unknown']):
            issue(result, 'unknown_obligation_schema', 'unknown')
        for item in raw['cite']:
            try: result['citations'].append(dict(valid=True, **_citation(item, passages, segments)))
            except (ValueError, TypeError) as error:
                result['citations'].append(dict(valid=False, raw=item)); issue(result, str(error), 'cite')
        def cite_at(reference, field):
            if not (_integer(reference) and 0 <= reference < len(result['citations'])
                    and result['citations'][reference]['valid']):
                issue(result, 'invalid_cite_index', field); return None
            return result['citations'][reference]
        def literal(text, reference, field):
            cite = cite_at(reference, field)
            if text is None:
                if raw['status'] != 'complete' and raw['unknown']:
                    result['unresolved'].append(dict(field=field, reason='null_text')); return
                issue(result, 'null_without_explicit_unknown', field); return
            if type(text) is not str or not text:
                issue(result, 'literal_text_schema', field); return
            if cite is None: return
            matches = _matches(cite['quote'], text)
            locations = [[cite['body_span'][0]+a, cite['body_span'][0]+b] for a, b in matches]
            result['literal_boundaries'].append(dict(field=field, text=text, cite_index=reference,
                passage=cite['passage'], body_matches=locations, prompt_matches=[[passages[cite['passage']]['prompt_span'][0]+a,
                    passages[cite['passage']]['prompt_span'][0]+b] for a, b in locations]))
            if not matches: issue(result, 'literal_not_in_cite', field)
            elif len(matches) > 1: result['unresolved'].append(dict(field=field, reason='boundary_ambiguity'))
        if raw['pred'] is None:
            result['unresolved'].append(dict(field='pred', reason='null_predicate'))
        elif type(raw['pred']) is list and len(raw['pred']) == 2: literal(*raw['pred'], 'pred')
        else: issue(result, 'predicate_schema', 'pred')
        for field, items, size, enums in [('args', raw['args'], 3, ROLES), ('ops', raw['ops'], 4, OPS)]:
            for j, item in enumerate(items):
                if not (type(item) is list and len(item) == size and type(item[0]) is str and item[0] in enums):
                    issue(result, field+'_schema', f'{field}.{j}'); continue
                literal(item[1], item[2], f'{field}.{j}')
                if field == 'ops':
                    if item[3] not in ('record', 'unknown'): issue(result, 'operator_scope_schema', f'ops.{j}')
                    elif item[3] == 'unknown': result['unresolved'].append(dict(field=f'ops.{j}', reason='scope_unknown'))
        for j, item in enumerate(raw['links']):
            if not (type(item) is list and len(item) == 3 and type(item[0]) is str and item[0] in LINKS
                    and type(item[1]) is str and ids[item[1]] == 1):
                issue(result, 'link_target_missing_or_ambiguous', f'links.{j}'); continue
            cite_at(item[2], f'links.{j}')
            if item[0] == 'unknown': result['unresolved'].append(dict(field=f'links.{j}', reason='unknown_link'))
        for key in raw['anchors']:
            if type(key) is not str or key not in anchors:
                issue(result, 'unknown_anchor_id', 'anchors'); continue
            anchor = anchors[key]
            if not any(c.get('valid') and c['passage'] == anchor['passage']
                       and c['body_span'][0] <= anchor['span'][0] < anchor['span'][1] <= c['body_span'][1]
                       for c in result['citations']): issue(result, 'anchor_not_exactly_cited', key)
        if raw['speech'] == 'unknown': result['unresolved'].append(dict(field='speech', reason='unknown_speech'))
        if result['unresolved'] and not raw['unknown']: issue(result, 'unacknowledged_unresolved', 'unknown')
        if raw['status'] == 'complete' and (raw['pred'] is None or raw['unknown'] or result['unresolved'] or not any(
                type(a) is list and len(a) == 3 and type(a[1]) is str and a[1] for a in raw['args'])):
            issue(result, 'complete_has_missing_or_unresolved_obligation', 'status')
        result['valid'] = not result['issues']
        result['mechanically_complete_candidate'] = result['valid'] and raw['status'] == 'complete'
        if result['valid']:
            for key in set(raw['anchors']): valid_mapped.setdefault(key, []).append(index)
    coverage_errors, statuses, scope_unknown = [], {'unknown': {}, 'nonrelational': {}}, []
    for item in coverage:
        for kind in statuses:
            for entry in item[kind]:
                if not (type(entry) is list and len(entry) == 2 and type(entry[0]) is str and entry[0].strip()
                        and type(entry[1]) is list):
                    coverage_errors.append(dict(code='coverage_entry_schema', passage=item['passage'], category=kind)); continue
                for key in entry[1]:
                    if type(key) is not str or key not in anchors or anchors[key]['passage'] != item['passage']:
                        coverage_errors.append(dict(code='coverage_anchor_identity', category=kind)); continue
                    statuses[kind].setdefault(key, []).append(entry[0])
        for entry in item['scope_unknown']:
            try:
                _require(type(entry) is list and len(entry) == 3 and type(entry[2]) is str and entry[2].strip(), 'scope_unknown_schema')
                scope_unknown.append(dict(reason=entry[2], **_citation([item['passage'], entry[0], entry[1]], passages, segments)))
            except (ValueError, TypeError) as error: coverage_errors.append(dict(code=str(error), category='scope_unknown'))
    accounting = []
    for key in anchors:
        unknown, nonrel = statuses['unknown'].get(key, []), statuses['nonrelational'].get(key, [])
        categories = int(key in raw_mapped)+bool(unknown)+bool(nonrel)
        category = ('conflict' if categories > 1 else 'mapped' if key in valid_mapped else
                    'invalid_mapped' if key in raw_mapped else 'explicit_unknown' if unknown else
                    'nonrelational' if nonrel else 'unaccounted')
        if category in ('conflict', 'invalid_mapped', 'unaccounted') or len(unknown) > 1 or len(nonrel) > 1:
            coverage_errors.append(dict(code='anchor_accounting_'+category, anchor=key))
        accounting.append(dict(id=key, category=category, raw_record_indices=raw_mapped.get(key, []),
            valid_record_indices=valid_mapped.get(key, []), unknown_reasons=unknown, nonrelational_reasons=nonrel))
    repeated_cites = {}
    for record in records:
        for index, cite in enumerate(record['citations']):
            if cite.get('valid'):
                key = (cite['passage'], *cite['body_span'])
                repeated_cites.setdefault(key, []).append([record['index'], index])
    return dict(request_status='validated_proposals_only', envelope_error=None, records=records,
        raw_coverage=value['coverage'],
        coverage_errors=coverage_errors, anchor_accounting=accounting,
        raw_model_mapped=list(raw_mapped), valid_mapped=list(valid_mapped),
        explicit_unknown=list(statuses['unknown']), nonrelational=list(statuses['nonrelational']),
        scope_unknown=scope_unknown,
        structural_citation_reuse=[dict(passage=k[0], body_span=list(k[1:]), uses=v)
                                  for k, v in repeated_cites.items() if len(v) > 1],
        mechanical_gate_passed=not coverage_errors and all(r['valid'] for r in records),
        mechanically_complete_record_ids=[r['id'] for r in records if r['mechanically_complete_candidate']],
        any_raw_complete_invalid=any(type(r['raw']) is dict and r['raw'].get('status') == 'complete'
                                     and not r['valid'] for r in records),
        semantic_licensed=False, independent_source_review_required=True, raw_retention_required=True)

"""The frozen five-world hours-scope diagnostic; never a risk scorer.

Only the independently licensed seven weekday fields are interpreted.  Source
payload edits preserve every other character, and the complete original answer
is encoded separately and retained in every world.  Preparation performs no
model inference.  Its failure is final for this protocol, not a rescue prompt.
"""

from collections import Counter
import math

from myproject.inputs import encode_response, encode_source

from ..data import text_hash
from ..relation_alignment import _exact_offsets
from .relation_inventory import align_source_spans, source_inventory


CASE, SOURCE = '5670', '13601'
SOURCE_SHA256 = 'f71690e1016bcb5c29096c1b98c232cd253d2ddab7360400791ff76af9ef8b60'
RESPONSE_SHA256 = '9f547386286ee2e59e130d4fb227b8e54120c6212cb3396a00292fc05fba61d8'
SOURCE_CHARACTERS = 2750
NATIVE_SHAPE = (894, 185, 1079)
NATIVE_SOURCE_TOKENS = 798
ORIGINAL_MINUTE_ID = 410
MINUTE_TARGET, HOUR_TARGET, END_HOUR_TARGET = 55, 53, 57
DAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
A, B, C = '17:30-23:0', '17:0-21:0', '9:0-14:0'
WORLD_ORDER = ('native', 'S_minus', 'S_plus', 'Tuesday_exception', 'Saturday_exception')
CONTROLLED_ORDER = WORLD_ORDER[1:]
UNSUPPORTED_ORDER = ('S_minus', 'Tuesday_exception', 'Saturday_exception')
R_PHASES = ((50, 'range_end'), (52, 'pre_time'), (54, 'pre_minute'),
            (55, 'post_minute'), (56, 'pre_end_hour'), (57, 'post_end_hour'))


class EligibilityError(ValueError):
    """A frozen input, literal, token axis or required carrier is ineligible."""


def _span(source, coordinate, literal=None):
    if (not isinstance(coordinate, (tuple, list)) or len(coordinate) != 2
            or any(type(x) is not int for x in coordinate)
            or not 0 <= coordinate[0] < coordinate[1] <= len(source)):
        raise EligibilityError('A licensed source character span is unknown or invalid')
    start, stop = coordinate
    if literal is not None and source[start:stop] != literal:
        raise EligibilityError('A licensed source literal changed')
    return [start, stop]


def _hours(source):
    inventory = source_inventory(source, SOURCE)
    fields = {f['path'][1]: f for f in inventory['fields']
              if len(f['path']) == 2 and f['path'][0] == 'hours'}
    if (inventory['status'] != 'structural_fields_only' or set(fields) != set(DAYS)
            or any(f['owner_path'] != ['hours'] or f['scalar_type'] != 'string'
                   for f in fields.values())):
        raise EligibilityError('Source is outside the closed seven-weekday hours schema')
    return inventory, fields


def _licensed_native(source, license_):
    if (license_.get('case_id') != CASE or license_.get('source_id') != SOURCE
            or license_.get('source_sha256') != text_hash(source)
            or license_.get('source_characters') != len(source)):
        raise EligibilityError('The independent source license does not bind this source')
    inventory, fields = _hours(source)
    records = license_.get('licensed_facts', [])
    if len(records) != 7:
        raise EligibilityError('All seven source fact licenses are required')
    licensed = {}
    expected = {d: A if d == 'Monday' else C if d == 'Sunday' else B for d in DAYS}
    for record in records:
        path = record.get('M0', {}).get('path')
        day = record.get('time_scope', {}).get('weekday')
        if day not in DAYS or path != ['hours', day] or day in licensed:
            raise EligibilityError('Source fact weekday/path identity is ambiguous')
        field = fields[day]
        if record.get('value', {}).get('raw') != expected[day] or field['value'] != expected[day]:
            raise EligibilityError('The frozen native hours values changed')
        spans = record.get('source_spans', {})
        value = spans.get('interval_value_literal', {})
        payload = spans.get('interval_payload', {})
        key = spans.get('weekday_key_literal', {})
        for stored, coordinate, literal in ((value, field['span'], field['raw_text']),
                                            (key, field['key_span'], field['raw_key'])):
            if [stored.get('start'), stored.get('end')] != coordinate or stored.get('text') != literal:
                raise EligibilityError('M0 source field spans differ from the manual license')
            _span(source, coordinate, literal)
        coordinate = [payload.get('start'), payload.get('end')]
        if (coordinate != [field['span'][0] + 1, field['span'][1] - 1]
                or field['raw_text'] != repr(expected[day]) or payload.get('text') != expected[day]):
            raise EligibilityError('Only the complete original unescaped interval payload is editable')
        _span(source, coordinate, expected[day])
        licensed[day] = dict(record, payload_span=coordinate)
    if set(licensed) != set(DAYS):
        raise EligibilityError('A weekday source license is missing')
    return inventory, fields, licensed


def _edit_payloads(source, licensed, assignments):
    edits = [dict(day=day, native_span=licensed[day]['payload_span'],
                  before=licensed[day]['value']['raw'], after=assignments[day])
             for day in DAYS if assignments[day] != licensed[day]['value']['raw']]
    edited = source
    for edit in sorted(edits, key=lambda x: x['native_span'][0], reverse=True):
        start, stop = edit['native_span']
        edited = edited[:start] + edit['after'] + edited[stop:]
    # Independently walk the retained chunks, documenting their exact identity.
    old, new, unchanged = 0, 0, []
    for edit in sorted(edits, key=lambda x: x['native_span'][0]):
        start, stop = edit['native_span']
        chunk = source[old:start]
        if edited[new:new + len(chunk)] != chunk:
            raise EligibilityError('A source character outside the licensed value edits changed')
        unchanged.append(dict(native_span=[old, start], world_span=[new, new + len(chunk)],
                              sha256=text_hash(chunk), utf8_bytes=len(chunk.encode('utf-8'))))
        new += len(chunk)
        edit['world_span'] = [new, new + len(edit['after'])]
        new += len(edit['after'])
        old = stop
    chunk = source[old:]
    if edited[new:] != chunk:
        raise EligibilityError('The source suffix outside licensed edits changed')
    unchanged.append(dict(native_span=[old, len(source)], world_span=[new, len(edited)],
                          sha256=text_hash(chunk), utf8_bytes=len(chunk.encode('utf-8'))))
    return edited, edits, unchanged


def _complete_token(offsets, start, stop):
    overlap = [i for i, (left, right) in enumerate(offsets)
               if left < stop and right > start and right > left]
    if len(overlap) != 1 or offsets[overlap[0]] != [start, stop]:
        raise EligibilityError('A required numeral/delimiter must be one complete token')
    return overlap[0]


def _world(sample, tokenizer, source, answer, maximum):
    prompt = sample.prompt.replace(sample.source, source, 1)
    pack = encode_source(tokenizer, prompt, source)
    chat = tokenizer.apply_chat_template([dict(role='user', content=prompt)],
                                        tokenize=False, add_generation_prompt=True)
    rendered = prompt if prompt in chat else prompt.strip()
    if not rendered or chat.count(rendered) != 1 or rendered.count(source) != 1:
        raise EligibilityError('Rendered source/prompt identity is ambiguous')
    source_start = chat.index(rendered) + rendered.index(source)
    encoded = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
    if encoded['input_ids'] != pack['prompt_with_source']:
        raise EligibilityError('Supplemental offsets differ from the mainline prompt IDs')
    prompt_offsets = _exact_offsets(encoded['offset_mapping'], len(chat))
    answer_offsets = _exact_offsets(answer['offsets'], len(sample.response))
    offsets = prompt_offsets + _exact_offsets(answer['offsets'], len(sample.response), len(chat))
    ids = pack['prompt_with_source'] + answer['answer_ids']
    mask = pack['source_mask'] + [False] * len(answer['answer_ids'])
    if len(ids) > maximum:
        raise EligibilityError('A complete frozen world exceeds the admitted input length')
    inventory, fields = _hours(source)
    anchors = []
    for day in DAYS:
        field = fields[day]
        value = field['value']
        if value not in {A, B, C} or field['raw_text'] != repr(value):
            raise EligibilityError('An hours value has an unlicensed spelling')
        start, stop = field['span'][0] + 1, field['span'][1] - 1
        minute_start = start + value.index(':') + 1
        minute_stop = start + value.index('-')
        for suffix, coordinate in (('literal', field['span']), ('payload', [start, stop]),
                                   ('start_minute', [minute_start, minute_stop])):
            anchors.append(dict(occurrence_id=day + ':' + suffix, span=coordinate,
                                raw_text=source[slice(*coordinate)]))
    restricted = dict(inventory, fields=anchors, lexical_occurrences=[])
    alignment = align_source_spans(restricted, source, chat + sample.response, source_start,
                                  offsets, ids, ids, mask)
    mapped = {r['occurrence_id']: r for r in alignment['occurrences']}
    closures, receivers = {}, []
    for day in DAYS:
        field = fields[day]
        end = source_start + field['span'][1]
        completed = [i for i, (left, right) in enumerate(prompt_offsets) if left < end <= right]
        if not completed:
            raise EligibilityError('A required source fact closure has unknown token coverage')
        position = min(completed)
        if not pack['source_mask'][position]:
            raise EligibilityError('A source closure is not on the original source mask')
        span = prompt_offsets[position]
        closure = dict(day=day, path=['hours', day], value=field['value'],
                       query_position=position, token_id=ids[position], model_offset=span,
                       token_text=chat[slice(*span)], literal_end_model_character=end,
                       suffix_after_value_literal=chat[end:span[1]],
                       status='boundary_overlap' if span[1] > end else 'value_literal_complete',
                       semantic_encoding_or_use='unknown_not_inferred')
        closures[day] = closure
        receivers.append(dict(kind='source_fact_closure', name=day, query_position=position,
                              source_path=['hours', day], token_id=ids[position]))
    p = len(pack['prompt_with_source'])
    for index, phase in R_PHASES:
        if index < 0 or index + 1 >= len(answer['answer_ids']):
            raise EligibilityError('A frozen response receiver/next-target axis is unavailable')
        receivers.append(dict(kind='response', name=phase, receiver_answer_index=index,
            query_position=p + index, token_id=answer['answer_ids'][index],
            receiver_character_span=answer_offsets[index], target_answer_index=index + 1,
            target_token_id=answer['answer_ids'][index + 1], target_character_span=answer_offsets[index + 1]))
    receivers.sort(key=lambda r: r['query_position'])
    if len(receivers) != 13 or len({r['query_position'] for r in receivers}) != 13:
        raise EligibilityError('The frozen six response/seven source receivers are not distinct')
    for i, receiver in enumerate(receivers):
        receiver['capture_row'] = i
        if receiver['kind'] == 'source_fact_closure':
            closures[receiver['name']]['capture_row'] = i
    return dict(source=source, source_sha256=text_hash(source), prompt=prompt,
        prompt_sha256=text_hash(prompt), source_start=source_start, prompt_character_count=len(chat),
        prompt_tokens=p, answer_tokens=len(answer['answer_ids']), full_tokens=len(ids),
        token_ids=ids, answer_ids=list(answer['answer_ids']), answer_offsets=answer_offsets,
        source_mask=mask, query_positions=list(range(p - 1, len(ids))), offsets=offsets,
        source_anchors=mapped, source_closures=closures, capture_receivers=receivers,
        capture_positions=[r['query_position'] for r in receivers], alignment=alignment,
        primary_query_position=p + MINUTE_TARGET - 1,
        hours_assignment={d: fields[d]['value'] for d in DAYS})


def _position_audit(reference, other):
    differences = [i for i, (a, b) in enumerate(zip(reference['token_ids'], other['token_ids'], strict=True))
                   if a != b]
    masks = []
    for world in (reference, other):
        payloads = [world['source_anchors'][day + ':payload']['model_span'] for day in DAYS]
        masks.append([any(left < b and right > a for a, b in payloads)
                      for left, right in world['offsets']])
    outside = [i for i in differences if not masks[0][i] and not masks[1][i]]
    closure_changes = [d for d in DAYS if reference['source_closures'][d]['query_position']
                       != other['source_closures'][d]['query_position']]
    return dict(token_id_difference_positions=differences,
                outside_all_licensed_hours_payload_difference_positions=outside,
                source_closure_position_changes=closure_changes,
                source_masks_equal_by_position=reference['source_mask'] == other['source_mask'],
                internal_position_identity_claimed=False, activation_interpatch_performed=False)


def prepare_worlds(sample, tokenizer, stored_axes, source_license, *, maximum=1152):
    """Prepare all five eligible worlds before root starts any model call.

    ``stored_axes`` has M2's token_ids/answer_ids/source_mask/query_positions and
    the primitive response's answer_offsets.  The caller binds file/asset SHAs;
    this pure function checks their selected literal and numerical content.
    """
    if (sample.id != CASE or sample.source_id != SOURCE or sample.task != 'Data2txt'
            or len(sample.source) != SOURCE_CHARACTERS or text_hash(sample.source) != SOURCE_SHA256
            or text_hash(sample.response) != RESPONSE_SHA256 or sample.prompt.count(sample.source) != 1
            or not getattr(tokenizer, 'is_fast', False) or type(maximum) is not int
            or not 1 <= maximum <= 1152):
        raise EligibilityError('The fixed original source/answer/case contract changed')
    for key in ('token_ids', 'answer_ids', 'query_positions'):
        if (not isinstance(stored_axes.get(key), list)
                or any(type(value) is not int or value < 0 for value in stored_axes[key])):
            raise EligibilityError('Stored token/position axes must contain exact nonnegative integers')
    if (not isinstance(stored_axes.get('source_mask'), list)
            or any(type(value) is not bool for value in stored_axes['source_mask'])):
        raise EligibilityError('Stored source mask must contain exact Boolean values')
    _, _, licensed = _licensed_native(sample.source, source_license)
    answer = encode_response(tokenizer, sample.response)
    expected_offsets = _exact_offsets(stored_axes['answer_offsets'], len(sample.response))
    if (len(answer['answer_ids']) != NATIVE_SHAPE[1] or answer['answer_ids'] != stored_axes['answer_ids']
            or _exact_offsets(answer['offsets'], len(sample.response)) != expected_offsets):
        raise EligibilityError('The complete original response IDs/offsets changed')
    assignments = {'native': {d: licensed[d]['value']['raw'] for d in DAYS}}
    for name, exception in zip(CONTROLLED_ORDER, ('Monday', 'Sunday', 'Tuesday', 'Saturday'), strict=True):
        assignments[name] = {d: A if d == exception else B for d in DAYS}
    worlds = {}
    for name in WORLD_ORDER:
        source, edits, unchanged = _edit_payloads(sample.source, licensed, assignments[name])
        world = _world(sample, tokenizer, source, answer, maximum)
        if world['hours_assignment'] != assignments[name]:
            raise EligibilityError('A source assignment does not match the licensed edited fields')
        world.update(id=name, edits=edits, unchanged_source_regions=unchanged,
            complete_interval_counts=dict(Counter(assignments[name].values())),
            direct_Mon_to_Sat_B_baseline=all(assignments[name][d] == B for d in DAYS[:6]),
            later_Sunday_C_baseline=assignments[name]['Sunday'] == C,
            whole_response_support='not_assessed_other_claims_unknown')
        worlds[name] = world
    native = worlds['native']
    if ((native['prompt_tokens'], native['answer_tokens'], native['full_tokens']) != NATIVE_SHAPE
            or sum(native['source_mask']) != NATIVE_SOURCE_TOKENS
            or any(native[k] != stored_axes[k] for k in
                   ('token_ids', 'answer_ids', 'source_mask', 'query_positions'))):
        raise EligibilityError('Native mainline prompt/source-mask/original full axis changed')
    minute_start, minute_stop = expected_offsets[MINUTE_TARGET]
    if (sample.response[minute_start:minute_stop] != '00'
            or _complete_token(expected_offsets, minute_start, minute_stop) != MINUTE_TARGET
            or answer['answer_ids'][MINUTE_TARGET] != ORIGINAL_MINUTE_ID):
        raise EligibilityError('Original target00 must be its complete frozen token410')
    # Borrow30 only from the actual native17:30 source occurrence.  No R variant
    # is formed or tokenized.  Verify the complete17 and colon local contexts.
    payload = native['source_anchors']['Monday:payload']['model_span'][0]
    source_17 = _complete_token(native['offsets'], payload, payload + 2)
    source_colon = _complete_token(native['offsets'], payload + 2, payload + 3)
    source_30 = _complete_token(native['offsets'], payload + 3, payload + 5)
    if (native['token_ids'][source_17] != answer['answer_ids'][HOUR_TARGET]
            or native['token_ids'][source_colon] != answer['answer_ids'][MINUTE_TARGET - 1]
            or source_colon != source_17 + 1 or source_30 != source_colon + 1
            or sample.response[slice(*expected_offsets[HOUR_TARGET])] != '17'
            or sample.response[slice(*expected_offsets[MINUTE_TARGET - 1])] != ':'):
        raise EligibilityError('Source17: and the original output minute context differ')
    candidate_ids = {'00': ORIGINAL_MINUTE_ID, '30': native['token_ids'][source_30]}
    if len(set(candidate_ids.values())) != 2 or tokenizer.decode([candidate_ids['30']]) != '30':
        raise EligibilityError('Candidate30 must be one distinct complete native source token')
    controlled = [worlds[n] for n in CONTROLLED_ORDER]
    if (len({len(w['source']) for w in controlled}) != 1
            or len({w['prompt_tokens'] for w in controlled}) != 1
            or len({w['full_tokens'] for w in controlled}) != 1
            or len({sum(w['source_mask']) for w in controlled}) != 1
            or any(w['complete_interval_counts'] != {A: 1, B: 6} for w in controlled)):
        raise EligibilityError('The four controlled worlds do not have matched counts/total axes')
    audit = {name: _position_audit(controlled[0], worlds[name]) for name in CONTROLLED_ORDER}
    return dict(schema='h017-five-world-hours-scope-v1', worlds=worlds, world_order=list(WORLD_ORDER),
        response_sha256=RESPONSE_SHA256, answer_ids=list(answer['answer_ids']), answer_offsets=expected_offsets,
        candidate_ids=candidate_ids, target_answer_index=MINUTE_TARGET,
        candidate30_scope='native source17:30 complete token with matching local17: IDs; no modified R encoded',
        candidate30_is_complete_range_correct_answer=False,
        R_receiver_indices=[i for i, _ in R_PHASES], source_closure_count=7,
        budget={'model_calls': 5, 'head_calls': 5},
        direct_rule_baseline={name: worlds[name]['direct_Mon_to_Sat_B_baseline'] for name in WORLD_ORDER},
        eligibility_audit=dict(controlled_common_prompt_tokens=controlled[0]['prompt_tokens'],
            controlled_common_full_tokens=controlled[0]['full_tokens'],
            controlled_common_source_mask_count=sum(controlled[0]['source_mask']),
            controlled_source_character_count=len(controlled[0]['source']),
            native_matches_controlled_total_lengths=native['full_tokens'] == controlled[0]['full_tokens'],
            controlled_pairwise_vs_S_minus=audit, internal_positions_audited_not_interpatched=True),
        primary_scope='Finite support-consistent comparison versus strict global-count-only null; position heuristics remain confounded')


def scientific_summary(observations):
    """The preregistered finite logit comparison, not detection evaluation.

    Each fixed world supplies candidates['00'/'30']['logit'] from its full head.
    Optional original_token_endpoints is the complete original185-row list.
    """
    if set(observations) != set(WORLD_ORDER):
        raise ValueError('All five predefined world observations are required')
    endpoints, margins = {}, {}
    for name in WORLD_ORDER:
        candidates = observations[name]['candidates']
        pair = {value: float(candidates[value]['logit']) for value in ('00', '30')}
        if not all(math.isfinite(v) for v in pair.values()):
            raise ValueError('Primary full-vocabulary candidate logits must be finite')
        endpoints[name], margins[name] = pair, pair['00'] - pair['30']
    contrasts = {}
    for name in UNSUPPORTED_ORDER:
        allowance = .008 + .0001 * sum(abs(v) for n in ('S_plus', name) for v in endpoints[n].values())
        difference = margins['S_plus'] - margins[name]
        contrasts[name] = dict(S_plus_minus_control=difference, engineering_allowance=allowance,
                               passed=difference > allowance)
    unsupported = [margins[n] for n in UNSUPPORTED_ORDER]
    secondary = {}
    supplied = ['original_token_endpoints' in observations[n] for n in WORLD_ORDER]
    if any(supplied) and not all(supplied):
        raise ValueError('Secondary endpoints must include every world, not a selected subset')
    if all(supplied):
        for name in WORLD_ORDER:
            rows = observations[name]['original_token_endpoints']
            if (len(rows) != NATIVE_SHAPE[1]
                    or [row.get('answer_index') for row in rows] != list(range(NATIVE_SHAPE[1]))):
                raise ValueError('Secondary endpoints must preserve all original targets')
            values = [float(rows[i]['logp']) for i in range(HOUR_TARGET, HOUR_TARGET + 7)]
            if not all(math.isfinite(v) for v in values):
                raise ValueError('Secondary original-token log probabilities must be finite')
            secondary[name] = dict(time_target_indices=list(range(HOUR_TARGET, HOUR_TARGET + 7)),
                                   original_logp=values, complete_interval_logp_sum=sum(values))
    return dict(primary_all_three_pass=all(r['passed'] for r in contrasts.values()),
        primary_margin='z00-z30 at original preminute receiverR54', margins=margins,
        candidate_logits=endpoints, primary_contrasts=contrasts,
        original_anchor_excluded_from_matched_gate=True,
        unsupported_world_variation=dict(minimum=min(unsupported), maximum=max(unsupported),
                                         spread=max(unsupported) - min(unsupported)),
        secondary=secondary, secondary_replaces_primary=False,
        interpretation='Pass rejects only strict global-count-only equality in this finite assay; source position/order heuristics are unresolved',
        candidate30_interpretation='A borrowed source token, not a complete Mon-Sat-correct answer or entailment/risk probability',
        new_detector_scores=0, auroc=None, original_generator_causality=False)

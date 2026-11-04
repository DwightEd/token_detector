"""Two frozen bool-owner source edits; no response tokenization/model/labels.

Caller binds sample/fields/original_axes to the sealed 5684 prerequisite. This
frontend licenses exact serialized-field edits, not whole-response truth. It
does not render/encode S0; two new prompts each use core encode_source once.
"""

from collections import Counter
import copy
import hashlib
from numbers import Integral

import numpy as np

from myproject.inputs import encode_source
from .relation_inventory import align_source_spans, source_inventory


VALET = ('attributes', 'BusinessParking', 'valet')
GARAGE = ('attributes', 'BusinessParking', 'garage')
OUTDOOR = ('attributes', 'OutdoorSeating')
STREET = ('attributes', 'BusinessParking', 'street')
WORLD_EDITS = {'target': (VALET, OUTDOOR), 'control': (GARAGE, OUTDOOR)}
P, T, N, SOURCE_TOKENS = 803, 233, 1036, 707


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _sample(sample, name):
    return sample[name] if isinstance(sample, dict) else getattr(sample, name)


def _span(value, limit):
    _require(isinstance(value, (list, tuple)) and len(value) == 2
             and all(isinstance(v, Integral) and not isinstance(v, bool) for v in value),
             'Exact non-Boolean integer span required')
    left, right = map(int, value)
    _require(0 <= left < right <= limit, 'Source span outside original text')
    return left, right


def _offsets(values, characters):
    result = []
    for value in values:
        _require(len(value) == 2 and all(isinstance(v, Integral) and not isinstance(v, bool)
                                        for v in value), 'Exact tokenizer offsets required')
        left, right = map(int, value)
        _require(0 <= left <= right <= characters, 'Tokenizer offset outside rendered chat')
        result.append([left, right])
    return result


class _RecordingTokenizer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.renders, self.calls = [], []

    def apply_chat_template(self, messages, **kwargs):
        _require(not self.renders and kwargs == dict(tokenize=False, add_generation_prompt=True),
                 'Exactly one core chat render per new source')
        rendered = self.tokenizer.apply_chat_template(messages, **kwargs)
        _require(isinstance(rendered, str), 'Chat render must be text')
        self.renders.append(rendered)
        return rendered

    def __call__(self, text, **kwargs):
        _require(not self.calls and kwargs == dict(add_special_tokens=False, return_offsets_mapping=True),
                 'Exactly one prompt encoding per new source')
        encoded = self.tokenizer(text, **kwargs)
        self.calls.append((text, encoded))
        return encoded

    def decode(self, *unused, **kwargs):
        raise RuntimeError('No response decoding is permitted')


def _axes(original, response):
    required = {
        'token_ids': ((N,), np.int64), 'answer_ids': ((T,), np.int64),
        'source_mask': ((N,), np.bool_), 'prechoice_positions': ((T,), np.int64),
        'all_query_positions': ((T + 1,), np.int64),
        'full_offsets': ((N, 2), np.int64), 'answer_offsets': ((T, 2), np.int64),
    }
    for key, (shape, dtype) in required.items():
        value = original.get(key)
        _require(isinstance(value, np.ndarray) and value.shape == shape and value.dtype == dtype,
                 'Original sealed axis dtype/shape changed: ' + key)
    _require(np.array_equal(original['token_ids'][P:], original['answer_ids']), 'Original R ID suffix changed')
    _require(np.array_equal(original['prechoice_positions'], np.arange(P-1, N-1))
             and np.array_equal(original['all_query_positions'], np.arange(P-1, N)), 'Original query shift changed')
    _require(not original['source_mask'][P:].any()
             and int(original['source_mask'].sum()) == SOURCE_TOKENS, 'Original source mask changed')
    _require(bool((original['token_ids'] >= 0).all()), 'Original token IDs must be nonnegative')
    offsets = original['answer_offsets']
    _require(bool(((offsets[:, 0] >= 0) & (offsets[:, 1] >= offsets[:, 0])
                    & (offsets[:, 1] <= len(response))).all()), 'Original R offsets outside supplied unchanged text')
    shift = original['full_offsets'][P:] - offsets
    _require(bool((shift == shift[0, 0]).all()) and shift[0, 0] > 0, 'Original R/chat offset shift differs')
    chat_characters = int(shift[0, 0])
    _require(bool((original['full_offsets'][:P] >= 0).all())
             and bool((original['full_offsets'][:P, 1] >= original['full_offsets'][:P, 0]).all())
             and bool((original['full_offsets'][:P] <= chat_characters).all()), 'Original prompt offsets invalid')
    for key, dtype in (('target', np.int64), ('token_id', np.int64),
                       ('answer_index', np.int32), ('unit_index', np.int32)):
        value = original.get(key)
        _require(isinstance(value, np.ndarray) and value.ndim == 1 and value.dtype == dtype,
                 'Original packed axis dtype changed: ' + key)
    targets = original['target']
    _require(len(targets) > 0 and bool(((targets >= 0) & (targets < T)).all())
             and bool((np.diff(targets) > 0).all())
             and all(len(original[key]) == len(targets) for key in ('token_id', 'answer_index', 'unit_index'))
             and np.array_equal(original['token_id'], original['answer_ids'][targets]), 'Original packed valid axis differs')
    _require(int(original['answer_ids'][73]) == 539 and int(original['answer_ids'][87]) == 2561,
             'Frozen original operator/foil token IDs changed')
    if 'response_sha256' in original:
        _require(original['response_sha256'] == _hash(response), 'Original response byte identity changed')
    return chat_characters


def _fields(source, source_id, supplied):
    _require(isinstance(supplied, list) and bool(supplied), 'Frozen source-field list required')
    inventory = source_inventory(source, source_id)
    _require(inventory['status'] == 'structural_fields_only', 'Explicit typed source fields required')
    actual = {tuple(field['path']): field for field in inventory['fields']}
    checked = {}
    spans = []
    for field in supplied:
        _require(isinstance(field, dict) and isinstance(field.get('path'), list), 'Explicit full field path required')
        path = tuple(field['path'])
        _require(path in actual and path not in checked, 'Missing/duplicate source-field path')
        expected = actual[path]
        span = _span(field.get('span'), len(source))
        _require(list(span) == expected['span'] and field.get('scalar_type') == expected['scalar_type']
                 and type(field.get('value')) is type(expected['value']) and field['value'] == expected['value']
                 and field.get('raw_text') == source[span[0]:span[1]], 'Frozen field type/value/span differs from source')
        if 'key_span' in field:
            _require(field['key_span'] == expected['key_span'], 'Frozen field key span differs')
        if 'source_field_span' in field:
            _require(field['source_field_span'] == [expected['key_span'][0], span[1]], 'Frozen full field span differs')
        checked[path] = expected
        spans.append(span)
    spans.sort()
    _require(all(left[1] <= right[0] for left, right in zip(spans, spans[1:])), 'Overlapping supplied scalar spans')
    for path, value in ((VALET, False), (GARAGE, False), (OUTDOOR, True), (STREET, True)):
        _require(path in checked and checked[path]['scalar_type'] == 'boolean'
                 and type(checked[path]['value']) is bool and checked[path]['value'] is value,
                 'Required original bool field/value absent')
        _require(checked[path]['raw_text'] == str(value), 'Only existing literal True/False payloads are eligible')
    return inventory, actual, checked


def _one_token(offsets, span, source_mask):
    left, right = span
    overlap = [index for index, (a, b) in enumerate(offsets)
               if a < right and b > left and b > a]
    _require(len(overlap) == 1 and source_mask[overlap[0]], 'Each edited bool must have one source token')
    index = overlap[0]
    _require(offsets[index][0] <= left and offsets[index][1] >= right, 'Boolean token does not cover complete value')
    return index


def prepare_swaps(sample, source_fields, tokenizer, original_axes):
    """Return worlds[target/control] plus an audit; no S0/R tokenizer or decode.

    Each world contains source/prompt/response, P/T/N, copied original ``axes``
    with the new prompt IDs/offsets, exact ``edits`` and ``field_proofs``. Caller
    must bind the supplied original response bytes to its sealed frontend case;
    optional original_axes['response_sha256'] additionally enforces that binding.
    No semantic scores, new labels or model data are accepted/selected here.
    """
    source, prompt, response = (_sample(sample, key) for key in ('source', 'prompt', 'response'))
    _require(all(type(value) is str and value for value in (source, prompt, response)), 'Nonempty raw S/P/R strings required')
    _require(prompt.count(source) == 1, 'Original source must occur exactly once in prompt')
    _require(getattr(tokenizer, 'is_fast', False), 'Fast offsets required; no fallback tokenizer')
    source_id = str(_sample(sample, 'source_id'))
    chat_characters = _axes(original_axes, response)
    original_inventory, actual_fields, checked = _fields(source, source_id, source_fields)
    worlds = {}
    old_chat_frame = None
    for name, paths in WORLD_EDITS.items():
        first, donor = paths
        edit_records = [dict(path=list(path), old_span=checked[path]['span'], old_text=checked[path]['raw_text'],
                             new_text=checked[other]['raw_text']) for path, other in ((first, donor), (donor, first))]
        new_source = source
        for edit in sorted(edit_records, key=lambda item: item['old_span'][0], reverse=True):
            a, b = edit['old_span']
            _require(new_source[a:b] == edit['old_text'], 'Descending edit original span changed')
            new_source = new_source[:a] + edit['new_text'] + new_source[b:]
        _require(len(new_source) == len(source), 'Swap changed total source character length')
        start = prompt.index(source)
        new_prompt = prompt[:start] + new_source + prompt[start+len(source):]
        _require(new_prompt.count(new_source) == 1 and len(new_prompt) == len(prompt), 'New source/prompt identity ambiguous')
        recorder = _RecordingTokenizer(tokenizer)
        pack = encode_source(recorder, new_prompt, new_source)
        _require(len(recorder.calls) == len(recorder.renders) == 1, 'Only one prompt string/chat call per world')
        chat, (encoded_text, encoded) = recorder.renders[0], recorder.calls[0]
        _require(chat == encoded_text and len(chat) == chat_characters, 'Rendered chat or original R character shift changed')
        fragment = new_prompt if new_prompt in chat else new_prompt.strip()
        _require(fragment and chat.count(fragment) == 1 and fragment.count(new_source) == 1, 'Rendered source/prompt ambiguous')
        source_start = chat.index(fragment) + fragment.index(new_source)
        old_chat = chat[:source_start] + source + chat[source_start+len(new_source):]
        _require(old_chat[source_start:source_start+len(source)] == source, 'Cannot reconstruct old source frame')
        if old_chat_frame is None:
            old_chat_frame = old_chat
        _require(old_chat == old_chat_frame, 'Chat frame differs beyond the source edits')
        align_source_spans(original_inventory, source, old_chat + response, source_start,
            original_axes['full_offsets'].tolist(), original_axes['token_ids'].tolist(),
            original_axes['token_ids'].tolist(), original_axes['source_mask'].tolist())
        offsets = _offsets(encoded['offset_mapping'], len(chat))
        ids = pack['prompt_with_source']
        _require(isinstance(ids, list) and len(ids) == P and len(offsets) == P
                 and all(type(value) is int and value >= 0 for value in ids), 'New complete prompt ID axis changed')
        _require(ids == encoded['input_ids'], 'Core/recorded prompt IDs differ')
        mask = pack['source_mask']
        _require(isinstance(mask, list) and all(type(value) is bool for value in mask)
                 and np.array_equal(np.asarray(mask), original_axes['source_mask'][:P])
                 and sum(mask) == SOURCE_TOKENS, 'New source mask changed')
        old_prompt = original_axes['token_ids'][:P]
        old_offsets = original_axes['full_offsets'][:P].tolist()
        positions = [_one_token(old_offsets, tuple(source_start + v for v in checked[path]['span']),
                                original_axes['source_mask'][:P]) for path in paths]
        _require(positions[0] != positions[1] and old_prompt[positions[0]] != old_prompt[positions[1]], 'Two distinct Boolean token IDs required')
        expected = old_prompt.copy()
        expected[positions] = expected[positions[::-1]]
        _require(np.array_equal(np.asarray(ids, dtype=np.int64), expected), 'Only the fixed two Boolean ID positions may swap')
        _require(Counter(np.asarray(ids)[mask].tolist()) == Counter(old_prompt[np.asarray(mask)].tolist()), 'Source model-token Counter changed')
        changed = np.flatnonzero(np.asarray(ids) != old_prompt).tolist()
        _require(changed == sorted(positions), 'New prompt changed other token positions')
        new_inventory = source_inventory(new_source, source_id)
        _require(new_inventory['status'] == 'structural_fields_only', 'Edited source no longer explicit typed data')
        new_fields = {tuple(field['path']): field for field in new_inventory['fields']}
        _require(set(new_fields) == set(actual_fields), 'Source structure changed outside Boolean payloads')
        for path, original in actual_fields.items():
            expected_value = (not original['value']) if path in paths else original['value']
            _require(new_fields[path]['scalar_type'] == original['scalar_type']
                     and type(new_fields[path]['value']) is type(expected_value)
                     and new_fields[path]['value'] == expected_value, 'Unlicensed source field changed')
        full_ids = ids + original_axes['answer_ids'].tolist()
        full_mask = mask + [False] * T
        full_offsets = offsets + original_axes['full_offsets'][P:].tolist()
        aligned = align_source_spans(new_inventory, new_source, chat + response, source_start,
                                      full_offsets, full_ids, expected.tolist() + original_axes['answer_ids'].tolist(), full_mask)
        mappings = {row['occurrence_id']: row for row in aligned['occurrences'] if row['kind'] == 'fields'}
        proofs = []
        for path, original in checked.items():
            field = new_fields[path]
            closure_character = source_start + field['span'][1] - 1
            closure = [index for index, (a, b) in enumerate(offsets) if a <= closure_character < b]
            _require(len(closure) == 1 and mask[closure[0]], 'Field closure is ambiguous on actual offsets')
            proofs.append(dict(path=list(path), original_value=original['value'], value=field['value'],
                old_value_span=original['span'], value_span=field['span'], key_span=field['key_span'],
                field_span=[field['key_span'][0], field['span'][1]],
                closure_token=closure[0], closure_token_id=ids[closure[0]],
                source_field_complete_after_character=field['span'][1]-1,
                mapping=mappings[field['occurrence_id']]))
        for edit, position in zip(edit_records, positions, strict=True):
            field = new_fields[tuple(edit['path'])]
            _require(_one_token(offsets, tuple(source_start + v for v in field['span']), mask) == position,
                     'Edited bool moved to a different token position')
            edit.update(new_span=field['span'], token_position=position,
                        old_token_id=int(old_prompt[position]), new_token_id=ids[position])
        axes = {key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
                for key, value in original_axes.items()}
        axes.update(token_ids=np.asarray(full_ids, dtype=np.int64), source_mask=np.asarray(full_mask, dtype=bool),
                    full_offsets=np.asarray(full_offsets, dtype=np.int64))
        worlds[name] = dict(world=name, source=new_source, prompt=new_prompt, response=response,
            P=P, T=T, N=N, axes=axes, chat=chat, edits=edit_records, field_proofs=proofs,
            source_start_in_chat=source_start, source_sha256=_hash(new_source),
            prompt_sha256=_hash(new_prompt), response_sha256=_hash(response))
    return dict(worlds=worlds, audit=dict(passed=True, P=P, T=T, N=N, source_tokens=SOURCE_TOKENS,
        original_source_sha256=original_inventory['source_sha256'], response_sha256=_hash(response),
        source_encodings=2, chat_template_renders=2, response_tokenizations=0, response_decodes=0,
        S0_encodings=0, model_calls=0, annotation_reads=0,
        response_identity_scope='Sealed caller bytes; optional supplied response SHA checked; R never encoded/decoded',
        semantic_scope='Only fixed Boolean field assignments; weak OR and earlier hours history unchanged'))

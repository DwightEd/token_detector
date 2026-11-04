"""Fake-only exact-axis/source-edit tests; no HF tokenizer, weights or labels."""

from dataclasses import replace
import copy
import hashlib

import numpy as np
import pytest

from experiments.datasets import Sample
from experiments.research.operator_owner_swap import (
    GARAGE, OUTDOOR, STREET, VALET, prepare_swaps,
)
from experiments.research.relation_inventory import source_inventory
from myproject.inputs import encode_source


class FakeTokenizer:
    is_fast = True

    def __init__(self):
        self.calls, self.messages = [], []

    @staticmethod
    def lex(text):
        ids, offsets, cursor = [], [], 0
        while cursor < len(text):
            if text.startswith('False', cursor):
                width, token = 5, 8000
            elif text.startswith('True', cursor):
                width, token = 4, 8001
            else:
                width = 1
                token = {'~': 539, '^': 2561}.get(text[cursor], ord(text[cursor]) + 100)
            ids.append(token); offsets.append([cursor, cursor + width]); cursor += width
        return dict(input_ids=ids, offset_mapping=offsets)

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.messages.append(messages[0]['content'])
        return '<u>' + messages[0]['content'] + '</u><a>'

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False and return_offsets_mapping is True
        self.calls.append(text)
        return self.lex(text)

    def decode(self, *unused, **kwargs):
        raise AssertionError('Neither R nor source decoding is permitted')


def fixture():
    template = ("{'attributes': {'BusinessParking': {'valet': False, 'garage': False, "
                "'street': True}, 'OutdoorSeating': True}, 'pad': '%s'}")
    empty = template % ''
    source = template % ('x' * (707 - len(FakeTokenizer.lex(empty)['input_ids'])))
    prompt = 'Q' * 86 + source
    response = list('r' * 233); response[73], response[87] = '~', '^'; response = ''.join(response)
    sample = Sample('5684', '13603', 'test', 'Data2txt', prompt, source, response, 'fake')
    tokenizer = FakeTokenizer()
    pack = encode_source(tokenizer, prompt, source)
    chat = tokenizer.calls[0]
    answer = FakeTokenizer.lex(response)
    assert len(pack['prompt_with_source']) == 803 and sum(pack['source_mask']) == 707
    offsets = FakeTokenizer.lex(chat)['offset_mapping']
    answer_offsets = np.asarray(answer['offset_mapping'], dtype=np.int64)
    axes = dict(token_ids=np.asarray(pack['prompt_with_source'] + answer['input_ids'], dtype=np.int64),
        answer_ids=np.asarray(answer['input_ids'], dtype=np.int64),
        source_mask=np.asarray(pack['source_mask'] + [False] * 233, dtype=bool),
        prechoice_positions=np.arange(802, 1035, dtype=np.int64),
        all_query_positions=np.arange(802, 1036, dtype=np.int64),
        answer_offsets=answer_offsets,
        full_offsets=np.vstack((np.asarray(offsets, dtype=np.int64), answer_offsets + len(chat))),
        target=np.arange(233, dtype=np.int64), token_id=np.asarray(answer['input_ids'], dtype=np.int64),
        answer_index=np.full(233, 8, dtype=np.int32), unit_index=np.arange(233, dtype=np.int32) // 7,
        response_sha256=hashlib.sha256(response.encode()).hexdigest())
    fields = source_inventory(source, '13603')['fields']
    tokenizer.calls.clear(); tokenizer.messages.clear()
    return sample, fields, tokenizer, axes


def test_exact_two_worlds_no_response_or_original_prompt_encoding():
    sample, fields, tokenizer, axes = fixture()
    before = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in axes.items()}
    result = prepare_swaps(sample, fields, tokenizer, axes)
    assert set(result['worlds']) == {'target', 'control'}
    assert result['audit']['source_encodings'] == result['audit']['chat_template_renders'] == 2
    assert result['audit']['response_tokenizations'] == result['audit']['response_decodes'] == 0
    assert result['audit']['S0_encodings'] == result['audit']['model_calls'] == 0
    assert len(tokenizer.calls) == len(tokenizer.messages) == 2
    assert all(value != sample.prompt for value in tokenizer.messages)
    assert all(value != sample.response for value in tokenizer.calls)
    for name, world in result['worlds'].items():
        parsed = {tuple(row['path']): row for row in source_inventory(world['source'], '13603')['fields']}
        assert parsed[OUTDOOR]['value'] is False
        assert parsed[STREET]['value'] is True
        assert parsed[VALET]['value'] is (name == 'target')
        assert parsed[GARAGE]['value'] is (name == 'control')
        assert world['response'] == sample.response
        assert world['prompt'] == sample.prompt.replace(sample.source, world['source'])
        assert (world['P'], world['T'], world['N']) == (803, 233, 1036)
        changed = np.flatnonzero(world['axes']['token_ids'] != axes['token_ids'])
        assert len(changed) == 2 and changed.max() < 803
        assert np.array_equal(world['axes']['token_ids'][changed], axes['token_ids'][changed[::-1]])
        assert np.array_equal(world['axes']['token_ids'][803:], axes['answer_ids'])
        for key in ('answer_ids', 'answer_offsets', 'target', 'token_id', 'unit_index',
                    'answer_index', 'prechoice_positions', 'all_query_positions'):
            assert np.array_equal(world['axes'][key], axes[key])
            assert world['axes'][key] is not axes[key]
        assert np.array_equal(world['axes']['source_mask'], axes['source_mask'])
        assert np.array_equal(world['axes']['full_offsets'][803:], axes['full_offsets'][803:])
    for key, value in before.items():
        assert np.array_equal(axes[key], value) if isinstance(value, np.ndarray) else axes[key] == value


def test_field_char_offsets_realign_without_moving_boolean_token_positions():
    sample, fields, tokenizer, axes = fixture()
    result = prepare_swaps(sample, fields, tokenizer, axes)
    old = {tuple(field['path']): field for field in fields}
    for name, world in result['worlds'].items():
        proofs = {tuple(field['path']): field for field in world['field_proofs']}
        # One earlier False->True shortens chars until the later True->False.
        assert proofs[STREET]['value_span'] == [v - 1 for v in old[STREET]['span']]
        assert proofs[OUTDOOR]['value_span'][0] == old[OUTDOOR]['span'][0] - 1
        assert proofs[OUTDOOR]['value_span'][1] == old[OUTDOOR]['span'][1]
        for edit in world['edits']:
            a, b = edit['new_span']
            assert world['source'][a:b] == edit['new_text']
            proof = proofs[tuple(edit['path'])]
            assert proof['closure_token'] == edit['token_position']
            assert proof['closure_token_id'] == edit['new_token_id']
            assert proof['mapping']['overlap_tokens'] == [edit['token_position']]


@pytest.mark.parametrize('failure', ['missing', 'wrong_bool_type', 'overlap', 'key_span', 'duplicate'])
def test_bad_frozen_field_contract_fails_before_any_tokenizer_call(failure):
    sample, fields, tokenizer, axes = fixture()
    fields = copy.deepcopy(fields)
    selected = next(field for field in fields if tuple(field['path']) == VALET)
    if failure == 'missing':
        fields.remove(selected)
    elif failure == 'wrong_bool_type':
        selected['value'] = 0
    elif failure == 'overlap':
        selected['span'] = next(field['span'] for field in fields if tuple(field['path']) == GARAGE)
    elif failure == 'key_span':
        selected['key_span'][0] += 1
    else:
        fields.append(copy.deepcopy(selected))
    with pytest.raises(ValueError):
        prepare_swaps(sample, fields, tokenizer, axes)
    assert tokenizer.calls == tokenizer.messages == []


@pytest.mark.parametrize('failure', ['repeated_source', 'response', 'nonbool_source', 'mask_dtype', 'offsets'])
def test_raw_original_identity_or_axis_drift_never_retokenizes_response(failure):
    sample, fields, tokenizer, axes = fixture()
    if failure == 'repeated_source':
        sample = replace(sample, prompt=sample.prompt + sample.source)
    elif failure == 'response':
        sample = replace(sample, response=sample.response[:-1] + '!')
    elif failure == 'nonbool_source':
        new = sample.source.replace('False', '0', 1)
        sample = replace(sample, source=new, prompt=sample.prompt.replace(sample.source, new))
    elif failure == 'mask_dtype':
        axes['source_mask'] = axes['source_mask'].astype(np.int64)
    else:
        axes['answer_offsets'] = axes['answer_offsets'].astype(np.float64)
    with pytest.raises(ValueError):
        prepare_swaps(sample, fields, tokenizer, axes)
    assert tokenizer.calls == tokenizer.messages == []


@pytest.mark.parametrize('failure', ['extra_token', 'mask', 'fractional_offset', 'bool_offset'])
def test_new_encoding_mismatch_fails_without_world_or_format_rescue(failure):
    sample, fields, tokenizer, axes = fixture()

    class Corrupt(FakeTokenizer):
        def __call__(self, text, **kwargs):
            encoded = super().__call__(text, **kwargs)
            if failure == 'extra_token':
                encoded['input_ids'][0] += 10
            elif failure == 'mask':
                start = text.index('{')
                encoded['offset_mapping'][0] = [start, start+1]
            elif failure == 'fractional_offset':
                encoded['offset_mapping'][0][0] = .25
            else:
                encoded['offset_mapping'][0][0] = False
            return encoded

    tokenizer = Corrupt()
    with pytest.raises(ValueError):
        prepare_swaps(sample, fields, tokenizer, axes)
    assert len(tokenizer.calls) == len(tokenizer.messages) == 1

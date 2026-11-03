"""Synthetic tokenizer/axis checks; no natural inputs, model or annotations."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.datasets import Sample
from experiments.relation_alignment import _original_axes, reconstruct_alignment
from experiments.research.relation_inventory import source_inventory
from myproject.inputs import encode_response, encode_source


class SyntheticFastTokenizer:
    is_fast = True
    all_special_ids = []
    added_tokens_decoder = {}

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return '<user>' + messages[0]['content'].strip() + '</user><assistant>'

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert add_special_tokens is False and return_offsets_mapping is True
        self.calls.append(text)
        ids, offsets, cursor = [], [], 0
        while cursor < len(text):
            # A hypothetical joint prompt+R encoding merges the boundary.
            width = 2 if text[cursor:cursor + 2] == '>r' else 1
            ids.append(777 if width == 2 else ord(text[cursor]) + 1000)
            offsets.append([cursor, cursor + width])
            cursor += width
        return dict(input_ids=ids, offset_mapping=offsets)

    def decode(self, ids):
        return ''.join('>r' if value == 777 else chr(value - 1000) for value in ids)


def _fixture(source='甲100乙200'):
    sample = Sample('case', 'source', 'test', 'QA', '  Question:\n' + source + '\nend  ',
                    source, 'response', 'generator')
    tokenizer = SyntheticFastTokenizer()
    prompt = encode_source(tokenizer, sample.prompt, sample.source)
    answer = encode_response(tokenizer, sample.response)
    ids = prompt['prompt_with_source'] + answer['answer_ids']
    axes = dict(token_ids=ids, answer_ids=answer['answer_ids'],
                source_mask=prompt['source_mask'] + [False] * len(answer['answer_ids']),
                query_positions=list(range(len(prompt['prompt_with_source']) - 1, len(ids))))
    tokenizer.calls.clear()
    return sample, tokenizer, axes, source_inventory(source, sample.source_id)


def test_trim_fallback_unicode_and_separate_answer_encoding_are_exact():
    sample, tokenizer, axes, inventory = _fixture()
    report = reconstruct_alignment(sample, tokenizer, axes, inventory)
    chat = tokenizer.apply_chat_template([{'content': sample.prompt}], tokenize=False, add_generation_prompt=True)
    assert report['status'] == 'original_axes_matched'
    assert report['full_tokens'] == len(axes['token_ids'])
    assert report['query_rows'] == len(axes['answer_ids']) + 1
    assert report['source_semantic_status'] == 'unknown_not_inferred'
    assert tokenizer.calls == [chat, sample.response, chat]
    assert chat + sample.response not in tokenizer.calls
    joint = tokenizer(chat + sample.response, add_special_tokens=False, return_offsets_mapping=True)
    assert joint['input_ids'] != axes['token_ids']
    for occurrence in report['alignment']['occurrences']:
        start, stop = occurrence['source_span']
        assert sample.source[start:stop] in ('100', '200')
        assert len(occurrence['contained_tokens']) == stop - start


@pytest.mark.parametrize('name', ['token_ids', 'answer_ids', 'source_mask', 'query_positions'])
def test_mismatched_original_axes_are_rejected(name):
    sample, tokenizer, axes, inventory = _fixture()
    axes[name] = axes[name].copy()
    axes[name][0] = not axes[name][0] if name == 'source_mask' else axes[name][0] + 1
    with pytest.raises(ValueError, match=r'original full token/mask/T\+1'):
        reconstruct_alignment(sample, tokenizer, axes, inventory)


def test_nonfast_tokenizer_is_not_allowed_to_guess_offsets():
    sample, tokenizer, axes, inventory = _fixture()
    tokenizer.is_fast = False
    with pytest.raises(ValueError, match='Fast tokenizer'):
        reconstruct_alignment(sample, tokenizer, axes, inventory)


@pytest.mark.parametrize('bad', [0.0, False])
def test_malformed_offset_values_cannot_be_silently_cast_to_integers(bad):
    sample, _, axes, inventory = _fixture()

    class MalformedOffsets(SyntheticFastTokenizer):
        def __call__(self, *args, **kwargs):
            result = super().__call__(*args, **kwargs)
            result['offset_mapping'][0][0] = bad
            return result

    with pytest.raises(ValueError, match='exact non-Boolean integers'):
        reconstruct_alignment(sample, MalformedOffsets(), axes, inventory)


def test_source_byte_identity_and_ambiguous_prompt_are_rejected():
    sample, tokenizer, axes, inventory = _fixture()
    with pytest.raises(ValueError):
        reconstruct_alignment(replace(sample, source='乙100甲200'), tokenizer, axes, inventory)
    repeated = replace(sample, prompt=sample.prompt + sample.source)
    with pytest.raises(ValueError):
        reconstruct_alignment(repeated, tokenizer, axes, inventory)


def test_axis_reader_does_not_decode_internal_state_fields(tmp_path):
    path = tmp_path / 'arrays.npz'
    # Object Q is intentionally forbidden to load; integer axes remain readable.
    np.savez_compressed(path, token_ids=np.array([1, 2], dtype=np.int64),
        answer_ids=np.array([2], dtype=np.int64), source_mask=np.array([True, False]),
        query_positions=np.array([0, 1], dtype=np.int64), query=np.array([object()], dtype=object))
    assert _original_axes(path)['token_ids'] == [1, 2]


def test_axis_reader_rejects_silent_integer_casts(tmp_path):
    path = tmp_path / 'arrays.npz'
    np.savez_compressed(path, token_ids=np.array([1., 2.]), answer_ids=np.array([2], dtype=np.int64),
        source_mask=np.array([True, False]), query_positions=np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match='integer'):
        _original_axes(path)

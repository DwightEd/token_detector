"""Encode a prompt, its source passage and an existing response."""

import re

import numpy as np


BOUNDARY = re.compile(r'\n+|[.!?。！？]+["”’\')\]]*(?=\s|$)')
LIST_MARKER = re.compile(r'\s*(?:\d+[.)]|[-*•])\s*')


def text_units(token_text, max_tokens=64):
    """Group at original newlines/sentence ends; cap local contexts at 64 tokens."""
    text = ''.join(token_text)
    ends = np.cumsum([len(piece) for piece in token_text])
    boundaries = [0]
    text_start = 0
    for match in BOUNDARY.finditer(text):
        fragment = text[text_start:match.end()]
        if fragment.strip() and not LIST_MARKER.fullmatch(fragment):
            stop = int(np.searchsorted(ends, match.end(), side='left')) + 1
            if stop > boundaries[-1]:
                boundaries.append(stop)
                text_start = int(ends[stop - 1])
    if boundaries[-1] != len(token_text):
        boundaries.append(len(token_text))

    units = []
    for left, right in zip(boundaries, boundaries[1:]):
        for start in range(left, right, max_tokens):
            units.append(dict(start=start, stop=min(start + max_tokens, right),
                              text_start=left, text_stop=right))
    return units


def encode_source(tokenizer, prompt: str, source: str) -> dict:
    """Mark the unique source passage inside the complete user prompt."""
    if not source or prompt.count(source) != 1:
        raise ValueError("source must be a nonempty substring occurring once in prompt")
    start = prompt.index(source)
    stop = start + len(source)
    chat = tokenizer.apply_chat_template(
        [dict(role="user", content=prompt)], tokenize=False, add_generation_prompt=True,
    )
    shift = chat.index(prompt)
    encoded = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)
    offsets = np.asarray(encoded["offset_mapping"])
    mask = (offsets[:, 0] < shift + stop) & (offsets[:, 1] > shift + start)
    return {
        "prompt_with_source": encoded["input_ids"], "source_mask": mask.tolist(),
        "prompt_without_source": [token for token, remove in zip(encoded["input_ids"], mask)
                                  if not remove],
    }


def encode_response(tokenizer, response: str) -> dict:
    """Retain original response tokens, character offsets and local text units."""
    encoded = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    token_text = [tokenizer.decode([token]) for token in encoded["input_ids"]]
    marked = {index for index, token in tokenizer.added_tokens_decoder.items() if token.special}
    special_ids = sorted(set(tokenizer.all_special_ids) | marked)
    return {
        "answer_ids": encoded["input_ids"], "offsets": encoded["offset_mapping"],
        "text": response, "special_ids": special_ids, "units": text_units(token_text),
    }

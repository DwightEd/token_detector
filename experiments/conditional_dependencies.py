"""Exact physical word-deletion effects inside a single frozen-LM passage."""

from collections import Counter

import numpy as np
import torch

from experiments.research.contextual_alignment import STOPWORDS, WORDS
from myproject.inputs import encode_response
from myproject.model import prefill, score_continuation


def lexical_anchors(text, offsets):
    """Unique content words whose token pieces overlap no other lexical word."""
    matches = list(WORDS.finditer(text))
    counts = Counter(m.group().casefold() for m in matches)
    offsets = np.asarray(offsets)
    token_words = ((offsets[:, 0, None] < np.array([m.end() for m in matches]))
                   & (offsets[:, 1, None] > np.array([m.start() for m in matches])))
    anchors = []
    for index, match in enumerate(matches):
        word = match.group().casefold()
        rows = np.flatnonzero(token_words[:, index])
        if (counts[word] == 1 and word not in STOPWORDS and len(rows)
                and np.all(token_words[rows].sum(1) == 1)):
            anchors.append({'word': word, 'start': match.start(), 'end': match.end(),
                            'tokens': rows.tolist()})
    return anchors


@torch.no_grad()
def deletion_dependencies(model, tokenizer, text):
    """e(u,v)=mean_v(log p_without_u - log p_full); backwards stays NaN.

    One intact and one physically deleted world per automatically selected anchor.
    Each world uses the stable library's chunked native probability calculation;
    the runner separately counts actual backbone calls, not just worlds.
    """
    encoded = encode_response(tokenizer, text)
    ids = encoded['answer_ids']
    chat = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': 'Continue the passage.'}],
        tokenize=False, add_generation_prompt=True)
    prefix = tokenizer(chat, add_special_tokens=False, return_offsets_mapping=True)['input_ids']
    anchors = lexical_anchors(text, encoded['offsets'])

    def probabilities(tokens):
        cache = prefill(model, prefix[:-1])
        return score_continuation(model, cache, prefix[-1], tokens)

    baseline = probabilities(ids)
    noop_error = float(np.max(np.abs(probabilities(ids) - baseline)))
    matrix = np.full((len(anchors), len(anchors)), np.nan)
    worlds = 2
    for sender, anchor in enumerate(anchors):
        receivers = [i for i, target in enumerate(anchors)
                     if anchor['tokens'][-1] < target['tokens'][0]]
        if not receivers:
            continue
        keep = np.ones(len(ids), dtype=bool)
        keep[anchor['tokens']] = False
        remaining = np.flatnonzero(keep)
        changed = probabilities([ids[i] for i in remaining])
        worlds += 1
        original_to_remaining = np.full(len(ids), -1, dtype=int)
        original_to_remaining[remaining] = np.arange(len(remaining))
        for receiver in receivers:
            rows = np.asarray(anchors[receiver]['tokens'])
            matrix[sender, receiver] = np.mean(changed[original_to_remaining[rows]] - baseline[rows])
    return {'anchors': anchors, 'matrix': matrix, 'logp': baseline,
            'token_ids': np.asarray(ids), 'offsets': np.asarray(encoded['offsets']),
            'worlds': worlds, 'noop_max_abs_logp': noop_error}

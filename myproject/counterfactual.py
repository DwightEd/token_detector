"""Matched response interventions and retrospective source-support interactions."""

import numpy as np
from scipy.special import logsumexp
import torch

from .model import prefill, score_continuation


@torch.no_grad()
def observe_responses(model, source, answers):
    """Teacher-force supplied responses with two prompt prefills, both SDPA.

    Reuse only the original prompt KV prefix. Columns are source-present and
    source-absent log probabilities on each response's own raw token axis.
    No routing observer, labels, generation or factual judge is involved.
    """
    if not answers or any(not len(answer) for answer in answers):
        raise ValueError('Provide nonempty response token sequences')
    worlds = []
    for name in ['prompt_with_source', 'prompt_without_source']:
        prompt = source[name]
        cache = prefill(model, prompt[:-1])
        rows = []
        for answer in answers:
            cache.crop(len(prompt)-1)
            rows.append(score_continuation(model, cache, prompt[-1], answer))
        worlds.append(rows)
    return [np.column_stack(pair).astype(np.float64)
            for pair in zip(*worlds, strict=True)]


def repair_support(original, alternatives):
    """Source-by-response-repair interaction on an unchanged future sequence.

    Inputs: [future, present/absent] and [candidate, future, present/absent].
    Uniform candidate priors are mixed at SEQUENCE level in each world.
    R = (log mixture_present - log mixture_absent) - original source support.
    This signed predictive interaction is not a factual-error probability.
    """
    original, alternatives = np.asarray(original, float), np.asarray(alternatives, float)
    if (original.ndim != 2 or original.shape[1] != 2 or not len(original)
            or alternatives.ndim != 3 or not len(alternatives)
            or alternatives.shape[1:] != original.shape
            or not np.isfinite(original).all() or not np.isfinite(alternatives).all()):
        raise ValueError('Matching finite nonempty future/candidate likelihood axes required')
    cumulative = np.cumsum(alternatives, axis=1)
    mixture = logsumexp(cumulative, axis=0)-np.log(len(alternatives))
    baseline = np.cumsum(original[:, 0]-original[:, 1])
    contrast = mixture[:, 0]-mixture[:, 1]-baseline
    totals = alternatives.sum(axis=1)
    return {'score': float(contrast[-1]), 'increments': np.diff(np.r_[0., contrast]),
            'candidate_gains': totals[:, 0]-totals[:, 1]-baseline[-1],
            'absent_weights': np.exp(totals[:, 1]-logsumexp(totals[:, 1]))}

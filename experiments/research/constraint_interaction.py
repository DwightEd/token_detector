"""Source-derived arithmetic witnesses and frozen token interaction features."""

from decimal import Decimal, localcontext
from hashlib import sha256
from itertools import combinations

import numpy as np
import torch
from scipy.special import logsumexp

from experiments.research.table_bindings import ATOMS
from myproject.model import attention_backend, input_ids, prefill


def numbers(text):
    def embedded(position):
        return 0 <= position < len(text) and (text[position].isalnum() or text[position] == '_')

    return {Decimal(m[0].replace(',', '')) for m in ATOMS.finditer(text)
            if not embedded(m.start()-1) and not embedded(m.end())}


def literal(value):
    return format(value, 'f').rstrip('0').rstrip('.') if '.' in format(value, 'f') else format(value, 'f')


def equations(a, b, c, swap=False):
    with localcontext() as context:
        context.prec = 80
        values = [a-b, a-c]
    if swap:
        values.reverse()
    text, spans = '', []
    for index, (y, value) in enumerate(zip([b, c], values, strict=True)):
        text += f'({literal(a)}) - ({literal(y)}) = ('
        start = len(text)
        text += literal(value)
        spans.append([start, len(text)])
        text += '); ' if index == 0 else ').'
    return {'text': text, 'rhs_spans': spans}


def witnesses(source, excluded, maximum=16):
    """Certificates use source literals; exclusion prevents query-number leakage.

    Arithmetic identities do not certify financial relevance or correct account binding.
    """
    blocked = numbers(excluded)
    blocked |= {-v for v in blocked}
    operands = sorted(v for v in numbers(source) if v not in blocked
                      and not (v == int(v) and 1900 <= v <= 2099))
    candidates = []
    with localcontext() as context:
        context.prec = 80
        for triple in combinations(operands, 3):
            key = '|'.join(literal(v) for v in triple)
            digest = sha256(key.encode()).hexdigest()
            if any({x-y, y-x} & blocked for x, y in combinations(triple, 2)):
                continue
            candidates.append((digest, triple))
        pairs = []
        for i, (digest, triple) in enumerate(sorted(candidates)[:maximum]):
            # Alternation after ranking guarantees sign balance. A separate
            # hash domain determines result order, independent of rank selection.
            index = 0 if i % 2 == 0 else 2
            a = triple[index]
            b, c = [v for j, v in enumerate(triple) if j != index]
            order = sha256(('order|'+digest).encode()).digest()[0]
            if order % 2:
                b, c = c, b
            pairs.append({'id': digest, 'a': literal(a), 'b': literal(b), 'c': literal(c),
                          'positive': equations(a, b, c), 'negative': equations(a, b, c, swap=True)})
    return pairs


def belief_worlds(pair):
    """Factorial probes: source-valid conclusion and agreement vary independently."""
    a, b, c = (Decimal(pair[k]) for k in ['a', 'b', 'c'])
    with localcontext() as context:
        context.prec = 80
        results = [a-b, a-c]
    worlds = []
    for prior_risk in [0, 1]:
        for risk in [0, 1]:
            text = pair['negative' if prior_risk else 'positive']['text']+'\n\n'
            values = results[::-1] if risk else results
            spans = []
            for index, (y, value) in enumerate(zip([b, c], values, strict=True)):
                text += ('The difference of ' if index == 0 else ' and the difference of ')
                text += f'({literal(a)}) and ({literal(y)}) is ('
                start = len(text); text += literal(value); spans.append([start, len(text)])
                text += '),' if index == 0 else ').'
            worlds.append({'text': text, 'rhs_spans': spans, 'risk': risk,
                           'coherence': int(prior_risk == risk), 'prior_risk': prior_risk})
    return worlds


def belief_model(features, pair_index, risk, coherence):
    """Equal proof-pair weights; four source-certified Gaussian observation states.

    Spherical covariance is a stated working model, not a calibration guarantee.
    No benchmark labels, weight optimization or fitted mixture weights enter.
    """
    features = np.asarray(features, float)
    pair_index, risk, coherence = (np.asarray(x) for x in [pair_index, risk, coherence])
    if (features.ndim != 2 or not np.isfinite(features).all()
            or any(x.shape != (len(features),) for x in [pair_index, risk, coherence])
            or not np.isin(risk, [0, 1]).all() or not np.isin(coherence, [0, 1]).all()):
        raise ValueError('Finite token features and matching certified states required')
    pairs = np.unique(pair_index)
    if len(pairs) < 2:
        raise ValueError('At least two proof groups required')
    means, variances = [], []
    for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        groups = [features[(pair_index == p) & (risk == r) & (coherence == c)] for p in pairs]
        if any(not len(g) for g in groups):
            raise ValueError('Every proof must cover all four states')
        mean = np.mean([g.mean(0) for g in groups], axis=0)
        means.append(mean)
        variances.append(np.mean([np.mean((g-mean)**2) for g in groups]))
    means = np.asarray(means)
    variance = max(float(np.mean(variances)), 1e-12)
    binary_means = means.reshape(2, 2, -1).mean(1)
    binary_variance = variance+float(np.mean((means-binary_means.repeat(2, axis=0))**2))
    return {'means': means, 'variance': variance, 'binary_means': binary_means,
            'binary_variance': binary_variance}


def belief_scores(values, model):
    values = np.asarray(values, float)
    terms = -np.sum((values[:, None, :]-model['means'][None, :, :])**2, axis=-1)/(2*model['variance'])
    binary = -np.sum((values[:, None, :]-model['binary_means'][None, :, :])**2, axis=-1)/(2*model['binary_variance'])
    return {'fact': logsumexp(terms[:, 2:], axis=1)-logsumexp(terms[:, :2], axis=1),
            'coherence': logsumexp(terms[:, [0, 2]], axis=1)-logsumexp(terms[:, [1, 3]], axis=1),
            'binary': binary[:, 1]-binary[:, 0]}


def belief_loo(features, pair_index, risk, coherence):
    """Leave an entire four-state proof out; each proof/state has equal weight."""
    features, pair_index, risk, coherence = (np.asarray(x) for x in [features, pair_index, risk, coherence])
    result = {k: [] for k in ['fact', 'coherence', 'binary']}
    for pair in np.unique(pair_index):
        keep = pair_index != pair
        model = belief_model(features[keep], pair_index[keep], risk[keep], coherence[keep])
        accuracy = {k: [] for k in result}
        for r, c in [(0, 0), (0, 1), (1, 0), (1, 1)]:
            observed = features[(pair_index == pair) & (risk == r) & (coherence == c)].mean(0)[None, :]
            scores = belief_scores(observed, model)
            for key, truth in [('fact', r), ('coherence', 1-c), ('binary', r)]:
                accuracy[key].append(float(np.sign(scores[key][0]*(2*truth-1))/2+.5))
        for key in result:
            result[key].append(float(np.mean(accuracy[key])))
    return {k: {'pair_accuracy': float(np.mean(v)), 'per_pair_accuracy': v} for k, v in result.items()}


def normalized(values):
    values = np.asarray(values, dtype=np.float64)
    return values/np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def representations(states):
    return {'interaction': normalized(states['bidirectional']-states['independent']),
            'causal': normalized(states['causal']-states['independent']),
            'raw': normalized(states['bidirectional'])}


def prototype_scores(values, positive, negative):
    positive, negative = np.asarray(positive), np.asarray(negative)
    if (positive.ndim != 2 or positive.shape != negative.shape or len(positive) < 2
            or not np.isfinite(positive).all() or not np.isfinite(negative).all()):
        raise ValueError('At least two finite matched witness pairs required')
    good, bad = positive.mean(0), negative.mean(0)
    direction = bad-good
    norm = direction @ direction
    if norm <= 1e-12:
        raise ValueError('Constraint direction is unavailable')
    return (np.asarray(values)-(good+bad)/2) @ direction/norm


def block_mask(model, prefix_length, length, mode):
    if mode not in {'bidirectional', 'independent', 'causal'} or min(prefix_length, length) < 1:
        raise ValueError('Positive prefix/span lengths and a known attention mode required')
    allowed = torch.ones((length, length), device=model.device, dtype=torch.bool)
    if mode == 'independent':
        allowed = torch.eye(length, device=model.device, dtype=torch.bool)
    elif mode == 'causal':
        allowed = allowed.tril()
    mask = torch.zeros((1, 1, length, prefix_length+length), device=model.device, dtype=model.dtype)
    mask[:, :, :, prefix_length:].masked_fill_(~allowed, torch.finfo(model.dtype).min)
    return mask


@torch.inference_mode()
def interaction_states(model, prompt, answer, units, cache=None, modes=None):
    """Read each original span with future context, preserving a causal prefix.

    Only the final causal branch survives in KV memory for subsequent spans;
    bidirectional or independent states never contaminate the historical prefix.
    An optional cache must originate from this exact prompt; only its causal
    prompt prefix is retained before starting a new answer.
    """
    if not prompt or not answer or not units:
        raise ValueError('Nonempty prompt, answer and complete spans required')
    if ([u['start'] for u in units] != [0]+[u['stop'] for u in units[:-1]]
            or units[-1]['stop'] != len(answer)
            or any(u['stop'] <= u['start'] for u in units)):
        raise ValueError('Spans must cover the complete original answer exactly')
    cache = prefill(model, prompt) if cache is None else cache
    if cache.get_seq_length() < len(prompt):
        raise ValueError('Cached source prefix is incomplete')
    cache.crop(len(prompt))
    modes = ('bidirectional', 'independent', 'causal') if modes is None else tuple(modes)
    if not modes or modes[-1] != 'causal' or len(set(modes)) != len(modes):
        raise ValueError('Distinct modes ending with the preserved causal branch required')
    blocks = {mode: [] for mode in modes}
    with attention_backend(model, 'sdpa'):
        for unit in units:
            a, b = unit['start'], unit['stop']
            prefix_length = len(prompt)+a
            for mode in blocks:
                cache.crop(prefix_length)
                hidden = model.model(input_ids=input_ids(model, answer[a:b]),
                    attention_mask=block_mask(model, prefix_length, b-a, mode),
                    past_key_values=cache, use_cache=True, return_dict=True).last_hidden_state[0]
                blocks[mode].append(hidden.float().cpu().numpy())
    return {mode: np.concatenate(values) for mode, values in blocks.items()}

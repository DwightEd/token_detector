"""Exact source-event inference with a fixed information-revision budget.

These are compatibility-conditioned diagnostics, not calibrated error probabilities.
The budget limits change from V3; it does not estimate whether a proof is correct.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logsumexp, xlogy

from myproject.measure import unit_ranges
from experiments.research.sequence import _grid, grouped_persistent_scores, witness_emissions


def proof_pack(pack, source, response, offsets, question=None):
    """Attach source-only proof metadata without changing native numerical fields."""
    from experiments.research.proofs import compile_proofs

    return {**pack, 'proofs': compile_proofs(source, response, offsets, pack['target'], question)}


@lru_cache(maxsize=16)
def _or_destination(size):
    masks = np.arange(size)
    return (masks[:, None] | masks[None, :]).ravel()


def _convolve(left, right):
    result = np.full(len(left), -np.inf)
    np.logaddexp.at(result, _or_destination(len(left)), (left[:, None]+right).ravel())
    return result


def bounded_event_probability(log_probability, budget=np.log(2.)):
    """Maximize event probability within forward KL, then minimize KL on ties.

    A finite log probability avoids underflow for rare events. Returns the
    event probability and weight of the hard conditional in a mixture with V3.
    """
    if (not np.isfinite(log_probability) or log_probability > 1e-10
            or not np.isfinite(budget) or budget < 0):
        raise ValueError('Finite log event probability <=0 and nonnegative budget required')
    log_probability = min(float(log_probability), 0.)
    p = float(np.exp(log_probability))
    if p == 1.:
        return 1., 0.
    if budget == 0:
        return p, 0.
    if -log_probability <= budget:
        return 1., 1.
    log_other = float(np.log1p(-p))

    def divergence(a):
        return (xlogy(a, a)-a*log_probability+xlogy(1-a, 1-a)
                -(1-a)*log_other-budget)

    a = float(brentq(divergence, p, 1., xtol=1e-14, rtol=1e-14))
    return a, (a-p)/(1-p)


class _MaskedBlock:
    """One original V3 block, extended by OR-hit bits for typed roles."""

    def __init__(self, values, membership, size):
        self.e = witness_emissions(values)
        self.membership = np.asarray(membership, int)
        pi, _, self.weights, self.transition = _grid(len(values), True)
        self.masks = np.arange(size)
        n, g = len(values), len(pi)
        self.forward = np.full((n, g, size, 2), -np.inf)
        self.forward[0, :, 0, 0] = np.log1p(-pi)
        self.forward[0, :, self.membership[0], 1] = np.log(pi)+self.e[0]
        t00, t01, t10, t11 = (a[:, None] for a in self.transition)
        rows = np.arange(g)[:, None]
        for t in range(1, n):
            previous, current = self.forward[t-1], self.forward[t]
            current[:, :, 0] = np.logaddexp(previous[:, :, 0]+t00, previous[:, :, 1]+t10)
            one = self.e[t]+np.logaddexp(previous[:, :, 0]+t01, previous[:, :, 1]+t11)
            destination = self.masks | self.membership[t]
            np.logaddexp.at(current[:, :, 1], (rows, destination[None, :]), one)
        terminal = logsumexp(self.forward[-1]+self.weights[:, None, None], axis=(0, 2))
        self.log_table = terminal-logsumexp(terminal)

    def marginals(self, log_accept):
        """Normalize integrated joints once, after the external circuit message."""
        g = len(self.weights)
        backward = np.broadcast_to(log_accept[None, :, None], (g, len(self.masks), 2)).copy()
        t00, t01, t10, t11 = (a[:, None] for a in self.transition)
        result = np.empty(len(self.e))
        for t in range(len(self.e)-1, -1, -1):
            joint = self.forward[t]+backward+self.weights[:, None, None]
            zero, one = logsumexp(joint[:, :, 0]), logsumexp(joint[:, :, 1])
            result[t] = np.exp(one-np.logaddexp(zero, one))
            if t:
                next_one = backward[:, self.masks | self.membership[t], 1]+self.e[t]
                previous_zero = np.logaddexp(t00+backward[:, :, 0], t01+next_one)
                previous_one = np.logaddexp(t10+backward[:, :, 0], t11+next_one)
                backward = np.stack([previous_zero, previous_one], axis=-1)
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite proof-conditioned marginal')
        return result


def _canonical_component(component, n):
    roles = []
    for tokens in component['role_tokens']:
        if not tokens or any(not isinstance(t, (int, np.integer)) or not 0 <= t < n for t in tokens):
            raise ValueError('Role tokens must index the original scored axis')
        roles.append(tuple(sorted(set(map(int, tokens)))))
    terms = []
    for term in component['repair_terms']:
        if any(not isinstance(a, (int, np.integer)) or not 0 <= a < len(roles) for a in term):
            raise ValueError('Repair term contains an invalid role')
        terms.append(frozenset(roles[int(a)] for a in term))
    if not terms:
        raise ValueError('An incompatible component must have admissible repairs')
    terms = set(terms)
    return [term for term in terms if not any(other < term for other in terms)]


def revise_scores(values, answer_index, unit_index, components, *, budget=np.log(2.),
                  max_roles=8, max_terms=256):
    """Join complete repair DNFs and revise each block-proof incidence component.

    Unknown/budget-exceeded components are neutral, without dropping tokens.
    Separate incidence components use separate budgets; they may share no block.
    """
    values, answers, units = map(np.asarray, (values, answer_index, unit_index))
    if values.ndim != 1 or answers.shape != values.shape or units.shape != values.shape:
        raise ValueError('Contrasts and original token axes must match')
    if not np.isfinite(values).all() or not np.isfinite(budget) or budget < 0:
        raise ValueError('Finite contrasts and nonnegative budget required')
    if max_roles < 1 or max_terms < 1:
        raise ValueError('Positive inference budgets required')
    base = (grouped_persistent_scores(values, answers, units, witness=True)['score']
            if len(values) else np.empty(0))
    hard, bounded = expit(base), expit(base)
    diagnostics = []
    if not len(values) or not components:
        return {'score': bounded, 'hard_score': hard, 'witness_score': base, 'components': diagnostics}
    starts, stops = unit_ranges(answers, units)
    block_id = np.repeat(np.arange(len(starts)), stops-starts)
    prepared, seen = [], set()
    for component in components:
        terms = _canonical_component(component, len(values))
        if any(not term for term in terms):
            continue
        key = tuple(sorted(tuple(sorted(term)) for term in terms))
        if key in seen:
            continue
        seen.add(key)
        tokens = sorted({t for term in terms for role in term for t in role})
        if len(set(answers[tokens])) != 1:
            raise ValueError('A proof cannot cross responses')
        prepared.append((terms, set(block_id[tokens])))
    clusters = []
    for index, (_, blocks) in enumerate(prepared):
        hits = [i for i, (_, existing) in enumerate(clusters) if blocks & existing]
        indices, merged = [index], set(blocks)
        for i in reversed(hits):
            prior, touched = clusters.pop(i); indices.extend(prior); merged.update(touched)
        clusters.append((indices, merged))
    for indices, block_set in clusters:
        terms_by_factor = [prepared[i][0] for i in indices]
        roles = sorted({role for terms in terms_by_factor for term in terms for role in term})
        metadata = {'blocks': sorted(map(int, block_set)), 'roles': len(roles),
                    'proof_components': len(indices), 'terms': sum(map(len, terms_by_factor))}
        if len(roles) > max_roles or metadata['terms'] > max_terms:
            diagnostics.append(metadata | {'status': 'unknown', 'reason': 'combined_incidence_budget'})
            continue
        size = 1 << len(roles)
        masks, membership = np.arange(size), np.zeros(len(values), int)
        for i, role in enumerate(roles):
            membership[list(role)] |= 1 << i
        role_bits = {role: 1 << i for i, role in enumerate(roles)}
        accepted = np.ones(size, bool)
        for terms in terms_by_factor:
            term_masks = [sum(role_bits[role] for role in term) for term in terms]
            accepted &= np.any([(masks & term) == term for term in term_masks], axis=0)
        blocks = sorted(block_set)
        engines = [_MaskedBlock(values[starts[b]:stops[b]], membership[starts[b]:stops[b]], size)
                   for b in blocks]
        neutral = np.full(size, -np.inf); neutral[0] = 0.
        prefix = [neutral]
        for engine in engines:
            prefix.append(_convolve(prefix[-1], engine.log_table))
        suffix = [neutral]
        for engine in reversed(engines):
            suffix.append(_convolve(engine.log_table, suffix[-1]))
        suffix.reverse()
        log_p = float(logsumexp(prefix[-1][accepted]))
        if log_p > 1e-10:
            raise ValueError('Event mass exceeds one')
        log_p = min(log_p, 0.)
        a, weight = bounded_event_probability(log_p, budget)
        for i, (b, engine) in enumerate(zip(blocks, engines, strict=True)):
            other = _convolve(prefix[i], suffix[i+1])
            accept_matrix = accepted[masks[:, None] | masks[None, :]]
            message = logsumexp(np.where(accept_matrix, other[None, :], -np.inf), axis=1)
            hard[starts[b]:stops[b]] = engine.marginals(message)
            bounded[starts[b]:stops[b]] = ((1-weight)*expit(base[starts[b]:stops[b]])
                                         +weight*hard[starts[b]:stops[b]])
        p = float(np.exp(log_p))
        kl = float(xlogy(a, a)-a*log_p+xlogy(1-a, 1-a)
                   -(1-a)*np.log1p(-p)) if p < 1. else 0.
        diagnostics.append(metadata | {'status': 'applied', 'event_log_probability': log_p,
            'revised_event_probability': a, 'hard_mixture_weight': weight, 'kl': kl,
            'absolute_probability_shift_bound': a-p})
    return {'score': bounded, 'hard_score': hard, 'witness_score': base, 'components': diagnostics}

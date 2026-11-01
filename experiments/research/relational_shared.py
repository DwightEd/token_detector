"""Exact shared-source energy conditioned on uncertain numeric relation conflicts.

This is an experimental energy model, not an entailment probability. Each
relation penalizes assignments that call BOTH endpoints of a conflicting link
clean. Supported/unknown antecedents remain neutral; no endpoint is labelled
guilty in advance. The original token contrasts and span boundaries are retained.
"""

import re

import numpy as np
from scipy.special import betaln, logsumexp

from myproject.detector import unit_ranges
from myproject.pooling import cavity_scores


def relation_factors(text, edges, target, unit_index):
    """Map original answer endpoints to scored tokens; conservatively abstain.

    Negation, conditionals and comparisons are outside the literal equality
    parser. A cross-span or unobserved antecedent retains neutral mass rather
    than renormalizing the remaining antecedents.
    """
    index = {int(t): i for i, t in enumerate(target)}
    factors, exclusions = [], {'cross_span': 0, 'unscored': 0, 'unsupported_syntax': 0}
    for edge in edges:
        # This guard only abstains; it is not an inference of claim polarity.
        breaks = list(re.finditer(r'[.!?](?=\s)|\n', text))
        left = max((m.end() for m in breaks if m.end() <= edge['start']), default=0)
        right = min((m.start() for m in breaks if m.start() >= edge['end']), default=len(text))
        context = text[left:right]
        if re.search(r"\b(?:not|no|never|neither|nor|without|if|unless|would|could|than|versus|vs)\b|"
                     r"n['’]t\b|\bat\s+(?:least|most)\b|[<>≤≥]", context, re.I):
            exclusions['unsupported_syntax'] += 1
            continue
        clauses = []
        for mention in edge['antecedents']:
            if mention['relation'] != 'conflict' or mention['weight'] <= 0:
                continue
            if not edge['target_tokens'] or not mention['tokens']:
                exclusions['unscored'] += 1
                continue
            endpoints = sorted(set(edge['target_tokens']+mention['tokens']))
            if len(endpoints) < 2 or not all(t in index for t in endpoints):
                exclusions['unscored'] += 1
                continue
            endpoints = [index[t] for t in endpoints]
            if len(set(np.asarray(unit_index)[endpoints])) != 1:
                exclusions['cross_span'] += 1
                continue
            clauses.append({'tokens': endpoints, 'weight': mention['weight']})
        if clauses:
            factors.append(clauses)
    return factors, exclusions


def constrained_scores(values, factors, max_clauses=12):
    """Exact positive-count / satisfied-clause DP for a single original span.

    phi_e(z)=1-sum_j w_ej 1[all tokens in clause_j have z=0].
    A hard clause may forbid the all-zero assignment. Returned scores are log
    responsibility odds under this working energy, not factuality probabilities.
    """
    values = np.asarray(values, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Nonempty finite contrasts required')
    if not factors:
        return cavity_scores(values)
    clauses = []
    for factor in factors:
        mass = 0.
        for clause in factor:
            tokens, weight = clause['tokens'], clause['weight']
            if (len(tokens) < 2 or len(set(tokens)) != len(tokens) or
                    any(not isinstance(t, (int, np.integer)) or t < 0 or t >= len(values) for t in tokens) or
                    not np.isfinite(weight) or weight <= 0):
                raise ValueError('Invalid clause')
            mass += weight
            clauses.append(clause)
        if not factor or mass > 1+1e-12:
            raise ValueError('Each relation needs positive conflict mass at most one')
    if len(clauses) > max_clauses:
        raise ValueError('Exact relation DP clause cap exceeded; do not drop examples')
    n, m = len(values), 1 << len(clauses)
    membership = np.zeros(n, dtype=int)
    for j, clause in enumerate(clauses):
        membership[clause['tokens']] |= 1 << j
    masks = np.arange(m)
    terminal = np.zeros(m)
    j = 0
    for factor in factors:
        unsatisfied = np.zeros(m)
        for clause in factor:
            unsatisfied += clause['weight']*((masks & (1 << j)) == 0)
            j += 1
        with np.errstate(divide='ignore'):
            terminal += np.log(np.maximum(0., 1-unsatisfied))
    forward = [np.full((1, m), -np.inf)]
    forward[0][0, 0] = 0.
    for t, value in enumerate(values):
        previous = forward[-1]
        next_values = np.full((t+2, m), -np.inf)
        next_values[:t+1] = previous
        destination = masks | membership[t]
        for k in range(t+1):
            np.logaddexp.at(next_values[k+1], destination, previous[k]+value)
        forward.append(next_values)
    counts = np.arange(n+1)
    backward = (betaln(1+counts, 1+n-counts)-betaln(1, 1))[:, None]+terminal
    evidence = float(logsumexp(forward[-1]+backward))
    scores = np.empty(n)
    for t in range(n-1, -1, -1):
        zero = backward[:t+1]
        one = values[t]+backward[1:t+2, masks | membership[t]]
        scores[t] = logsumexp(forward[t]+one)-logsumexp(forward[t]+zero)
        backward = np.logaddexp(zero, one)
    if not np.isfinite(scores).all() or not np.isfinite(evidence):
        raise ValueError('Degenerate relation posterior')
    return {'score': scores, 'cavity_log_odds': scores-values, 'log_evidence': evidence}


def grouped_relational_scores(values, answer_index, unit_index, factors):
    """Use exactly the baseline spans, including neutral no-certificate spans."""
    values, answer_index, unit_index = map(np.asarray, (values, answer_index, unit_index))
    if values.ndim != 1 or answer_index.shape != values.shape or unit_index.shape != values.shape:
        raise ValueError('Mismatched token axes')
    starts, stops = unit_ranges(answer_index, unit_index)
    score = np.empty(len(values))
    assigned = 0
    for start, stop in zip(starts, stops, strict=True):
        local = []
        for factor in factors:
            endpoints = [t for clause in factor for t in clause['tokens']]
            if all(start <= t < stop for t in endpoints):
                local.append([{'tokens': [int(t-start) for t in c['tokens']], 'weight': c['weight']}
                              for c in factor])
                assigned += 1
        score[start:stop] = constrained_scores(values[start:stop], local)['score']
    if assigned != len(factors):
        raise ValueError('A relation crosses original spans or is out of bounds')
    return score

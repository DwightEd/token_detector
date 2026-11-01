"""Exact binary-chain inference conditioned on nonlocal conflict clauses."""

import numpy as np

from myproject.graph import chain_log_odds


def constraint_log_odds(emission, penalties, clauses):
    """Condition a chain on at least one risky token in every conflict clause.

    Clauses are position sets, not labels or a choice of the guilty endpoint.
    The forward/backward state tracks the set of already satisfied clauses.
    Cost is O(T * 2**K); K is capped at 12 rather than silently approximated.
    These are model marginals, not calibrated factuality probabilities.
    """
    emission = np.asarray(emission, dtype=float)
    penalties = np.asarray(penalties, dtype=float)
    if (emission.ndim != 1 or not len(emission) or penalties.shape != (len(emission)-1,)
            or not np.isfinite(emission).all() or not np.isfinite(penalties).all()
            or np.any(penalties < 0)):
        raise ValueError('Finite nonempty emissions and nonnegative adjacent penalties required')
    clauses = sorted({tuple(sorted(set(c))) for c in clauses})
    if len(clauses) > 12:
        raise ValueError('Exact inference supports at most 12 conflict clauses')
    if any(len(c) < 2 or any(not isinstance(t, (int, np.integer)) or t < 0 or t >= len(emission)
                            for t in c) for c in clauses):
        raise ValueError('Each conflict needs at least two valid token positions')
    if not clauses:
        return chain_log_odds(emission, penalties)
    membership = np.zeros(len(emission), dtype=np.int64)
    for bit, clause in enumerate(clauses):
        membership[list(clause)] |= 1 << bit
    masks = np.arange(1 << len(clauses))
    forward = np.full((len(emission), len(masks), 2), -np.inf)
    forward[0, 0, 0] = 0.
    forward[0, membership[0], 1] = emission[0]
    for t in range(1, len(emission)):
        for previous in [0, 1]:
            for current in [0, 1]:
                destination = masks | membership[t] if current else masks
                weight = forward[t-1, :, previous]+emission[t]*current
                weight = weight-penalties[t-1]*(previous != current)
                np.logaddexp.at(forward[t, :, current], destination, weight)
    backward = np.full_like(forward, -np.inf)
    backward[-1, -1, :] = 0.
    for t in range(len(emission)-2, -1, -1):
        for previous in [0, 1]:
            for current in [0, 1]:
                destination = masks | membership[t+1] if current else masks
                weight = backward[t+1, destination, current]+emission[t+1]*current
                weight = weight-penalties[t]*(previous != current)
                backward[t, :, previous] = np.logaddexp(backward[t, :, previous], weight)
    joint = forward+backward
    return np.logaddexp.reduce(joint[:, :, 1], axis=1)-np.logaddexp.reduce(joint[:, :, 0], axis=1)

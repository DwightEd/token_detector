"""Exact Beta-mixture pooling of unchanged per-token source log contrasts."""

import numpy as np
from scipy.special import betaln, logsumexp


def _values(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Expected a nonempty finite token contrast vector')
    return values


def _parameters(alpha, beta):
    if not np.isfinite([alpha, beta]).all() or min(alpha, beta) <= 0:
        raise ValueError('Beta parameters must be positive and finite')


def _axes(contrasts, *indices):
    contrasts = _values(contrasts)
    axes = [np.asarray(index) for index in indices]
    if any(axis.ndim != 1 or len(axis) != len(contrasts) for axis in axes):
        raise ValueError('Token metadata must be one-dimensional and match contrast length')
    return contrasts, *axes


def forward(values):
    """Log Bernstein coefficients: sum over assignments with k absent-world tokens."""
    values = _values(values)
    n = len(values)
    result = np.full((n + 1, n + 1), -np.inf)
    result[0, 0] = 0.
    for t, value in enumerate(values):
        result[t+1, :t+2] = np.logaddexp(
            np.r_[result[t, :t+1], -np.inf],
            np.r_[-np.inf, result[t, :t+1] + value])
    return result


def _terminal(n, alpha, beta):
    count = np.arange(n + 1)
    return betaln(alpha + count, beta + n - count) - betaln(alpha, beta)


def cavity_scores(values, alpha=1., beta=1.):
    """Exact token log-responsibility odds; not a calibrated factual-error posterior.

    The own contrast is never averaged. Other observed factors provide its
    cavity prior; autoregressive histories remain the original fixed histories.
    """
    _parameters(alpha, beta)
    values = _values(values)
    n = len(values)
    prefix = forward(values)
    suffix = np.full_like(prefix, -np.inf)
    suffix[n] = _terminal(n, alpha, beta)
    for t in range(n-1, -1, -1):
        suffix[t, :t+1] = np.logaddexp(suffix[t+1, :t+1],
                                                  values[t] + suffix[t+1, 1:t+2])
    correction = np.empty(n)
    for t in range(n):
        correction[t] = (logsumexp(prefix[t, :t+1] + suffix[t+1, 1:t+2])
                         - logsumexp(prefix[t, :t+1] + suffix[t+1, :t+1]))
    return {'score': values + correction, 'cavity_log_odds': correction,
            'log_evidence': float(suffix[0, 0])}


def grouped_scores(contrasts, answer_index, unit_index, alpha=1., beta=1.):
    """Keep original row order; states never cross an answer or span boundary."""
    from myproject.measure import unit_ranges

    contrasts, answer_index, unit_index = _axes(contrasts, answer_index, unit_index)
    starts, stops = unit_ranges(answer_index, unit_index)
    score, correction, evidence = np.empty_like(contrasts), np.empty_like(contrasts), []
    for start, stop in zip(starts, stops):
        result = cavity_scores(contrasts[start:stop], alpha, beta)
        score[start:stop] = result['score']
        correction[start:stop] = result['cavity_log_odds']
        evidence.append(result['log_evidence'])
    return {'score': score, 'cavity_log_odds': correction, 'log_evidence': np.asarray(evidence)}

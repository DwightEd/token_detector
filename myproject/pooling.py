"""Exact Beta-mixture pooling of unchanged per-token source log contrasts."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import betaln, digamma, logsumexp


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


def prior_objective(log_parameters, coefficients, weights):
    """Source-weighted composite evidence plus fixed independent Gamma(2,1) priors.

    The MAP density is with respect to alpha,beta. Log parameters are only an
    optimization coordinate; no transformation Jacobian is added to this target.
    """
    alpha, beta = np.exp(log_parameters)
    _parameters(alpha, beta)
    objective = float(np.log(alpha) - alpha + np.log(beta) - beta)
    gradient = np.array([1/alpha - 1, 1/beta - 1], dtype=float)
    for coefficient, weight in zip(coefficients, weights, strict=True):
        n = len(coefficient) - 1
        count = np.arange(n + 1)
        terms = coefficient + _terminal(n, alpha, beta)
        evidence = logsumexp(terms)
        posterior = np.exp(terms - evidence)
        objective += weight * evidence
        common = -digamma(alpha + beta + n) + digamma(alpha + beta)
        gradient += weight * np.array([
            posterior @ digamma(alpha + count) - digamma(alpha) + common,
            posterior @ digamma(beta + n - count) - digamma(beta) + common])
    return -objective, -gradient * np.array([alpha, beta])


def fit_prior(spans, weights):
    """Fit only two population hyperparameters; no labels or neural optimization."""
    coefficients = [forward(values)[-1] for values in spans]
    weights = np.asarray(weights, dtype=float)
    if (not coefficients or weights.shape != (len(coefficients),)
            or not np.isfinite(weights).all() or np.any(weights <= 0)):
        raise ValueError('Each nonempty span needs a positive finite population weight')
    attempts = []
    for start in [(1., 1.), (.5, 2.), (2., .5)]:
        result = minimize(prior_objective, np.log(start), args=(coefficients, weights),
                          method='BFGS', jac=True, options={'gtol': 1e-8, 'maxiter': 500})
        value, gradient = prior_objective(result.x, coefficients, weights)
        attempts.append({'start': list(start), 'parameters': np.exp(result.x).tolist(),
                         'objective': float(-value), 'gradient_max': float(np.max(abs(gradient))),
                         'success': bool(result.success), 'iterations': int(result.nit),
                         'message': str(result.message)})
    accepted = [a for a in attempts if np.isfinite(a['objective']) and a['gradient_max'] <= 1e-6]
    if not accepted:
        raise RuntimeError(f'No prior fit met the frozen gradient tolerance: {attempts}')
    best = max(accepted, key=lambda a: a['objective'])
    alpha, beta = best['parameters']
    return {'alpha': alpha, 'beta': beta, 'mean': alpha/(alpha+beta),
            'concentration': alpha+beta, 'objective': best['objective'],
            'population_weight': float(weights.sum()), 'attempts': attempts}


def grouped_scores(contrasts, answer_index, unit_index, alpha=1., beta=1.):
    """Keep original row order; states never cross an answer or span boundary."""
    from myproject.detector import unit_ranges

    contrasts, answer_index, unit_index = _axes(contrasts, answer_index, unit_index)
    starts, stops = unit_ranges(answer_index, unit_index)
    score, correction, evidence = np.empty_like(contrasts), np.empty_like(contrasts), []
    for start, stop in zip(starts, stops):
        result = cavity_scores(contrasts[start:stop], alpha, beta)
        score[start:stop] = result['score']
        correction[start:stop] = result['cavity_log_odds']
        evidence.append(result['log_evidence'])
    return {'score': score, 'cavity_log_odds': correction, 'log_evidence': np.asarray(evidence)}


def population_spans(contrasts, answer_index, unit_index, source_index):
    """One unit of fit weight/source, shared over its answers and their spans."""
    from myproject.detector import unit_ranges

    contrasts, answer_index, unit_index, source_index = _axes(
        contrasts, answer_index, unit_index, source_index)
    for answer in np.unique(answer_index):
        if len(np.unique(source_index[answer_index == answer])) != 1:
            raise ValueError('Each global answer identity must belong to one source')
    starts, stops = unit_ranges(answer_index, unit_index)
    spans, weights = [], []
    for start, stop in zip(starts, stops):
        source, answer = source_index[start], answer_index[start]
        if not np.all(source_index[start:stop] == source):
            raise ValueError('A span cannot cross a source')
        answers = np.unique(answer_index[source_index == source])
        span_count = np.sum(answer_index[starts] == answer)
        spans.append(contrasts[start:stop])
        weights.append(1 / (len(answers) * span_count))
    return spans, np.asarray(weights)

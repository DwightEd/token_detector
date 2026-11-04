"""Unlabeled population-prior fitting for research comparisons."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import digamma, logsumexp
from experiments.research.pooling import _parameters, _axes, _terminal, forward


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



def population_spans(contrasts, answer_index, unit_index, source_index):
    """One unit of fit weight/source, shared over its answers and their spans."""
    from myproject.measure import unit_ranges

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
